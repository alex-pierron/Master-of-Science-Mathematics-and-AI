import datetime
import time

import einops
import numpy as np
import torch
import torch.nn.functional as F


@torch.no_grad()
def evaluate_blip2(model, data_loader, fabric):
    model.eval()

    fabric.print("Computing features for evaluation...")
    start_time = time.time()

    tar_img_feats = []
    query_feats = []
    captions = []
    pair_ids = []

    for ref_img, tar_feat, caption, pair_id, *_ in data_loader:
        pair_ids.extend(pair_id.cpu().numpy().tolist())
        captions.extend(caption)

        device = ref_img.device
        """
        if self.train_vit:
            sample = {"image": ref_img, "text_input": None}
            ref_img_embs = self.model.extract_features(sample,mode="image")
        else:
            with torch.no_grad():
                sample = {"image": ref_img, "text_input": None}
                ref_img_embs = self.model.extract_features(sample,mode="image")
        """
        # Encode the target image
        tar_feat = tar_feat.to(device)
        tar_img_feat = F.normalize(tar_feat, dim=-1)
        #text = self.txt_processors['train'](caption).to(device)
        sample = {"image": ref_img, "text_input": caption}
        query_feat = model.model.extract_features(sample)
        query_feat  = query_feat.multimodal_embeds
        query_feat = F.normalize(query_feat[:, 0, :], dim=-1)
        query_feats.append(query_feat.cpu())

        # Encode the target image
        tar_img_feats.append(tar_feat.cpu())


    pair_ids = torch.tensor(pair_ids, dtype=torch.long)
    query_feats = torch.cat(query_feats, dim=0)
    tar_img_feats = torch.cat(tar_img_feats, dim=0)

    query_feats = F.normalize(query_feats, dim=-1)
    tar_img_feats = F.normalize(tar_img_feats, dim=-1)

    ref_img_ids = [data_loader.dataset.pairid2ref[pair_id.item()] for pair_id in pair_ids]
    tar_img_ids = [data_loader.dataset.pairid2tar[pair_id.item()] for pair_id in pair_ids]
    ref_img_ids = torch.tensor(ref_img_ids, dtype=torch.long)
    tar_img_ids = torch.tensor(tar_img_ids, dtype=torch.long)

    if fabric.world_size > 1:
        # Gather tensors from every process
        query_feats = fabric.all_gather(query_feats)
        tar_img_feats = fabric.all_gather(tar_img_feats)
        ref_img_ids = fabric.all_gather(ref_img_ids)
        tar_img_ids = fabric.all_gather(tar_img_ids)

        query_feats = einops.rearrange(query_feats, "d b e -> (d b) e")
        tar_img_feats = einops.rearrange(tar_img_feats, "d b e -> (d b) e")
        ref_img_ids = einops.rearrange(ref_img_ids, "d b -> (d b)")
        tar_img_ids = einops.rearrange(tar_img_ids, "d b -> (d b)")

    if fabric.global_rank == 0:
        sim_q2t = (query_feats @ tar_img_feats.t()).cpu().numpy()
        # Add zeros where ref_img_id == tar_img_id
        for i in range(len(ref_img_ids)):
            for j in range(len(tar_img_ids)):
                if i < len(sim_q2t) and j < len(sim_q2t[i]):
                    if ref_img_ids[i] == tar_img_ids[j]:
                        sim_q2t[i][j] = -10

        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print("Evaluation time {}".format(total_time_str))

        eval_result = eval_recall(sim_q2t)
        fabric.print(eval_result)

        fabric.log_dict(
            {
                "val/R1": eval_result["R1"],
                "val/R5": eval_result["R5"],
                "val/R10": eval_result["R10"],
                "val/R_mean": eval_result["R_mean"],
            }
        )

    fabric.barrier()


@torch.no_grad()
def eval_recall(scores_q2t):
    # Query->Target
    ranks = np.zeros(scores_q2t.shape[0])

    for index, score in enumerate(scores_q2t):
        inds = np.argsort(score)[::-1]
        ranks[index] = np.where(inds == index)[0][0]

    # Compute metrics
    tr1 = 100.0 * len(np.where(ranks < 1)[0]) / len(ranks)  # type: ignore
    tr5 = 100.0 * len(np.where(ranks < 5)[0]) / len(ranks)
    tr10 = 100.0 * len(np.where(ranks < 10)[0]) / len(ranks)
    tr50 = 100.0 * len(np.where(ranks < 50)[0]) / len(ranks)

    tr_mean = (tr1 + tr5 + tr10) / 3

    eval_result = {
        "R1": round(tr1, 4),
        "R5": round(tr5, 4),
        "R10": round(tr10, 4),
        "R50": round(tr50, 4),
        "R_mean": round(tr_mean, 4),
    }
    return eval_result
