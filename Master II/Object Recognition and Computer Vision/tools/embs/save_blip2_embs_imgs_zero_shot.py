import os
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from tqdm.auto import tqdm

script_path = os.path.abspath(__file__)
script_dir = os.path.dirname(script_path)
project_root = os.path.abspath(os.path.join(script_dir, "..", ".."))
sys.path.append(project_root)

from src.data.embs_blip2 import ImageDataset
from src.model.blip2_embs import blip2_embs

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_blip2_config(model="base"):
    config = dict()
    if model == "pretrain":
        config[
            "model_type"
        ] = "pretrain"
        config["batch_size_train"] = 32
        config["batch_size_test"] = 16
        config["vit_grad_ckpt"] = True
        config["vit_ckpt_layer"] = 4
        config["init_lr"] = 1e-5
        config["negative_all_rank"] = False
        
    elif model == "pretrain_vitL":
        config[
            "model_type"
        ] = "pretrain_vitL"
        config["vit"] = "large"
        config["batch_size_train"] = 16
        config["batch_size_test"] = 32
        config["vit_grad_ckpt"] = True
        config["vit_ckpt_layer"] = 12
        config["init_lr"] = 5e-6

        config["image_size"] = 224
        config["queue_size"] = 57600
        config["alpha"] = 0.4
        config["k_test"] = 256
        config["negative_all_rank"] = True

    elif model == "coco":
        config["model_type"] = "coco"
        config["vit"] = "large"
        config["batch_size_train"] = 16
        config["batch_size_test"] = 32
        config["image_size"] = 384
        config["queue_size"] = 57600
        config["negative_all_rank"] = True

    return config


@torch.no_grad()
def main(args):
    dataset = ImageDataset(
        image_dir=args.image_dir,
        img_ext=args.img_ext,
        save_dir=args.save_dir,
    )

    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        pin_memory=True,
        num_workers=args.num_workers,
    )

    print("Creating model")
    config = get_blip2_config(args.model_type)
    model = blip2_embs(
        model_type=config["model_type"],
        negative_all_rank=config["negative_all_rank"],
    )

    model = model.to(device)
    model.eval()

    for imgs, video_ids in tqdm(loader):
        imgs = imgs.to(device)
        img_embs = model.model.extract_features({"image": imgs}, mode="image")
        img_feats = img_embs.image_embeds_proj.cpu()

        for img_feat, video_id in zip(img_feats, video_ids):
            img_feat = F.normalize(img_feat, dim=-1)
            torch.save(img_feat, args.save_dir / f"{video_id}.pth")
            torch.save(img_feat, args.save_dir / f"{video_id}.pth")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--image_dir", type=Path, required=True, help="Path to image directory"
    )
    parser.add_argument("--save_dir", type=Path)
    parser.add_argument("--img_ext", type=str, default="png")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument(
        "--model_type", type=str, default="pretrain", choices=["pretrain", "pretrain_vitL","coco"]
    )
    args = parser.parse_args()

    subdirectories = [subdir for subdir in args.image_dir.iterdir() if subdir.is_dir()]
    if len(subdirectories) == 0:
        args.save_dir = args.image_dir.parent / f"blip-embs256-{args.model_type}"
        args.save_dir.mkdir(exist_ok=True)
        main(args)
    else:
        for subdir in subdirectories:
            args.image_dir = subdir
            args.save_dir = (
                subdir.parent.parent / f"blip-embs256-{args.model_type}" / subdir.name
            )
            args.save_dir.mkdir(exist_ok=True, parents=True)
            main(args)