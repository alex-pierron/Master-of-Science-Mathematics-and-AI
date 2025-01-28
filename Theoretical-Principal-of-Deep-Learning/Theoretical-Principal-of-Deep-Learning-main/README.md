Analysis of a research paper: "The sample complexity of pattern classification with neural networks: the size of the weights is more important than the size of the network" by Peter Bartlett (1998)

## Authors: Alex Pierron & Matisse Roche
  This work is part of our Master II "Mathematics and Artificial Intelligence" done at Paris-Saclay University and CentraleSupélec.

### Date: February 16, 2024

## Abstract:

This document presents an analysis of the research paper "The sample complexity of pattern classification with neural networks: the size of the weights is more important than the size of the network" by Peter Bartlett. The paper studies the generalization properties of neural networks and provides theoretical bounds on the error rate. It is one of the first paper achieving a computable error bound for neural networks under certain asumptions.

The document begins by providing a brief overview of supervised learning and the challenges of establishing generalization guarantees. It then introduces the notion of fat-shattering dimension as an alternative to VC dimension for measuring the complexity of a hypothesis class.

The main results of the paper are presented, including Theorem 2 and Theorem 28, which provide bounds on the error rate of neural networks under certain conditions. The proof of Theorem 2 is also presented in detail.

The document then discusses the implications of the paper's results and the limitations of the theoretical bounds. It also presents the results of numerical simulations that we conducted to evaluate the performance of the bounds in practice.

Finally, the document concludes by summarizing the main contributions of the paper and suggesting directions for future research.

### Keywords: supervised learning, neural networks, generalization, fat-shattering dimension, VC dimension, error rate, theoretical bounds, numerical simulations


## Python modules:
numpy, scipy, matplotlib, seaborn, torch, torchvision, pandas, tqdm, pillow-simd (optionnal)
```
conda create -n myenv python=3.9
conda activate myenv
conda install numpy scipy matplotlib seaborn pytorch torchvision scikit-learn tensorboard opencv pandas tqdm pillow-simd
```

## Prerequisites:
Optimization, Statistics and Probabilities \
Basic knowledge of machine learning \
Familiarity with neural networks

## How to use this repo:
Codes used to produce our simulations is entirely available in this repo. The code is supposed to run directly when you run the notebook properly. Do not run the notebook if you just want to see the results, the complete computation of the notebook takes several hours.

This document can be used as a reference for understanding  and illustrating the theoretical foundations of neural network generalization given in the article treated. It can also be used for further research on this topic.

