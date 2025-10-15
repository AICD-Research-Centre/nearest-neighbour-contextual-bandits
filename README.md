# Nearest-Neighbour Contextual Bandits

This repository contains code and experiments for the paper:

**A Hierarchical Nearest Neighbour Approach to Contextual Bandits** (TMLR 2025)

## Overview
This project implements and evaluates state-of-the-art nearest neighbour algorithms for contextual bandit problems, as described in the attached paper. The codebase includes algorithm implementations, dataset wrappers, and experiment notebook to reproduce the results.

## Paper
If you use this code, please cite the following paper:

> Stephen Pasteris, Madeleine Dwyer, Chris Hicks, and Vasilios Mavroudis. "A Hierarchical Nearest Neighbour Approach to Contextual Bandits." Transactions on Machine Learning Research (TMLR), 2025.

## Installation
Clone the repository and install dependencies:

```bash
git clone https://github.com/AICD-Research-Centre/nearest-neighbour-contextual-bandits.git
cd nearest-neighbour-contextual-bandits
python -m venv nncb_env
source nncb_env/bin/activate
pip install -r requirements.txt
```

## Usage
Run experiments using the provided jupyter notebook (experiments_notebook.ipynb).

## Algorithms
We implemented 5 SOTA contextual bandit algorithms which utilise nearest neighbour/similarity information.

### Hierarchical Nearest Neighbour (HNN)
The algorithm introduced in this paper.

Additionally, we use the CBNN algorithm from: Stephen Pasteris, Chris Hicks, and Vasilios Mavroudis. "Nearest neighbour with bandit feedback." NeurIPS, 2023.

### Contextual Bandit with Nearest Neighbour (NN)
The algorithm from: Stephen Pasteris, Chris Hicks, and Vasilios Mavroudis. "Nearest neighbour with bandit feedback." NeurIPS, 2023.

In order to implement this algorithm, we also have implemented the ternary tree rebalancing (`nncb/algorithms/pasteris/tt_rebalancing.py`) from: Kiminori Matsuzaki and Akimasa Morihata. "Mathematical engineering technical reports balanced ternary-tree representation of binary trees and balancing algorithms." 2008.

Additionally we implemented Navigating Nets (`nncb/utils/navigating_net.py`), from: Robert Krauthgamer and James R. Lee. "Navigating nets: simple algorithms for proximity search." In ACM-SIAM Symposium on Discrete Algorithms, 2004 

To switch to navigating nets use, change the following line from `nncb/algorithms/pasteris/[cbnn|hnn].py`:
```python
DEFAULT_LEAF_LIST_TYPE = LeafList
```
to:
```python
DEFAULT_LEAF_LIST_TYPE = NavigatingNetList
```

### K-Nearest Neighbours with UCB (KNN_UCB) and with KL divergence and UCB (KNN_KL_UCB)
The algorithms from: Henry W. J. Reeve, Joseph Charles Mellor, and Gavin Brown. "The k-nearest neighbour ucb algorithm for multi-armed bandits with covariates." ArXiv, abs/1803.00316, 2018.

### Contextual Bandits with Similarity Information (Slivkins)
The algorithm from this reference: Aleksandrs Slivkins. "Contextual bandits with similarity information." ArXiv, abs/0907.3986, 2009.

Requires a base algorithm, for which we used EXP3 (as suggested in their paper), which is implemented in `nncb/algorithms/slivkins/exp3.py`, from this reference: Peter Auer, Nicolò Cesa-Bianchi, Yoav Freund, and Robert E. Schapire. "The nonstochastic multiarmed bandit problem." SIAM J. Comput., 32:48–77, 2002b.

## Datasets
Dataset wrappers are included in `nncb/dataset_wrappers/`. See the notebook for usage details.
Dataset files need to be added to `nncb/dataset_files/`. 
Our experiments were on intrusion detection/firewall problems, we used:

### UCI Firewall dataset
This dataset was used as a stochastic setting test of the algorithms.

The dataset can be found here: https://archive.ics.uci.edu/dataset/542/internet+firewall+data
Our code expects this to be stored in `nncb/dataset_files/firewall_dataset.arff`.

### CICIDS2017 dataset
This dataset was used as an adversarial setting test of the algorithm, specifically using a subset of the datasets Wednesday data, containing a change in DoS attack types.

The dataset can be found here: https://www.unb.ca/cic/datasets/ids-2017.html
Our code expects this to be stored in `nncb/dataset_files/CICIDS2017/MachineLearningCVE/`. Specifically, the experiments from our paper expect the `Wednesday-workingHours.pcap_ISCX.csv` file to be located there.

## Reproducing Results
Results from experiments are saved in the `results/` directory. To reproduce, follow instructions in the notebook.

## License
See `LICENSE` for details.