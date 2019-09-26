# Addressing Failure Prediction by Learning Model Confidence
 [Charles Corbière](https://chcorbi.github.io/),  [Nicolas Thome](http://cedric.cnam.fr/~thomen/), [Avner Bar-Hen](https://ab-h.github.io/), [Matthieu Cord](http://webia.lip6.fr/~cord/), [Patrick Pérez](https://ptrckprz.github.io/) \
*International Conference on Neural Information Processing Systems (NeurIPS), 2019*

![](./teaser.png)

If you find this code useful for your research, please cite our paper:

```
@inproceedings{corbiere2019ConfidNet,
  title={Addressing Failure Prediction by Learning Model Confidence},
  author={Corbi{\'e}re, Charles and Thome, Nicolas and Bar-Hen, Avner and Cord, Mathieu and P{\'e}rez, Patrick},
  booktitle={NeurIPS},
  year={2019}
}
```

## Abstract
Assessing reliably the confidence of a deep neural net and predicting its failures is of primary importance for the practical deployment of these models. In this paper, we propose a new target criterion for model confidence, corresponding to the *True Class Probability* (TCP).We show how using the TCP is more suited than relying on the classic *Maximum Class Probability* (MCP). We provide in addition theoretical guarantees for TCP in the context of failure prediction. Since the true class is by essence unknown at test time, we propose to learn TCP criterion on the training set, introducing a specific learning scheme adapted to this context. Extensive experiments are conducted for validating the relevance of the proposed approach. We study various network architectures, small and large scale datasets for image classification and semantic segmentation. We show that our approach consistently outperforms several strong methods, from MCP to Bayesian uncertainty, as well as
recent approaches specifically designed for failure prediction.

## Installation
1. Clone the repo:
```bash
$ git clone https://github.com/valeoai/ConfidNet
$ cd ConfidNet
```

2. Install this repository and the dependencies using pip:
```bash
$ pip install -e ConfidNet
```

With this, you can edit the ConfidNet code on the fly and import function 
and classes of ConfidNet in other project as well.

3. Optional. To uninstall this package, run:
```bash
$ pip uninstall ConfidNet
```

You can take a look at the [Dockerfile](./Dockerfile) if you are uncertain about steps to install this project.

#### Datasets

MNIST, SVHN, CIFAR-10 and CIFAR-100 datasets are managed by Pytorch dataloader. First time you run a script, the dataloader will download the dataset in ```confidnet/data/datasetname-data```.

## Running the code

### Training
```
python3 train.py -c confs/your_config_file.yaml 
```

### Testing
```
python3 test.py -c confs/your_config_file.yaml -e NUM_EPOCHS

```