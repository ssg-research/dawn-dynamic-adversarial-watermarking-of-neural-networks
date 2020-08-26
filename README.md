# DAWN: Dynamic Adversarial Watermarking of Neural Networks

This repo contains code that allows you to reproduce experiments for the watermarking scheme presented in *DAWN: Dynamic Adversarial Watermarking of Neural Networks*.

[Link to the arxiv version](https://arxiv.org/abs/1906.00830)

**DISCLAIMER:** provided source code does **NOT** include experiments with model extraction attacks.
Usage and distribution of such code is potentially harmful and should be done separately at their authors' disclosure.
Reach out ([PRADA](https://arxiv.org/abs/1805.02628) and [Knockoff Nets](https://arxiv.org/abs/1812.02766)).

Additionally, to simplify reproduction, experiments are slimmed down to MNIST and CIFAR10 datasets.
GTSRB, Caltech, CUBS and ImageNet require manual downloads - while not difficult, it adds unnecessary complexity.
However, in this repo we provide:

- complete config files for experiments
- victim models (MNIST and CIFAR10)
- PRADA-stolen models (MNIST and CIFAR10)
- watermarks to reproduce included experiments

## Environment Setup

You need to have a working `conda` setup. Then you can a new env using provided `environment.yml` file:

```
conda env create -f environment.yml
```

and activate it:

```
conda activate dawn` or `source activate dawn
```

## Running Code

Note that all scripts should be run from the root directory of this project.

### Perfect Attacker

To run experiments that simulate attacker with perfect knowledge about the victim model, use `main.py`:

```
usage: main.py [-h] [--config_file CONFIG_FILE]

optional arguments:
  -h, --help                    show this help message and exit
  --config_file CONFIG_FILE     Configuration file for the experiment.
```

There are many config files in `configurations/perfect` that cover vanilla training as well as with regularization to prevent watermark embedding. You can run any of the experiments by specifying its corresponding `.ini` file, e.g.:

```
python3 main.py --config_file configurations/perfect/mnist-to-mnist-ws250-l5.ini
```

Note that you we provide trained victim models for MNIST and CIFAR10.


### Watermark Removal

#### Detection

Evaluation of the detection method is self contained in `watermark_detection.ipynb`.
It covers MNIST, CIFAR10 and a ResNet variant of CIFAR10.
We provide data from the logit layers (ground truth and watermarks) to conduct these experiments.

#### Noisy Verification

To run the experiments that evaluate resilience of the verification process to added perturbation use the `noisy_verification.py` script.
We provide two attacker/surrogate models obtained using PRADA attacks as well as their corresponding watermarks.

```
usage: noisy_verification.py [-h] [--config_file CONFIG_FILE]
                             [--watermark WATERMARK] [--model MODEL]

optional arguments:
  -h, --help                    show this help message and exit
  --config_file CONFIG_FILE     Configuration file for the experiment.
  --watermark WATERMARK         Path to the saved watermark Loader.
  --model MODEL                 Path to the saved model.
```

CIFAR10:

```
python3 noisy_verification.py \
--config_file configurations/noisy-verification/verification-cifar-prada-single-1000.ini \
--watermark data/scores/attacker_cifar_prada_l5_single_1000_watermark.pkl \
--model data/models/attacker_cifar_prada.pt
```

MNIST:

```
python3 noisy_verification.py \
--config_file configurations/noisy-verification/verification-mnist-prada-single-10.ini \
--watermark data/scores/attacker_mnist_prada_l5_single_10_watermark.pkl \
--model data/models/attacker_mnist_prada.pt
```

Results will printed in the terminal and saved in `data/scores/verification`.

#### Pruning

To run the experiments that evaluate watermark's resilience to pruning use the `prune.py` script.
We provide two attacker/surrogate models obtained using PRADA attacks as well as their corresponding watermarks.

```
usage: prune.py [-h] [--config_file CONFIG_FILE] [--watermark WATERMARK]
                [--model MODEL]

optional arguments:
  -h, --help                    show this help message and exit
  --config_file CONFIG_FILE     Configuration file for the experiment.
  --watermark WATERMARK         Path to the saved watermark Loader.
  --model MODEL                 Path to the saved model.
```

CIFAR10:

```
python3 prune.py \
--config_file configurations/pruning/pruning-mnist-prada-single-10.ini \
--watermark data/scores/attacker_mnist_prada_l5_single_10_watermark.pkl \
--model data/models/attacker_mnist_prada.pt
```

MNIST:

```
python3 prune.py \
--config_file configurations/pruning/pruning-cifar-prada-single-1000.ini \
--watermark data/scores/attacker_cifar_prada_l5_single_1000_watermark.pkl \
--model data/models/attacker_cifar_prada.pt
```

Results will printed in the terminal and saved in `data/scores/pruning`.

#### Mapping Function

Experiments with the mapping function for the MNIST dataset are self-contained in `mapping.ipynb`.
