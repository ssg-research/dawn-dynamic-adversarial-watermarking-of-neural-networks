import argparse
import configparser
import copy
import datetime
import os
import pickle
import random
from typing import Dict, Any, List, Tuple

import torch
import torch.nn as nn
import torch.nn.utils.prune as prune
import torch.utils.data as data
import torchvision as tv
from tqdm import tqdm

import config_helper
import logger
import models
import score

random.seed(42)

log = logger.Logger(prefix=">>>")

class SimpleDataset(data.Dataset):
    def __init__(self, dataset: List[Tuple[Any, int]]) -> None:
        self.data, self.labels = zip(*dataset)
        self.count = len(self.labels)

    def __getitem__(self, index: int) -> (Any, int):
        return self.data[index], self.labels[index]

    def __len__(self) -> int:
        return self.count


def main(config: configparser.ConfigParser, model_path: str, watermark_path: str) -> None:
    #  Setup model architecture and load model from file.
    model = setup_model(
        config["DEFAULT"]["model_architecture"],
        model_path,
        int(config["DEFAULT"]["number_of_classes"]))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device=device)

    #  Load test set and transform it.
    test_set = download_test(
        config["DEFAULT"]["dataset_name"],
        config["DEFAULT"]["test_save_path"],
        int(config["DEFAULT"]["input_size"])
    )
    test_set = data.DataLoader(test_set, batch_size=int(config["DEFAULT"]["batch_size"]))

    watermark_set = load_file(watermark_path)

    pruning_save_path = config["DEFAULT"]["pruning_save_path"]
    if not os.path.exists(pruning_save_path):
        log.warn(pruning_save_path + " does not exist. Creating...")
        os.makedirs(pruning_save_path)
        log.info(pruning_save_path + " Created.")

    pruning_results = prune_model(model, test_set, watermark_set, int(config["DEFAULT"]["number_of_classes"]))

    date = datetime.datetime.today().strftime('%Y-%m-%d')
    path_body = pruning_save_path + config["DEFAULT"]["model_name"]

    save_scores(
        pruning_results,
        path_body + date)


def download_test(dataset_name: str, victim_data_path: str, input_size: int) -> data.Dataset:
    mean = [0.5, 0.5, 0.5]
    std = [0.5, 0.5, 0.5]

    if dataset_name == "MNIST":
        dataset = tv.datasets.MNIST
        transformations = tv.transforms.Compose([
            tv.transforms.ToTensor(),
            tv.transforms.Normalize(mean, std)
        ])
    elif dataset_name == "CIFAR10":
        dataset = tv.datasets.CIFAR10
        transformations = tv.transforms.Compose([
            tv.transforms.Resize(input_size),
            tv.transforms.CenterCrop(input_size),
            tv.transforms.ToTensor(),
            tv.transforms.Normalize(mean, std)
        ])
    else:
        log.error("MNIST and CIFAR10 are the only supported datasets at the moment. Throwing...")
        raise ValueError(dataset_name)

    test_set = dataset(victim_data_path, train=False, transform=transformations, download=True)

    log.info("Test samples: {}\nSaved in: {}".format(dataset_name, len(test_set), victim_data_path))
    return test_set


def setup_model(model_architecture: str, model_path: str, number_of_classes: int) -> nn.Module:
    available_models = {
        "MNIST_L5": models.MNIST_L5,
        "CIFAR10_BASE": models.CIFAR10_BASE
    }

    model = available_models[model_architecture]()

    if model is None:
        log.error("Incorrect model architecture specified or architecture not available.")
        raise ValueError(model_architecture)

    models.load_state(model, model_path)

    return model


def load_file(file_path: str) -> List[Tuple]:
    with open(file_path, "rb") as f:
        return pickle.load(f)


def prune_model(model: nn.Module, test_set: data.DataLoader, watermark_set: List, number_of_classes: int) -> Dict[float, Dict[str, Any]]:
    #  Pruning experiment with multiple pruning levels
    pruning_levels = [0.01, 0.05, 0.1, 0.25, 0.4, 0.5, 0.75, 0.9]
    pruning_results = {}

    log.info("Accuracy before pruning:")
    _ = test_model(model, test_set, number_of_classes)
    _ = test_watermark(model, watermark_set)

    for level in pruning_levels:
        model_local = copy.deepcopy(model)
        # parameters_to_prune = model_local.parameters()

        for _, module in model_local.named_modules():
            if isinstance(module, torch.nn.Conv2d):
                prune.l1_unstructured(module, name='weight', amount=level)
            elif isinstance(module, torch.nn.Linear):
                prune.l1_unstructured(module, name="weight", amount=level)

        log.info("Testing with pruning level {}.".format(level))
        test_float_score, test_dict_score = test_model(model_local, test_set, number_of_classes)
        watermark_float_score = test_watermark(model_local, watermark_set)
        pruning_results[level] = {
            "test": (test_float_score, test_dict_score),
            "watermark": watermark_float_score
        }

    return pruning_results


def test_model(model: nn.Module, test_set: data.DataLoader, number_of_classes: int) -> Tuple[score.FloatScore, score.DictScore]:
    """Test the model on the test dataset."""
    # model.eval is used for ImageNet models, batchnorm or dropout layers will work in eval mode.
    model.eval()

    def test_average() -> score.FloatScore:
        correct = 0
        total = 0

        with torch.set_grad_enabled(False):
            for (inputs, yreal) in tqdm(test_set, unit="images", desc="Testing model (average)", leave=True, ascii=True):
                inputs, yreal = inputs.cuda(), yreal.cuda()

                ypred = model(inputs)
                _, predicted = torch.max(ypred.data, 1)

                total += yreal.size(0)
                correct += (predicted == yreal).sum().item()

        accuracy = 100 * correct / total
        log.info("Accuracy of the network on the {} test images (average): {}".format(total, accuracy))
        with open('epoch_logs.txt', 'a+') as file:
            file.write('Test Acc: {}\n'.format(accuracy))
        return score.FloatScore(accuracy)

    def test_per_class() -> score.DictScore:
        class_correct = list(0. for _ in range(number_of_classes))
        class_total = list(0. for _ in range(number_of_classes))
        total = 0

        with torch.no_grad():
            for (inputs, yreal) in tqdm(test_set, unit="images", desc="Testing model (per class)", leave=True, ascii=True):
                inputs, yreal = inputs.cuda(), yreal.cuda()

                total += yreal.size(0)

                ypred = model(inputs)
                _, predicted = torch.max(ypred, 1)
                c = (predicted == yreal).squeeze()
                for i in range(yreal.shape[0]):
                    label = yreal[i]
                    class_correct[label] += c[i].item()
                    class_total[label] += 1

        log.info("Accuracy of the network on the {} test images (per-class):".format(total))

        per_class_accuracy = {}
        for i in range(number_of_classes):
            accuracy = 100 * class_correct[i] / (class_total[i] + 0.0001)
            per_class_accuracy[i] = accuracy
            print('Accuracy of %5s : %2d %%' % (
                i, accuracy))

        return score.DictScore(per_class_accuracy)

    return test_average(), test_per_class()


def test_watermark(model: nn.Module, watermark_set: List) -> score.FloatScore:
    model.eval()

    correct = 0
    total = 0
    with torch.no_grad():
        for (inputs, yreal) in tqdm(watermark_set, unit="images", desc="Testing watermark (average)", leave=True, ascii=True):
            inputs, yreal = inputs.cuda(), yreal.cuda()

            ypred = model(inputs)
            _, predicted = torch.max(ypred.data, 1)
            total += yreal.size(0)
            correct += (predicted == yreal).sum().item()

    accuracy = 100 * correct / total
    log.info("Accuracy of the network on the {} test images (average): {}".format(total, accuracy))
    return score.FloatScore(accuracy)


def save_scores(pruning_results: Dict[float, Dict[str, Any]], file_path: str) -> None:
    with open(file_path + '.pkl', 'wb') as f:
        pickle.dump(pruning_results, f, pickle.HIGHEST_PROTOCOL)


def handle_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config_file",
        type=str,
        default=None,
        help="Configuration file for the experiment.")
    parser.add_argument(
        "--watermark",
        type=str,
        default=None,
        help="Path to the saved watermark Loader.")
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Path to the saved model.")
    args = parser.parse_args()

    if args.config_file is None:
        raise ValueError("Configuration file must be provided.")

    if args.watermark is None:
        raise ValueError("Watermark path must be provided.")

    if args.config_file is None:
        raise ValueError("Model path must be provided.")

    return args


if __name__ == "__main__":
    args = handle_args()
    config = config_helper.load_config(args.config_file)
    watermark_path = args.watermark
    model_path = args.model

    config_helper.print_config(config)
    log.info("Model path: {}.".format(model_path))
    log.info("Watermark path: {}".format(watermark_path))

    main(config, model_path, watermark_path)
