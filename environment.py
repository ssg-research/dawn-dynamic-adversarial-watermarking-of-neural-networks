import configparser
from pathlib import Path
import random
from collections import namedtuple
from typing import List, Tuple, Any, Dict, NamedTuple
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import torchvision as tv
import logger
import models


class SimpleDataset(data.Dataset):
    def __init__(self, dataset: List[Tuple[Any, int]]) -> None:
        self.data, self.labels = zip(*dataset)
        self.count = len(self.labels)

    def __getitem__(self, index: int) -> Tuple[Any, int]:
        return self.data[index], self.labels[index]

    def __len__(self) -> int:
        return self.count


log = logger.Logger(prefix=">>>")


def prepare_environment(config: configparser.ConfigParser) -> NamedTuple:
    model_save_path = config["VICTIM"]["model_save_path"]
    create_dir_if_doesnt_exist(model_save_path)

    scores_save_path = config["DEFAULT"]["scores_save_path"]
    create_dir_if_doesnt_exist(scores_save_path)

    # DOWNLOAD DATASETS
    victim_dataset = config["DEFAULT"]["dataset_name"]
    victim_data_path = config["VICTIM"]["victim_dataset_save_path"]
    watermark_data_path = config["ATTACKER"]["watermark_dataset_save_path"]
    watermark_dataset = config["ATTACKER"]["watermark_set"]
    batch_size = int(config["DEFAULT"]["batch_size"])
    number_of_classes = int(config["DEFAULT"]["number_of_classes"])
    force_greyscale = config["ATTACKER"].getboolean("force_greyscale")
    normalize_with_imagenet_vals = config["ATTACKER"].getboolean("normalize_with_imagenet_vals")

    input_size = int(config["DEFAULT"]["input_size"])

    training_transforms, watermark_transforms = setup_transformations(victim_dataset, watermark_dataset, force_greyscale, normalize_with_imagenet_vals, input_size)
    # DOWNLOAD VICTIM DATASET
    train_set, test_set = download_victim(victim_dataset, victim_data_path, training_transforms)

    # DOWNLOAD WATERMARK DATASET
    watermark_size = int(config["ATTACKER"]["watermark_size"])

    # SUBCLASS TRAINING SET IF THE SETS ARE THE SAME, OTHERWISE JUST TAKE SAMPLES
    if victim_dataset == watermark_dataset:
        watermark_set, train_set = construct_watermark_set(train_set, watermark_size, number_of_classes, partition=True)
    else:
        watermark_set = download_watermark(watermark_dataset, watermark_data_path, watermark_transforms)
        watermark_set, _ = construct_watermark_set(watermark_set, watermark_size, number_of_classes, partition=False)

    train_set = data.DataLoader(train_set, batch_size=batch_size, shuffle=True)
    test_set = data.DataLoader(test_set, batch_size=batch_size)
    watermark_set = data.DataLoader(watermark_set, batch_size=batch_size)

    # SETUP VICTIM MODEL
    victim_retrain = config["VICTIM"].getboolean("retrain")
    victim_model_path = model_save_path + config["VICTIM"]["model_name"]
    victim_model = setup_model(
        retrain=victim_retrain,
        resume=config["VICTIM"].getboolean("resume"),
        model_architecture=config["VICTIM"]["model_architecture"],
        model_path=victim_model_path,
        number_of_classes=number_of_classes
    )

    # SETUP ATTACKER MODEL
    attacker_retrain = config["ATTACKER"].getboolean("retrain")
    attacker_model_path = model_save_path + config["ATTACKER"]["model_name"]
    attacker_model = setup_model(
        retrain=attacker_retrain,
        resume=config["ATTACKER"].getboolean("resume"),
        model_architecture=config["ATTACKER"]["model_architecture"],
        model_path=attacker_model_path,
        number_of_classes=number_of_classes
    )

    # SETUP TRAINING PROCEDURE
    criterion = nn.functional.cross_entropy
    optimizer = optim.Adam
    TrainingOps = namedtuple("TrainingOps",
        [
            "epochs",
            "criterion",
            "optimizer",
            "victim_model",
            "dataset_name",
            "use_cuda",
            "training_loader",
            "resume_from_checkpoint_path",
            "victim_model_architecture",
            "attacker_model_architecture"
        ])

    training_ops = TrainingOps(
        int(config["DEFAULT"]["epochs"]),
        criterion,
        optimizer,
        victim_model,
        config["DEFAULT"]["dataset_name"],
        config["DEFAULT"]["use_cuda"],
        train_set,
        get_with_default(config, "STRATEGY", "resume_from_checkpoint_path", str),
        config["VICTIM"]["model_architecture"],
        config["ATTACKER"]["model_architecture"]
    )

    # SETUP TEST PROCEDURE
    TestOps = namedtuple("TestOps", ["test_loader", "use_cuda", "batch_size", "number_of_classes"])
    test_ops = TestOps(
        test_set,
        config["DEFAULT"]["use_cuda"],
        batch_size,
        number_of_classes
    )

    # SETUP WATERMARK EMBEDDING
    WatermarkOps = namedtuple("WatermarkOps",
        [
            "epochs",
            "criterion",
            "optimizer",
            "attacker_model",
            "use_cuda",
            "training_loader",
            "watermark_loader",
            "number_of_classes",
            "weight_decay",
            "watermark_data_path",
            "watermark_transforms"
        ])

    watermark_ops = WatermarkOps(
        int(config["DEFAULT"]["epochs"]),
        criterion,
        optimizer,
        attacker_model,
        config["DEFAULT"]["use_cuda"],
        train_set,
        watermark_set,
        number_of_classes,
        float(config["ATTACKER"]["decay"]),
        watermark_data_path,
        watermark_transforms
    )

    # SETUP EXPERIMENT ENVIRONMENT
    Environment = namedtuple("Environment",
        [
            "victim_retrain",
            "attacker_retrain",
            "victim_model_path",
            "attacker_model_path",
            "training_ops",
            "test_ops",
            "watermark_ops",
        ])
    return Environment(
        victim_retrain,
        attacker_retrain,
        victim_model_path,
        attacker_model_path,
        training_ops,
        test_ops,
        watermark_ops)


def get_with_default(config: configparser.ConfigParser, section: str, name: str, type_, default=None):
    if config.has_option(section, name):
        return type_(config.get(section, name))
    else:
        return default


def create_dir_if_doesnt_exist(path_to_dir: str) -> None:
    path = Path(path_to_dir)
    if not path.exists():
        log.warn(path_to_dir + " does not exist. Creating...")
        path.mkdir(parents=True, exist_ok=True)
        log.info(path_to_dir + " Created.")


def setup_transformations(training_set: str, watermark_set: str, force_greyscale: bool, normalize_with_imagenet_vals: bool, input_size: int) -> Tuple[tv.transforms.Compose, tv.transforms.Compose]:
    mean = [0.5, 0.5, 0.5]
    std = [0.5, 0.5, 0.5]
    if normalize_with_imagenet_vals:
        mean =  [0.485, 0.456, 0.406]
        std  =  [0.229, 0.224, 0.225]
    train_transforms = {
        "MNIST": {
            "train": tv.transforms.Compose([
                tv.transforms.ToTensor(),
                tv.transforms.Normalize(mean, std)
            ]),
            "val": tv.transforms.Compose([
                tv.transforms.ToTensor(),
                tv.transforms.Normalize(mean, std)
            ])
        },
        "CIFAR10": {
            'train': tv.transforms.Compose([
                tv.transforms.Resize(input_size),
                tv.transforms.CenterCrop(input_size),
                tv.transforms.ToTensor(),
                tv.transforms.Normalize(mean, std)
            ]),
            'val': tv.transforms.Compose([
                tv.transforms.Resize(input_size),
                tv.transforms.CenterCrop(input_size),
                tv.transforms.ToTensor(),
                tv.transforms.Normalize(mean, std)
            ]),
        }
    }

    greyscale = [tv.transforms.Grayscale()] if force_greyscale else []
    watermark_transforms = {
        "MNIST": tv.transforms.Compose([
            tv.transforms.ToTensor(),
            tv.transforms.Normalize(mean, std)
        ]),
        "CIFAR10": tv.transforms.Compose(greyscale + [
            tv.transforms.Resize(input_size),
            tv.transforms.CenterCrop(input_size),
            tv.transforms.ToTensor(),
            tv.transforms.Normalize(mean, std)
        ])
    }

    train_transform = train_transforms[training_set]
    if train_transform is None:
        log.error("Specified training set transform is not available.")
        raise ValueError(training_set)

    watermark_transform = watermark_transforms[watermark_set]
    if watermark_transform is None:
        log.error("Specified watermark set transform is not available.")
        raise ValueError(watermark_set)

    return train_transform, watermark_transform


def download_victim(victim_dataset_name: str, victim_data_path: str, transformations: Dict[str, tv.transforms.Compose]) -> Tuple[data.Dataset, data.Dataset]:
    if victim_dataset_name == "MNIST":
        dataset = tv.datasets.MNIST
    elif victim_dataset_name == "CIFAR10":
        dataset = tv.datasets.CIFAR10
    else:
        log.error("MNIST and CIFAR10 are the only supported victim datasets at the moment. Throwing...")
        raise ValueError(victim_dataset_name)
    train_set = dataset(victim_data_path, train=True, transform=transformations["train"], download=True)
    test_set = dataset(victim_data_path, train=False, transform=transformations["val"], download=True)

    log.info("Training ({}) samples: {}\nTest samples: {}\nSaved in: {}".format(victim_dataset_name, len(train_set), len(test_set), victim_data_path))
    return train_set, test_set


def download_watermark(watermark_dataset_name: str, watermark_data_path: str, transformations: tv.transforms.Compose) -> data.Dataset:
    if watermark_dataset_name == "MNIST":
        dataset = tv.datasets.MNIST
    elif watermark_dataset_name == "CIFAR10":
        dataset = tv.datasets.CIFAR10
    else:
        log.error("MNIST and CIFAR10 are the only supported attacker datasets at the moment. Throwing...")
        raise ValueError(watermark_dataset_name)

    watermark_set = dataset(watermark_data_path, train=False, transform=transformations, download=True)
    log.info("Watermark ({}) samples: {}\nSaved in: {}".format(watermark_dataset_name, len(watermark_set), watermark_data_path))

    return watermark_set


def construct_watermark_set(watermark_set: data.Dataset, watermark_size: int, number_of_classes: int, partition: bool) -> Tuple[data.Dataset, data.Dataset]:
    len_ = watermark_set.__len__()
    watermark, train = data.dataset.random_split(watermark_set, (watermark_size, len_ - watermark_size))
    log.info("Split set into: {} and {}".format(len(watermark), len(train)))

    watermark = SimpleDataset([(img, another_label(label, number_of_classes)) for img, label in watermark])
    if partition:
        return watermark, train
    else:
        return watermark, None


def setup_model(retrain: bool, resume: bool, model_architecture: str, model_path: str, number_of_classes: int) -> nn.Module:
    available_models = {
        "MNIST_L2": models.MNIST_L2,
        "MNIST_L2_DRP03": models.MNIST_L2_DRP03,
        "MNIST_L2_DRP05": models.MNIST_L2_DRP05,
        "MNIST_L5": models.MNIST_L5,
        "MNIST_L5_DRP03": models.MNIST_L5_DRP03,
        "MNIST_L5_DRP05": models.MNIST_L5_DRP05,
        "CIFAR10_BASE": models.CIFAR10_BASE,
        "CIFAR10_BASE_DRP03": models.CIFAR10_BASE_DRP03,
        "CIFAR10_BASE_DRP05": models.CIFAR10_BASE_DRP05,
        "RN34" : tv.models.resnet34,
        "VGG16" : tv.models.vgg16,
        "DN121_DRP03": tv.models.densenet121,
        "DN121_DRP05": tv.models.densenet121
    }

    # variables in pre-trained ImageNet models are model-specific.
    if "RN34" in model_architecture:
        model = available_models[model_architecture](pretrained=True)
        n_features = model.fc.in_features
        model.fc = nn.Linear(n_features, number_of_classes)
    elif "VGG16" in model_architecture:
        model = available_models[model_architecture](pretrained=True)
        n_features = model.classifier[6].in_features
        model.classifier[6] = nn.Linear(n_features, number_of_classes)
    elif "DN121_DRP03" in model_architecture:
        model = available_models[model_architecture](pretrained=True, drop_rate=0.3)
    elif "DN121_DRP05" in model_architecture:
        model = available_models[model_architecture](pretrained=True, drop_rate=0.5)
    else:
        model = available_models[model_architecture]()

    if model is None:
        log.error("Incorrect model architecture specified or architecture not available.")
        raise ValueError(model_architecture)

    if not retrain:
        models.load_state(model, model_path)

    if resume:
        models.load_state(model, model_path)

    return model


def another_label(real_label: int, number_of_classes: int) -> int:
    new_label = real_label
    while new_label == real_label:
        new_label = random.randint(0, number_of_classes - 1)
    return new_label
