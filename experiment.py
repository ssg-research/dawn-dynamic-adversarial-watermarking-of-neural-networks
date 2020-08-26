from typing import List, Dict, Tuple, NamedTuple
import torch
import torch.nn as nn
from tqdm import tqdm
import logger
import score
import random

random.seed(42)

log = logger.Logger(prefix=">>>")

class Experiment(object):
    def __init__(self, environment):
        self.training_ops = environment.training_ops
        self.test_ops = environment.test_ops
        self.watermark_ops = environment.watermark_ops


class ExperimentTraining(Experiment):
    def __init__(self, environment: NamedTuple) -> None:
        super(ExperimentTraining, self).__init__(environment)

    def train_victim(self, log_interval: int = None) -> Tuple[nn.Module, Dict[str, List[score.Score]], List[Tuple[torch.Tensor, int]]]:
        if (self.training_ops.victim_model_architecture  == 'RN34'):
            # Run training with optimization scheduler when models are resnet
            victim_model, scores, ground_truth_logit = self.train_victim_with_scheduler(log_interval=1000, learning_rate = 0.1)
        else:
            epochs = self.training_ops.epochs
            criterion = self.training_ops.criterion
            optimizer = self.training_ops.optimizer
            victim_model = self.training_ops.victim_model
            use_cuda = self.training_ops.use_cuda
            training_loader = self.training_ops.training_loader

            lr = 0.001
            optimizer = optimizer(victim_model.parameters(), lr=lr)
            if use_cuda:
                victim_model = victim_model.cuda()

            scores = {
                "test_average": [],
                "test_per_class": [],
                "loss": []
            }

            ground_truth_logit = []

            for epoch in range(epochs):
                log.info("Epoch {}/{}".format(epoch + 1, epochs))
                running_loss = 0.0
                for i, (inputs, yreal) in enumerate(tqdm(training_loader, unit="images", desc="Training victim model", leave=True, ascii=True), 0):
                    if use_cuda:
                        inputs, yreal = inputs.cuda(), yreal.cuda()

                    optimizer.zero_grad()

                    if epoch == 100:
                        for g in optimizer.param_groups:
                            g["lr"] = 0.0005
                    if epoch == 200:
                        for g in optimizer.param_groups:
                            g["lr"] = 0.0003

                    ypred = victim_model(inputs)
                    loss = criterion(ypred, yreal)
                    loss.backward()
                    optimizer.step()

                    if epoch == (epochs - 1):
                        ground_truth_logit.append(ypred)

                    # book-keeping
                    if log_interval is not None:
                        running_loss += loss.item()
                        if i % log_interval == (log_interval - 1):
                            log.info('[%d/%d, %5d] loss: %.3f' %
                                  (epoch + 1, epochs, i + 1, running_loss / log_interval))
                            running_loss = 0.0

                scores["loss"].append(score.FloatScore(running_loss))

                if (epoch + 1) % 5 == 0:
                    log.info("Testing at {}".format(epoch + 1))
                    test_average, test_per_class = self.test_model(victim_model)
                    scores["test_average"].append(test_average)
                    scores["test_per_class"].append(test_per_class)

            log.info('Finished training victim.')
        return victim_model, scores, ground_truth_logit


    def train_attacker(self, log_interval: int = None) -> (nn.Module, Dict[str, List[score.Score]]):
        """Try to integrate the watermark into the procedure.
        fine-tunning and embedding are repeated in turns.
        Here attacker is just a victim with a watermark."""
        if (self.training_ops.attacker_model_architecture  == 'RN34'):
            # run training with optimization scheduler when models are resnet
            attacker_model, scores, watermark_logit, ground_truth_logit, watermark = self.train_attacker_with_scheduler(log_interval=1000, learning_rate = 0.1)
        else:
            epochs = self.training_ops.epochs
            criterion = self.watermark_ops.criterion
            optimizer = self.watermark_ops.optimizer
            use_cuda = self.watermark_ops.use_cuda
            training_loader = self.training_ops.training_loader
            watermark_loader = self.watermark_ops.watermark_loader
            attacker_model = self.watermark_ops.attacker_model
            weight_decay = self.watermark_ops.weight_decay

            optimizer = optimizer(attacker_model.parameters(), lr=0.001, weight_decay=weight_decay)
            if use_cuda:
                attacker_model = attacker_model.cuda()

            scores = {
                "test_average": [],
                "test_per_class": [],
                "test_watermark": [],
                "loss": []
            }

            watermark_logit = []
            ground_truth_logit = []

            for epoch in range(epochs):
                log.info("Epoch {}/{}".format(epoch + 1, epochs))

                running_loss = 0.0

                attacker_model.train()
                for i, (inputs, yreal) in enumerate(tqdm(training_loader, unit="images", desc="Training attacker model (regular)", leave=True, ascii=True), 0):
                    if use_cuda:
                        inputs, yreal = inputs.cuda(), yreal.cuda()

                    optimizer.zero_grad()
                    ypred = attacker_model(inputs)

                    loss = criterion(ypred, yreal)
                    loss.backward()
                    optimizer.step()

                    if epoch == (epochs - 1):
                        ground_truth_logit.append(ypred)

                    if log_interval is not None:
                        running_loss += loss.item()
                        if i % log_interval == (log_interval - 1):
                            log.info('[%d/%d, %5d] loss: %.3f' %
                                  (epoch + 1, epochs, i + 1, running_loss / log_interval))
                            running_loss = 0.0

                scores["loss"].append(score.FloatScore(running_loss))

                for i, (inputs, yreal) in enumerate(tqdm(watermark_loader, unit="images", desc="Training attacker model (watermark)", leave=True, ascii=True), 0):
                    if use_cuda:
                        inputs, yreal = inputs.cuda(), yreal.cuda()

                    optimizer.zero_grad()

                    ypred = attacker_model(inputs)
                    loss = criterion(ypred, yreal)
                    loss.backward()
                    optimizer.step()

                    if epoch == (epochs - 1):
                        watermark_logit.append(ypred)

                # book-keeping
                if (epoch + 1) % 5 == 0:
                    log.info("Testing at {}".format(epoch + 1))
                    avg_score, per_class_score = self.test_model(attacker_model)
                    watermark_score = self.test_watermark(attacker_model)
                    scores["test_average"].append(avg_score)
                    scores["test_per_class"].append(per_class_score)
                    scores["test_watermark"].append(watermark_score)

            log.info('Finished training attacker.')

            ground_truth_logit += watermark_logit

            watermark = [(inputs, yreal) for (inputs, yreal) in watermark_loader]
        return attacker_model, scores, watermark_logit, ground_truth_logit, watermark

    def train_victim_with_scheduler(self, log_interval: int = None, learning_rate: float = 0.1) -> Tuple[nn.Module, Dict[str, List[score.Score]], List[torch.Tensor]]:

        use_cuda = self.watermark_ops.use_cuda
        training_loader = self.training_ops.training_loader
        victim_model = self.training_ops.victim_model
        epochs = self.training_ops.epochs
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(victim_model.parameters(), lr=learning_rate, momentum=0.5, weight_decay=5e-4)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=60, gamma=0.1)

        scores = {
            "test_average": [],
            "test_per_class": [],
            "test_watermark": [],
            "loss": []
        }

        ground_truth_logit = []

        for epoch in range(epochs):
            log.info("Epoch {}/{}".format(epoch + 1, epochs))
            scheduler.step(None)
            victim_model.train()
            running_loss = 0.0

            for i, (inputs, yreal) in enumerate(tqdm(training_loader, unit="images", desc="Training victim model", leave=True, ascii=True), 0):
                if use_cuda:
                    inputs, yreal, victim_model = inputs.cuda(), yreal.cuda(), victim_model.cuda()

                optimizer.zero_grad()

                with torch.set_grad_enabled(True):
                    ypred = victim_model(inputs)
                    loss = criterion(ypred, yreal)
                    loss.backward()
                    optimizer.step()

                if epoch == (epochs - 1):
                    ground_truth_logit.append(ypred)

                # book-keeping
                if log_interval is not None:
                    running_loss += loss.item()
                    if i % log_interval == (log_interval - 1):
                        log.info('[%d/%d, %5d] loss: %.3f' %
                              (epoch + 1, epochs, i + 1, running_loss / log_interval))
                        running_loss = 0.0

            scores["loss"].append(score.FloatScore(running_loss))

            if (epoch + 1) % 5 == 0:
                log.info("Testing at {}".format(epoch + 1))
                avg_score, per_class_score = self.test_model(victim_model)
                watermark_score = self.test_watermark(victim_model)
                scores["test_average"].append(avg_score)
                scores["test_per_class"].append(per_class_score)
                scores["test_watermark"].append(watermark_score)

        log.info('Finished training victim.')
        return victim_model, scores, ground_truth_logit

    def train_attacker_with_scheduler(self, log_interval: int = None, learning_rate: float = 0.1) -> (nn.Module, Dict[str, List[score.Score]]):

        use_cuda = self.watermark_ops.use_cuda
        training_loader = self.training_ops.training_loader
        watermark_loader = self.watermark_ops.watermark_loader
        attacker_model = self.watermark_ops.attacker_model
        epochs = self.training_ops.epochs
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(attacker_model.parameters(), lr=learning_rate, momentum=0.5, weight_decay=5e-4)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=60, gamma=0.1)

        scores = {
            "test_average": [],
            "test_per_class": [],
            "test_watermark": [],
            "loss": []
        }

        watermark_logit = []
        ground_truth_logit = []

        for epoch in range(epochs):
            log.info("Epoch {}/{}".format(epoch + 1, epochs))
            scheduler.step()
            attacker_model.train()
            running_loss = 0.0
            running_corrects = 0

            for inputs, yreal in tqdm(training_loader, unit="images", desc="Training attacker model (regular)", leave=True, ascii=True):
                if use_cuda:
                    inputs, yreal, attacker_model = inputs.cuda(), yreal.cuda(), attacker_model.cuda()

                optimizer.zero_grad()

                with torch.set_grad_enabled(True):
                    ypred = attacker_model(inputs)
                    loss = criterion(ypred, yreal)
                    _, preds = torch.max(ypred, 1)
                    loss.backward()
                    optimizer.step()

                if epoch == (epochs - 1):
                    ground_truth_logit.append(ypred)

                # book-keeping
                running_corrects += torch.sum(preds == yreal.data)
                running_loss += loss.item() * inputs.size(0)

            epoch_loss = running_loss / len(training_loader.dataset)
            epoch_acc = 100* running_corrects.double() / len(training_loader.dataset)
            with open('epoch_logs.txt', 'a+') as file:
                file.write('Epoch: {} Train Acc: {}\n'.format(epoch, epoch_acc))

            scores["loss"].append(score.FloatScore(running_loss))

            for inputs, yreal in tqdm(watermark_loader, unit="images", desc="Training attacker model (watermark)", leave=True, ascii=True):
                if use_cuda:
                    inputs, yreal, attacker_model = inputs.cuda(), yreal.cuda(), attacker_model.cuda()

                optimizer.zero_grad()

                with torch.set_grad_enabled(True):
                    ypred = attacker_model(inputs)
                    loss = criterion(ypred, yreal)
                    loss.backward()
                    optimizer.step()

                if epoch == (epochs - 1):
                    watermark_logit.append(ypred)

            if (epoch + 1) % 5 == 0:
                log.info("Testing at {}".format(epoch + 1))
                avg_score, per_class_score = self.test_model(attacker_model)
                watermark_score = self.test_watermark(attacker_model)
                scores["test_average"].append(avg_score)
                scores["test_per_class"].append(per_class_score)
                scores["test_watermark"].append(watermark_score)

        log.info('Finished training attacker.')

        ground_truth_logit += watermark_logit

        watermark = [(inputs, yreal) for (inputs, yreal) in watermark_loader]
        return attacker_model, scores, watermark_logit, ground_truth_logit, watermark

    def test_model(self, model: nn.Module) -> Tuple[score.FloatScore, score.DictScore]:
        """Test the model on the test dataset."""
        # model.eval is used for ImageNet models, batchnorm or dropout layers will work in eval mode.
        model.eval()
        use_cuda = self.test_ops.use_cuda
        test_loader = self.test_ops.test_loader
        number_of_classes = self.test_ops.number_of_classes

        if use_cuda:
            model = model.cuda()

        def test_average() -> score.FloatScore:
            correct = 0
            total = 0
            with torch.set_grad_enabled(False):
                for (inputs, yreal) in tqdm(test_loader, unit="images", desc="Testing model (average)", leave=True, ascii=True):
                    if use_cuda:
                        inputs, yreal = inputs.cuda(), yreal.cuda()

                    ypred = model(inputs)
                    _, predicted = torch.max(ypred.data, 1)

                    total += yreal.size(0)
                    correct += (predicted == yreal).sum().item()

            accuracy = 100 * correct / total
            log.info("Accuracy of the network on the {} test images (average): {}".format(total, accuracy))
            with open('epoch_logs_' + self.training_ops.dataset_name + '.txt', 'a+') as file:
                file.write('Test Acc: {}\n'.format(accuracy))
            return score.FloatScore(accuracy)

        def test_per_class() -> score.DictScore:
            class_correct = list(0. for _ in range(number_of_classes))
            class_total = list(0. for _ in range(number_of_classes))
            total = 0

            with torch.no_grad():
                for (inputs, yreal) in tqdm(test_loader, unit="images", desc="Testing model (per class)", leave=True, ascii=True):
                    if use_cuda:
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

    def test_watermark(self, model: nn.Module) -> score.FloatScore:
        model.eval()
        use_cuda = self.watermark_ops.use_cuda
        watermark_loader = self.watermark_ops.watermark_loader

        correct = 0
        total = 0
        with torch.no_grad():
            for (inputs, yreal) in tqdm(watermark_loader, unit="images", desc="Testing watermark (average)", leave=True, ascii=True):
                if use_cuda:
                    inputs, yreal = inputs.cuda(), yreal.cuda()

                ypred = model(inputs)
                _, predicted = torch.max(ypred.data, 1)

                total += yreal.size(0)
                correct += (predicted == yreal).sum().item()

        accuracy = 100 * correct / total
        log.info("Accuracy of the network on the {} test images (average): {}".format(total, accuracy))
        return score.FloatScore(accuracy)
