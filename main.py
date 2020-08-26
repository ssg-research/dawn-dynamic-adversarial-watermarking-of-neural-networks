# Author: Sebastian Szyller sebastian.szyller@aalto.fi Samuel Marchal samuel.marchal@aalto.fi
# Copyright 2019 Secure Systems Group, Aalto University, https://ssg.aalto.fi
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import configparser
import datetime
import pickle
from typing import List, Dict, Tuple

import torch

import config_helper
import environment
import experiment
import logger
import models
import score

log = logger.Logger(prefix=">>>")


def main(config: configparser.ConfigParser) -> None:
    env = environment.prepare_environment(config)
    agent = experiment.ExperimentTraining(env)

    if env.victim_retrain:
        victim_model, scores, ground_truth_logit = agent.train_victim(log_interval=1000)
        date = datetime.datetime.today().strftime('%Y-%m-%d')
        path_body = config["DEFAULT"]["scores_save_path"] + config["VICTIM"]["model_name"]

        save_scores(
            scores,
            path_body + date)

        save_queries(
            ground_truth_logit,
            path_body + "_ground_truth_" + date)

        models.save_state(victim_model, env.victim_model_path)
    else:
        victim_model = env.training_ops.victim_model

    _ = agent.test_model(victim_model)

    experiment_training(
        env,
        agent,
        config["DEFAULT"]["scores_save_path"] +
        config["ATTACKER"]["model_name"])

def experiment_training(env, training_agent: experiment.ExperimentTraining, path_body: str) -> None:
    if env.attacker_retrain:
        attacker_model, scores, watermark_logit, ground_truth_logit, full_watermark = training_agent.train_attacker(log_interval=1000)
        date = datetime.datetime.today().strftime('%Y-%m-%d')

        save_scores(
            scores,
            path_body + date)

        save_queries(
            watermark_logit,
            path_body + "_watermark_" + date)

        save_queries(
            ground_truth_logit,
            path_body + "_ground_truth_" + date)

        save_full_watermark(
            full_watermark,
            path_body + "_watermark_full_" + date)

        models.save_state(attacker_model, env.attacker_model_path)
    else:
        attacker_model = env.training_ops.attacker_model

    _ = training_agent.test_model(attacker_model)  # test set to to check valid accuracy
    _ = training_agent.test_watermark(attacker_model)  # watermarking set to check persistence


def save_scores(scores_dict: Dict[str, List[score.Score]], file_path: str) -> None:
    with open(file_path + '.pkl', 'wb') as f:
        pickle.dump(scores_dict, f, pickle.HIGHEST_PROTOCOL)


def save_queries(watermark: List[Tuple[torch.Tensor, int]], file_path: str) -> None:
    with open(file_path + '.pkl', 'wb') as f:
        pickle.dump(watermark, f, pickle.HIGHEST_PROTOCOL)


def save_full_watermark(full_watermark: List[Tuple[torch.FloatTensor, int]], file_path: str) -> None:
    with open(file_path + '.pkl', 'wb') as f:
        pickle.dump(full_watermark, f, pickle.HIGHEST_PROTOCOL)


def handle_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config_file",
        type=str,
        default=None,
        help="Configuration file for the experiment.")

    args = parser.parse_args()
    if args.config_file is None:
        raise ValueError("Configuration file must be provided.")

    return args


if __name__ == "__main__":
    args = handle_args()
    config = config_helper.load_config(args.config_file)
    config_helper.print_config(config)

    main(config)
