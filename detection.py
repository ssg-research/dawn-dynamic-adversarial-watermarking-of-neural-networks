# Authors: Sebastian Szyller, Buse Gul Atli
# Copyright 2020 Secure Systems Group, Aalto University, https://ssg.aalto.fi
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
import pickle

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import decomposition
from sklearn import pipeline
from sklearn import preprocessing
from sklearn import cluster
from sklearn import metrics

sns.set()

from typing import Dict, Any


def dummy_data():
    import numpy as np
    a = np.random.normal(0, 0.1, (100, 10))
    b = np.random.normal(0, 0.1, (100, 10))
    return np.vstack((a, b))


def main(logits_file_path: str) -> None:
    print(logits_file_path)  # load file here
    data = dummy_data()

    kmeans = cluster.KMeans(n_clusters=2)
    fun = pipeline.make_pipeline(preprocessing.StandardScaler(), decomposition.PCA(n_components=3))
    data_transformed = fun.fit_transform(data)

    ypred = kmeans.fit_predict(data_transformed)
    print("> silhouette score: {}".format(metrics.silhouette_score(data_transformed, ypred)))

    fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2)

    first = ax1.scatter(data_transformed[:100, 0], data_transformed[:100, 1], c=data_transformed[:100, 2], marker="o", label="Legitimate")
    second = ax1.scatter(data_transformed[100:200, 0], data_transformed[100:200, 1], c=data_transformed[100:200, 2], marker="x", label="Watermarked")
    ax1.figure.colorbar(second, ax=ax1, label="Third component")

    ax1.set_xlabel("First component")
    ax1.set_ylabel("Second component")
    ax1.set_title("Ground")
    for class_, m in zip([0, 1], ["o", "x"]):
        ax2.scatter(data_transformed[ypred == class_, 0], data_transformed[ypred == class_, 1], c=data_transformed[ypred == class_, 2], marker=m)

    ax2.figure.colorbar(second, ax=ax2, label="Third component")

    ax2.set_xlabel("First component")
    ax2.set_ylabel("Second component")
    ax2.set_title("Clustering labels")
    plt.legend()
    plt.show()


def load_file(file_path: str) -> Dict[str, Any]:
    with open(file_path, "rb") as f:
        return pickle.load(f)


def handle_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--logits_file",
        type=str,
        default=None,
        help="Results data file for plotting.")

    args = parser.parse_args()
    if args.logits_file is None:
        raise ValueError("Logits data file must be provided.")

    return args


if __name__ == "__main__":
    args = handle_args()
    main(args.logits_file)
