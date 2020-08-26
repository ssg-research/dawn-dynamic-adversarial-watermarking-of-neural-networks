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


import configparser


def load_config(file_path) -> configparser.ConfigParser:
    config = configparser.ConfigParser()
    config.read(file_path)
    return config


def print_config(config: configparser.ConfigParser) -> None:
    default = config["DEFAULT"]

    print("[DEFAULT]")
    default_opts = []
    for k, v in default.items():
        default_opts.append(k)
        print("{}: {}".format(k, v))
    default_opts = set(default_opts)

    for section in config.sections():
        print("\n[{}]".format(section))
        keys = set(config[section]) - default_opts
        for key in keys:
            print("{}: {}".format(key, config[section][key]))
