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


class Logger:
    def __init__(self, prefix: str) -> None:
        self.prefix = prefix

    def __compose(self, level: str, msg: str) -> str:
        return "{} {}: {}".format(self.prefix, level, msg)

    def info(self, msg: str) -> None:
        print(self.__compose("INFO", msg))

    def warn(self, msg: str) -> None:
        print(self.__compose("WARN", msg))

    def error(self, msg: str) -> None:
        print(self.__compose("ERROR", msg))
