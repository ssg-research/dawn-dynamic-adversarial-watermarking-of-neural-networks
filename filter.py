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


import hashlib
import itertools
import math
import hmac
import bitstring
import torch


def default_key(length: int):
    import string
    import random
    random.seed(42)
    return ''.join(random.choices(string.ascii_uppercase + string.digits, k=length))


class WatermarkFilter:
    def __init__(self, key: str, shape: (int, int, int), precision: int = 16, probability: float = 5 / 1000, hash_f: callable = hashlib.sha256) -> None:
        self.key = key.encode("utf-8")
        self.shape = shape
        self.hash_f = hash_f
        self.precision = precision
        self.probability = probability
        self.bound = math.floor((2 ** self.precision) * self.probability)
        self.strip = 25  # None for production; used for efficiency

    @staticmethod
    def __bytes_to_bits(msg: str) -> str:
        return bitstring.BitArray(hex=msg).bin

    def __hmac(self, key: bytes, msg: bytes, strip: int = None) -> str:
        if strip is not None:
            key = itertools.islice(key, 0, strip)
            msg = itertools.islice(msg, 0, strip)

        return hmac.new(key, msg, self.hash_f).hexdigest()

    def is_watermark(self, image: torch.FloatTensor) -> bool:
        if image.shape != self.shape:
            raise AssertionError("Image shape {} different from expected {}.".format(image.shape, self.shape))

        hashed = self.__hmac(key=self.key, msg=image.numpy().tobytes())
        bits = self.__bytes_to_bits(hashed)
        return int(bits[:self.precision], 2) <= self.bound
