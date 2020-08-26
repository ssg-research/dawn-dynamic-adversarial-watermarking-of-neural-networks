from typing import Any, Dict


class Score:
    def __init__(self, value: Any) -> None:
        self.value = value

    def __call__(self) -> Any:
        return self.value


class FloatScore(Score):
    def __init__(self, value: float) -> None:
        super(FloatScore, self).__init__(value)

    def __call__(self) -> float:
        return self.value


class DictScore(Score):
    def __init__(self, value: Dict[int, float]) -> None:
        super(DictScore, self).__init__(value)

    def __call__(self) -> Dict[int, float]:
        return self.value
