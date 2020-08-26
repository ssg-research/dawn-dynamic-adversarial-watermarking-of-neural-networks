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
