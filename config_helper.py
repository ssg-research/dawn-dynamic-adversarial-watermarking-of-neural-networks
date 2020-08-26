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
