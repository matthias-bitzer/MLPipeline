import training
import config

def start():
    config1 = config.Config()
    training.initial_training(config1)


if __name__ == '__main__':
    start()