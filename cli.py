import argparse

class Cli:

 def __init__(self):
    self.parser = argparse.ArgumentParser()
    self.parser.add_argument("--train", help="flag for training", type=bool)

 def parse(self):
    args = self.parser.parse_args()
    return args

