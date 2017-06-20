import argparse
import helpers
import torch

parser = argparse.ArgumentParser()
parser.add_argument('path')
args = parser.parse_args()
helpers.validate_path(args.path)


def evaluate():
    pass
