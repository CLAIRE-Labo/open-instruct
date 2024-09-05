import argparse
from pathlib import Path


def main(args):
    pass


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Chat')
    parser.add_argument('model', type=Path, help='Path to the model file')
