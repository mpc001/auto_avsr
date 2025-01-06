import logging
from argparse import ArgumentParser

import torch
import torchaudio
from datamodule.data_module import DataModule
from pytorch_lightning import Trainer


# Set environment variables and logger level
logging.basicConfig(level=logging.WARNING)


def get_trainer(args):
    return Trainer(num_nodes=1, devices=1, accelerator="gpu")


def get_lightning_module(args):
    # Set modules and trainer
    from lightning import ModelModule
    modelmodule = ModelModule(args)
    return modelmodule


def parse_args():
    parser = ArgumentParser()
    parser.add_argument(
        "--modality",
        type=str,
        help="Type of input modality",
        required=True,
        choices=["audio", "video"],
    )
    parser.add_argument(
        "--root-dir",
        type=str,
        help="Root directory of preprocessed dataset",
        required=True,
    )
    parser.add_argument(
        "--test-file",
        default="lrs3_test_transcript_lengths_seg16s.csv",
        type=str,
        help="Filename of testing label list. (Default: lrs3_test_transcript_lengths_seg16s.csv)",
        required=True,
    )
    parser.add_argument(
        "--pretrained-model-path",
        type=str,
        help="Path to the pre-trained model",
        required=True,
    )
    parser.add_argument(
        "--decode-snr-target",
        type=float,
        default=999999,
        help="Level of signal-to-noise ratio (SNR)",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Flag to use debug level for logging",
    )
    return parser.parse_args()


def init_logger(debug):
    fmt = "%(asctime)s %(message)s" if debug else "%(message)s"
    level = logging.DEBUG if debug else logging.INFO
    logging.basicConfig(format=fmt, level=level, datefmt="%Y-%m-%d %H:%M:%S")


def cli_main():
    args = parse_args()
    init_logger(args.debug)
    modelmodule = get_lightning_module(args)
    datamodule = DataModule(args)
    trainer = get_trainer(args)
    trainer.test(model=modelmodule, datamodule=datamodule)


if __name__ == "__main__":
    cli_main()
