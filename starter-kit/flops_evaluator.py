# IMPORTANT: If you encounter issues with PAPI, please refer to this post: https://stackoverflow.com/questions/32308175/papi-avail-no-events-available. It could easily solve your problems.

import os
from argparse import ArgumentParser

import torch
from model import EarthQuakeModel

# This works for pypapi 6.0 (https://flozz.github.io/pypapi/)
from pypapi import events, papi_high

os.environ["PAPI_EVENTS"] = "PAPI_SP_OPS"  # FLOPs in single precision
os.environ["PAPI_OUTPUT_DIRECTORY"] = "papi_output"


def main(checkpoint: str):
    network_input = torch.ones([1, 4, 512, 512], dtype=torch.float32)
    model = EarthQuakeModel.load_from_checkpoint(checkpoint, map_location="cpu").eval()

    papi_high.hl_region_begin("model")
    model(network_input)
    papi_high.hl_region_end("model")

    # You will find a file in the papi_output folder with the key PAPI_SP_OPS. The value represents the value of flops to submit.


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    args = parser.parse_args()
    main(args.checkpoint)
