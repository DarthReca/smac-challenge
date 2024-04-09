# IMPORTANT: If you encounter issues with PAPI, please refer to this post: https://stackoverflow.com/questions/32308175/papi-avail-no-events-available. It could easily solve your problems.

import os
import torch
from pypapi import events, papi_high # This works for pypapi 6.0 (https://flozz.github.io/pypapi/)

os.environ["PAPI_EVENTS"] = "PAPI_SP_OPS" # FLOPs in single precision
os.environ["PAPI_OUTPUT_DIRECTORY"] = "papi_output"

network_input = torch.ones([1, 4, 512, 512], dtype=torch.float32)

papi_high.hl_region_begin("model")
# ADD HERE YOUR PIPELINE
papi_high.hl_region_end("model")

# You will find a file in the papi_output folder with the key PAPI_SP_OPS. The value represents the value of flops to submit.
