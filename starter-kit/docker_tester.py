from argparse import ArgumentParser
from hashlib import sha256

import pandas as pd
from model import EarthQuakeModel
from torchgeo.datasets import QuakeSet
from tqdm import tqdm
from pypapi import events, papi_low 


def main(checkpoint: str):
    # Load the dataset
    dataset = QuakeSet(root="data", split="test", download=False)
    # Load checkpoint
    model = EarthQuakeModel.load_from_checkpoint(checkpoint, map_location="cpu", in_chans=4).eval()
    papi_low.library_init()
    evs = papi_low.create_eventset()
    papi_low.add_event(evs, events.PAPI_FP_OPS)
    # Make predictions
    predictions = []
    for metadata, sample in tqdm(zip(dataset.data, dataset)):
        papi_low.start(evs)
        sample["image"] = sample["image"].unsqueeze(0)
        out = model.predict_step(sample, 0)
        flops = papi_low.stop(evs)
        predictions.append({"key": metadata["key"], "magnitude": out, "affected": int(out > 1), "flops": flops[0]})
    pd.DataFrame(predictions).to_csv("temp/submission.csv", index=False)


if __name__ == "__main__":
    main("checkpoint/checkpoint.ckpt")
