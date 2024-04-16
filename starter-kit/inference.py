from argparse import ArgumentParser
from hashlib import sha256

import pandas as pd
from model import EarthQuakeModel
from torchgeo.datasets import QuakeSet
from tqdm import tqdm


def main(checkpoint: str):
    # Load the dataset
    dataset = QuakeSet(root="data", split="test", download=True)
    # Load checkpoint
    model = EarthQuakeModel.load_from_checkpoint(checkpoint, map_location="cpu").eval()
    # Make predictions
    predictions = []
    for i, sample in tqdm(enumerate(dataset)):
        prediction = model(sample["image"].unsqueeze(0)).item()
        metadata = dataset.data[i]
        # Note: The key generation made in this way for public evaluation only.
        key = f"{metadata['key']}/{metadata['patch']}/{metadata['images'][1]}"
        key = sha256(key.encode()).hexdigest()
        predictions += [
            {"key": key, "magnitude": prediction, "affected": int(prediction > 1)}
        ]
    pd.DataFrame(predictions).to_csv("predictions.csv", index=False)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    args = parser.parse_args()
    main(args.checkpoint)
