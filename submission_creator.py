# This script evaluate the format of your submission and add flops

from argparse import ArgumentParser

import pandas as pd


def main(predictions_file: str, flops: int):
    # Load predictions
    predictions = pd.read_csv(predictions_file)
    # Check format
    assert all(
        col in predictions.columns for col in ["key", "magnitude", "affected"]
    ), "Missing columns in predictions file"
    # Check values
    assert predictions["magnitude"].dtype == float, "Magnitude should be a float"
    assert predictions["affected"].dtype == int, "Affected should be an int"
    assert all(
        0 <= predictions["magnitude"] <= 10
    ), "Magnitude should be between 0 and 10"
    assert all(0 <= predictions["affected"] <= 1), "Affected should be 0 or 1"
    # Add flops
    predictions["flops"] = flops
    # Save
    predictions.to_csv("submission.csv", index=False)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--predictions", type=str, required=True)
    parser.add_argument("--flops", type=int, required=True)
    args = parser.parse_args()
    main(args.predictions, args.flops)
