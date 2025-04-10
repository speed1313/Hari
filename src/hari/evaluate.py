from hari.prepare_dataset import prepare_haystacks_across_lengths_and_positions
from hari.gpt4o import retrieve_needle
from datasets import load_dataset
import unicodedata
import numpy as np
from argparse import ArgumentParser


def parse_args():
    parser = ArgumentParser()
    parser.add_argument(
        "--context_length_max",
        type=int,
        default=2048,
        help="Maximum context length for the haystack",
    )
    parser.add_argument(
        "--context_length_min",
        type=int,
        default=1024,
        help="Minimum context length for the haystack",
    )
    parser.add_argument(
        "--interval",
        type=int,
        default=256,
        help="Interval for the context length",
    )
    parser.add_argument(
        "--depth",
        type=int,
        default=5,
        help="Depth for the haystack",
    )
    parser.add_argument(
        "--needle",
        type=str,
        default="京都でおすすめの観光地は、ロームシアター京都の３階にあるラウンジです。",
        help="Needle to insert into the haystack",
    )
    parser.add_argument(
        "--question",
        type=str,
        default="京都でおすすめの観光地はどこですか？",
        help="Question to ask the model",
    )


if __name__ == "__main__":
    args = parse_args()

    ds = load_dataset("wikimedia/wikipedia", "20231101.ja", split="train")
    ds = ds.shuffle(seed=42)
    # Prepare haystacks across lengths and positions
    all_haystacks = prepare_haystacks_across_lengths_and_positions(
        ds, args.needle, lengths=[1024, 2048], depth=args.depth
    )

    result_info = {}
    lengths = sorted(set([info["length"] for info in all_haystacks]))
    depths = sorted(set([info["depth"] for info in all_haystacks]))

    # Generate 2D accuracy matrix
    z_scores = np.zeros((len(depths), len(lengths)))
    for i, depth in enumerate(depths):
        for j, length in enumerate(lengths):
            haystack = None
            for info in all_haystacks:
                if info["length"] == length and info["depth"] == depth:
                    haystack = info["haystack"]
                    break
            retrieved = retrieve_needle(haystack, args.question)
            print(f"Haystack Length: {length}, Depth: {depth}, Retrieved: {retrieved}")
            # Normalize the retrieved string
            retrieved = unicodedata.normalize("NFKC", retrieved)
            # Normalize the needle string
            accuracy = int(args.ground_truth in retrieved)
            z_scores[i, j] = accuracy
