from hari.prepare_dataset import prepare_haystacks_across_lengths_and_positions
from hari.model import Model
from hari.judger import Judger
from datasets import load_dataset
from argparse import ArgumentParser
import weave
import os
import json
from dataclasses import dataclass, asdict


@dataclass
class Result:
    model: str
    context_length: int
    depth_percent: int
    needle: str
    model_response: str
    score: int


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
    parser.add_argument(
        "--use_weave",
        default=True,
        help="Use Weave for the retrieval",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gpt-4o-2024-11-20",
        choices=["gpt-4o-2024-11-20", "gpt-4o-mini-2024-07-18"],
        help="Model to use for the retrieval",
    )
    parser.add_argument(
        "--judger_model",
        type=str,
        default="gpt-4o-2024-11-20",
        choices=["gpt-4o-2024-11-20", "gpt-4o-mini-2024-07-18"],
        help="Model to use for the judger",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="result",
        help="Output directory for the results",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.use_weave:
        weave.init("hari")

    ds = load_dataset("wikimedia/wikipedia", "20231101.ja", split="train")
    ds = ds.shuffle(seed=42)
    # Prepare haystacks across lengths and positions
    all_haystacks = prepare_haystacks_across_lengths_and_positions(
        ds, args.needle, lengths=[1024, 2048], depth=args.depth
    )
    lengths = sorted(set([info["length"] for info in all_haystacks]))
    depths = sorted(set([info["depth"] for info in all_haystacks]))
    model = Model(args.model)
    judger = Judger(args.judger_model)

    # Generate 2D accuracy matrix
    results = []
    for example in all_haystacks:
        haystack = example["haystack"]
        length = example["length"]
        depth = example["depth"]
        retrieved = model.retrieve_needle(haystack, args.question)
        score = judger.judge_retrieval(retrieved, args.needle, args.question)
        results.append(
            Result(
                model=args.model,
                context_length=length,
                depth_percent=depth,
                needle=args.needle,
                model_response=retrieved,
                score=score,
            )
        )
    # sort by context_length and depth_percent
    results.sort(key=lambda x: (x.context_length, x.depth_percent))

    output_path = os.path.join(args.output_dir, args.model, "result.jsonl")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        for result in results:
            f.write(json.dumps(asdict(result), ensure_ascii=False) + "\n")
