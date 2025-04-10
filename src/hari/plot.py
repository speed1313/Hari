import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
from argparse import ArgumentParser
import os

# font size
plt.rcParams.update({"font.size": 14})


def parse_args():
    parser = ArgumentParser()
    parser.add_argument(
        "--result_path",
        type=str,
        default="result/gpt-4o-2024-11-20/result.jsonl",
        help="Path to the result jsonl file",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    pivot_table = pd.read_json(args.result_path, lines=True)
    pivot_table = pivot_table.pivot(
        index="depth_percent", columns="context_length", values="score"
    )
    print(pivot_table)

    cmap = LinearSegmentedColormap.from_list(
        "custom_cmap", ["#F0496E", "#EBB839", "#0CD79F"]
    )
    # Create the heatmap with better aesthetics
    plt.figure(figsize=(17.5, 8))  # Can adjust these dimensions as needed
    sns.heatmap(
        pivot_table,
        # annot=True,
        fmt="g",
        cmap=cmap,
        cbar_kws={"label": "Score"},
        vmin=1,
        vmax=5,
        linewidths=0.5,
        linecolor="white",
    )

    # More aesthetics
    plt.xlabel("Context Length")  # X-axis label
    plt.ylabel("Depth Percent")  # Y-axis label
    plt.xticks(rotation=45)  # Rotates the x-axis labels to prevent overlap
    plt.yticks(rotation=0)  # Ensures the y-axis labels are horizontal
    plt.tight_layout()  # Fits everything neatly into the figure area
    # Show the plot
    plt.savefig(os.path.join(os.path.dirname(args.result_path), "heatmap.png"))
