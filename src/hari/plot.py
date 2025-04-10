import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap

# pivot_table = pd.DataFrame(z_scores, index=depths, columns=lengths)

# sample table
# pivot_table = pd.DataFrame(
#     [
#         [0, 0, 0, 0, 0],
#         [0, 1, 1, 0, 0],
#         [0, 1, 1, 1, 0],
#         [0, 1, 1, 1, 0],
#         [0, 1, 1, 1, 0],
#     ]
# )

# load retrieval accuracy from jsonl file
path = "result/gpt-4o-2024-11-20/result.jsonl"
# {"score": 1, "context_length": 1024, "depth_percent": 0.0}
pivot_table = pd.read_json(path, lines=True)
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
    vmin=0,
    vmax=1,
    linewidths=0.5,
    linecolor="white",
)

# More aesthetics
plt.xlabel("Token Limit")  # X-axis label
plt.ylabel("Depth Percent")  # Y-axis label
plt.xticks(rotation=45)  # Rotates the x-axis labels to prevent overlap
plt.yticks(rotation=0)  # Ensures the y-axis labels are horizontal
plt.tight_layout()  # Fits everything neatly into the figure area
# Show the plot
plt.savefig("retrieval_accuracy_heatmap.png")
