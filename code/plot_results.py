import pandas as pd
import matplotlib.pyplot as plt
import os

df = pd.read_excel("../results/results_summary.xlsx")

metrics = ["ROUGE-1", "ROUGE-2", "ROUGE-L"]
scores = [df[metric][0] for metric in metrics]

plt.figure(figsize=(8, 5))
bars = plt.bar(metrics, scores, color=['skyblue', 'salmon', 'lightgreen'])

for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, yval + 0.01, f"{yval:.2f}", ha='center')

plt.ylim(0, 0.5)
plt.title(f"ROUGE Scores for {df['Model'][0]} on {df['Dataset'][0]}")
plt.ylabel("Score")
plt.grid(axis="y", linestyle="--", alpha=0.7)

os.makedirs("../results", exist_ok=True)
plt.tight_layout()
plt.savefig("../results/bart_xsum_rouge_scores.png")
print("Chart saved to: ../results/bart_xsum_rouge_scores.png")
