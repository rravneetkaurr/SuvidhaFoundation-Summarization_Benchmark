from rouge_score import rouge_scorer
import pandas as pd

reference_df = pd.read_csv("../data/xsum/xsum_subset.csv")
with open("../results/bart_xsum_output.txt", "r", encoding="utf-8") as f:
    generated_summaries = f.read().splitlines()

scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)

rouge1_scores = []
rouge2_scores = []
rougeL_scores = []

print("Calculating ROUGE scores")
for ref, pred in zip(reference_df["reference_summary"], generated_summaries):
    scores = scorer.score(ref, pred)
    rouge1_scores.append(scores["rouge1"].fmeasure)
    rouge2_scores.append(scores["rouge2"].fmeasure)
    rougeL_scores.append(scores["rougeL"].fmeasure)

rouge1 = sum(rouge1_scores) / len(rouge1_scores)
rouge2 = sum(rouge2_scores) / len(rouge2_scores)
rougeL = sum(rougeL_scores) / len(rougeL_scores)

print(f"\nROUGE-1: {rouge1:.4f}")
print(f"ROUGE-2: {rouge2:.4f}")
print(f"ROUGE-L: {rougeL:.4f}")

df = pd.DataFrame([{
    "Model": "BART",
    "Dataset": "XSum",
    "ROUGE-1": round(rouge1, 4),
    "ROUGE-2": round(rouge2, 4),
    "ROUGE-L": round(rougeL, 4),
    "Training Time": "N/A",
    "GPU/TPU used": "N/A",
    "Notes": "Summarized using pretrained BART-large-cnn"
}])

df.to_excel("../results/results_summary.xlsx", index=False)
print("\nResults saved to: ../results/results_summary.xlsx")
