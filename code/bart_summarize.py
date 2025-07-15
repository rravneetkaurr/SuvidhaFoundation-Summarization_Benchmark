from transformers import BartTokenizer, BartForConditionalGeneration
import pandas as pd
import torch
import os

# Load BART model and tokenizer
tokenizer = BartTokenizer.from_pretrained("facebook/bart-large-cnn")
model = BartForConditionalGeneration.from_pretrained("facebook/bart-large-cnn")

# Load the dataset (you can switch to cnn_dm_subset.csv later)
df = pd.read_csv("../data/xsum/xsum_subset.csv")  # Or change to cnn_dm_subset.csv
articles = df['article'].tolist()

generated_summaries = []

print("🧠 Generating summaries with BART...")
for i, article in enumerate(articles):
    inputs = tokenizer(article, max_length=1024, truncation=True, return_tensors="pt")
    summary_ids = model.generate(
        inputs["input_ids"],
        max_length=150,
        num_beams=4,
        early_stopping=True
    )
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    generated_summaries.append(summary)
    print(f"{i+1}/{len(articles)} ✅")

# Save output
os.makedirs("../results", exist_ok=True)
output_path = "../results/bart_xsum_output.txt"
with open(output_path, "w", encoding="utf-8") as f:
    for summary in generated_summaries:
        f.write(summary + "\n")

print(f"\n✅ Done! Summaries saved to: {output_path}")
