from datasets import load_dataset
import pandas as pd
import os

print("🔽 Downloading XSum dataset...")
dataset = load_dataset("EdinburghNLP/xsum")
train_data = dataset['train'].select(range(500))  # reduce to 10 if slow

df = pd.DataFrame({
    'article': train_data['document'],
    'reference_summary': train_data['summary']
})

output_path = "../data/xsum/xsum_subset.csv"
os.makedirs("../data/xsum", exist_ok=True)
df.to_csv(output_path, index=False)

print(f"✅ XSum saved to {output_path}")
