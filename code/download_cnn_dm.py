from datasets import load_dataset
import pandas as pd
import os

# Load CNN/DailyMail dataset
print("🔽 Downloading CNN/DailyMail dataset...")
dataset = load_dataset("cnn_dailymail", '3.0.0')

# Only use a small subset for now to save time
# You can change `500` to a larger number if needed
train_data = dataset['train'].select(range(500))

# Convert to pandas DataFrame
df = pd.DataFrame({
    'article': train_data['article'],
    'reference_summary': train_data['highlights']
})

# Save to CSV in data/cnn_dm/
output_path = "../data/cnn_dm/cnn_dm_subset.csv"
os.makedirs("../data/cnn_dm", exist_ok=True)
df.to_csv(output_path, index=False)

print(f"✅ Dataset saved to {output_path}")
