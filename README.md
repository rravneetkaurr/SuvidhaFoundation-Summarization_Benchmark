# MULTI-DOCUMENT ABSTRACTIVE SUMMARIZATION BENCHMARKING

## Objective: 

This project benchmarks summarization models (like BART) on datasets such as XSum using ROUGE evaluation.

## What’s Completed:

- BART model run on XSum dataset (10 samples)
- Generated summaries saved
- ROUGE scores computed (ROUGE-1, ROUGE-2, ROUGE-L)
- Evaluation results saved in Excel
- Bar chart generated

## Folder Structure:

code/            → All Python scripts  
data/xsum/       → Input sample CSV file  
results/         → Output summaries, Excel file, and chart  
requirements.txt → Python libraries needed  
README.md        → Project info and instructions  

## How to Run:

1. Set up environment:
    ```
    python -m venv venv
    venv\Scripts\activate
    pip install -r requirements.txt
    ```

2. Run BART summarization:
    ```
    cd code
    python bart_summarize.py
    ```

3. Evaluate ROUGE:
    ```
    python evaluate_rouge.py
    ```

4. Generate chart:
    ```
    python plot_results.py
    ```

## ROUGE Results (BART on XSum):

- ROUGE-1: 0.1987  
- ROUGE-2: 0.0681  
- ROUGE-L: 0.1582
