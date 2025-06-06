import pandas as pd
import numpy as np
import os

# Directory containing the CSVs
DATA_DIR = os.path.dirname(os.path.abspath(__file__))

# File paths
csv1_path = os.path.join(DATA_DIR, 'model_vus_pr_summary_uni.csv')
csv2_path = os.path.join(DATA_DIR, 'author_uni_mergedTable_VUS-PR.csv')
output_csv_path = os.path.join(DATA_DIR, 'COPY_vus_pr_comparison_summary_uni.csv')

try:
    # --- Load Data ---
    # Load the first CSV (assumed to be in wide format)
    df1 = pd.read_csv(csv1_path)
    
    # Load the second CSV (assumed to be already in wide format)
    df2 = pd.read_csv(csv2_path)

    # --- Compare ---
    # Align the two wide-format dataframes on the 'file' column
    merged = pd.merge(df1, df2, on='file', suffixes=('_ours', '_author'))

    # Get model columns that are present in both datasets
    df1_model_cols = set(df1.columns) - {'file'}
    df2_model_cols = set(df2.columns) - {'file'}
    common_models = sorted(list(df1_model_cols.intersection(df2_model_cols)))

    if not common_models:
        raise ValueError("No common model columns found between the two CSV files.")

    results = []
    for model in common_models:
        ours = merged[f'{model}_ours']
        author = merged[f'{model}_author']
        abs_diff = (ours - author).abs()
        
        # Calculate counts for different thresholds
        count_gt_5_percent = (abs_diff > 0.05).sum()
        count_gt_25_percent = (abs_diff > 0.25).sum()
        count_gt_50_percent = (abs_diff > 0.50).sum()

        avg_diff = abs_diff.mean()
        max_diff = abs_diff.max()
        max_diff_file = "N/A"
        num_anomaly = "N/A"
        if not abs_diff.isnull().all():
            max_diff_idx = abs_diff.idxmax()
            max_diff_file = merged.loc[max_diff_idx, 'file']
            num_anomaly = merged.loc[max_diff_idx, 'num_anomaly']

        results.append({
            'model': model,
            'avg_abs_diff': avg_diff,
            'max_abs_diff': max_diff,
            'file_with_max_diff': max_diff_file,
            'num_anomaly': num_anomaly,
            'datasets_diff_gt_5%': count_gt_5_percent,
            'datasets_diff_gt_25%': count_gt_25_percent,
            'datasets_diff_gt_50%': count_gt_50_percent
        })

    # Save results to CSV
    results_df = pd.DataFrame(results)
    results_df.to_csv(output_csv_path, index=False)
    print(f'Comparison complete. Results saved to {output_csv_path}')

except Exception as e:
    print(f"An error occurred: {e}") 