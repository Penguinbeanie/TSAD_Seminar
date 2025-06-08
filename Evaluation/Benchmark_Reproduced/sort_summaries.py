import pandas as pd
import os

def sort_csv(file_path):
    """Sorts a CSV file by 'avg_abs_diff' in descending order."""
    df = pd.read_csv(file_path)
    df_sorted = df.sort_values(by='avg_abs_diff', ascending=False)
    df_sorted.to_csv(file_path, index=False)
    print(f"Sorted {file_path}")

if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.realpath(__file__))
    sort_csv(os.path.join(script_dir, 'COPY_vus_pr_comparison_summary_multi.csv'))
    sort_csv(os.path.join(script_dir, 'COPY_vus_pr_comparison_summary_uni.csv')) 