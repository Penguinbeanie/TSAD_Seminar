import pandas as pd


# Read the input CSV file
input_file = 'Evaluation/Benchmark_Reproduced/merged_all_uni_metrics.csv'
df = pd.read_csv(input_file)

# Pivot the data to have models as columns and VUS-PR as values
pivoted_df = df.pivot(index='file', columns='model', values='VUS-PR')

# Reset the index to make 'file' a regular column
pivoted_df = pivoted_df.reset_index()

# Save to a new CSV file
output_file = 'Evaluation/Benchmark_Reproduced/model_vus_pr_summary_uni.csv'
pivoted_df.to_csv(output_file, index=False)

print(f"Created summary file: {output_file}") 