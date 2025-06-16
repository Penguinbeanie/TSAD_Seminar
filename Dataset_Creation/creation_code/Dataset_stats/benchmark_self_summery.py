import pandas as pd
import os
import glob

def analyze_benchmark_data():
    """
    Analyzes benchmark data from CSV files in a specified directory, calculates
    summary statistics for VUS-PR and VUS-ROC metrics, and saves the results
    to a new CSV file.
    """
    # The list of statistical models as provided in the instructions.
    statistical_models = [
        "Sub-IForest", "Sub-LOF", "IForest", "KShapeAD", "SAND", "KMeansAD",
        "Sub-MCD", "LOF", "Sub-KNN", "POLY", "Sub-PCA", "Sub-HBOS",
        "Sub-OCSVM", "MatrixProfile", "SR"
    ]
    
    # The list of neural network models as provided in the instructions.
    neural_net_models = [
        "LSTMAD", "USAD", "TranAD", "OmniAnomaly", "CNN", "AnomalyTransformer",
        "FITS", "AutoEncoder"
    ]

    # The script is intended to be run from the workspace root.
    input_dir = r"Evaluation/self/dataset_based"
    output_dir = r"Evaluation/self/dataset_based"

    # Find all CSV files for datasets 010 to 013.
    filepaths = []
    for i in range(10, 14):
        # Pad with 0 for numbers less than 10, although we start at 10.
        dataset_prefix = f"{i:03d}" 
        filepaths.extend(glob.glob(os.path.join(input_dir, f"{dataset_prefix}*.csv")))
    
    if not filepaths:
        print(f"No CSV files found for datasets 010-013 in '{input_dir}'.")
        return

    results = []
    
    for filepath in filepaths:
        try:
            df = pd.read_csv(filepath)
            # Standardize model name for consistency.
            df['Model'] = df['Model'].replace('KMeansAD_U', 'KMeansAD')
            dataset_name = os.path.basename(filepath)
            
            # Filter data for statistical and neural-net models
            df_stat = df[df['Model'].isin(statistical_models)]
            df_nn = df[df['Model'].isin(neural_net_models)]

            for metric in ["VUS-PR", "VUS-ROC"]:
                # Find the best performing model and its score for the metric.
                best_model_row = df.loc[df[metric].idxmax()]
                best_model = best_model_row['Model']
                best_score = best_model_row[metric]
                
                # Calculate average scores
                avg_score_all = df[metric].mean()
                avg_score_stat = df_stat[metric].mean() if not df_stat.empty else 0
                avg_score_nn = df_nn[metric].mean() if not df_nn.empty else 0
                
                results.append({
                    "Dataset": dataset_name,
                    "Metric": metric,
                    "Best Model": best_model,
                    "Best Score": best_score,
                    "Average Score": avg_score_all,
                    "Average Statistical Score": avg_score_stat,
                    "Average Neural-Net Score": avg_score_nn
                })
        except Exception as e:
            print(f"Error processing file {filepath}: {e}")
            
    if not results:
        print("No data was processed successfully.")
        return

    output_df = pd.DataFrame(results)
    # Define the output file path for the summary.
    output_filename = "benchmark_summary.csv"
    output_filepath = os.path.join(output_dir, output_filename)
    
    # Save the summary dataframe to a CSV file.
    output_df.to_csv(output_filepath, index=False)
    print(f"Benchmark summary saved to {output_filepath}")

if __name__ == "__main__":
    analyze_benchmark_data()