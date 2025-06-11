import pandas as pd
import numpy as np
from pathlib import Path

def calculate_anomaly_metrics(df):
    # Convert labels to numpy array for easier processing
    labels = df['Label'].values
    
    # Find sequences of anomalies
    # A sequence starts when 0->1 and ends when 1->0
    sequence_starts = np.where((labels[1:] == 1) & (labels[:-1] == 0))[0] + 1
    sequence_ends = np.where((labels[1:] == 0) & (labels[:-1] == 1))[0] + 1
    
    # If the sequence ends with an anomaly, add the last index
    if labels[-1] == 1:
        sequence_ends = np.append(sequence_ends, len(labels))
    
    # Calculate metrics
    total_sequences = len(sequence_starts)
    total_anomaly_points = np.sum(labels == 1)
    
    if total_sequences > 0:
        sequence_lengths = sequence_ends - sequence_starts
        avg_length = np.mean(sequence_lengths)
        max_length = np.max(sequence_lengths)
        min_length = np.min(sequence_lengths)
    else:
        avg_length = 0
        max_length = 0
        min_length = 0
    
    anomaly_ratio = total_anomaly_points / len(labels)
    
    return {
        'Total Anomaly Sequences': total_sequences,
        'Total Anomaly Points': total_anomaly_points,
        'Average Anomaly Length': round(avg_length, 2),
        'Longest Anomaly Length': max_length,
        'Shortest Anomaly Length': min_length,
        'Anomaly Ratio': round(anomaly_ratio, 4)
    }

def main():
    # Input and output paths
    input_path = r"C:\Users\Kai\Documents\Time_Series_Anomaly_Detection\TSAD_Seminar\Dataset_Creation\prelim_datasets\013_RAMmixed_13_Hardware_tr_2000_1st_2083.csv"
    output_dir = Path(r"C:\Users\Kai\Documents\Time_Series_Anomaly_Detection\TSAD_Seminar\Dataset_Creation\creation_code\Dataset_stats")
    
    # Read the dataset
    df = pd.read_csv(input_path)
    
    # Calculate metrics
    metrics = calculate_anomaly_metrics(df)
    
    # Create output file name based on input file
    input_filename = Path(input_path).stem
    output_path = output_dir / f"{input_filename}_metrics.txt"
    
    # Write results to file
    with open(output_path, 'w') as f:
        f.write("Dataset Anomaly Statistics\n")
        f.write("=========================\n\n")
        for metric, value in metrics.items():
            f.write(f"{metric}: {value}\n")
    
    print(f"Metrics have been saved to: {output_path}")

if __name__ == "__main__":
    main()
