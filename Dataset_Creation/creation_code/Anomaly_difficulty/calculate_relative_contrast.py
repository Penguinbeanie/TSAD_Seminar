import numpy as np
import pandas as pd
from tslearn.metrics import cdist_dtw
import os
import time
from datetime import datetime


def series_to_subsequences(time_series: np.ndarray, window_size: int) -> np.ndarray:
    """
    Converts a 1D time series into a 2D array of overlapping subsequences.

    Args:
        time_series (np.ndarray): The input 1D time series.
        window_size (int): The desired length of each subsequence.

    Returns:
        np.ndarray: A 2D array where each row is a subsequence. Returns an
                    empty array if the time series is shorter than the window size.
    """
    # Check if the series is long enough
    if len(time_series) < window_size:
        return np.array([])
        
    # Calculate the number of subsequences that can be created
    num_subsequences = len(time_series) - window_size + 1
    
    # Use numpy's stride tricks for a fast and memory-efficient implementation
    shape = (num_subsequences, window_size)
    strides = (time_series.strides[0], time_series.strides[0])
    return np.lib.stride_tricks.as_strided(time_series, shape=shape, strides=strides)

def calculate_relative_contrast(subsequences: np.ndarray) -> float:
    """
    Calculates the Relative Contrast (RC) for a set of time series subsequences
    using DTW distance metric.

    This implementation follows the definition from the paper:
    Rc = E[Dmean(s)] / E[Dmin(s)]

    Args:
        subsequences (np.ndarray): A 2D numpy array where each row is a time series
                                   subsequence. Shape: (n_subsequences, length_of_subsequence).

    Returns:
        float: The calculated Relative Contrast (RC) score.
    """
    start_time = time.time()
    n_subsequences = subsequences.shape[0]
    print(f"\nStarting relative contrast calculation for {n_subsequences} subsequences...")
    print(f"Estimated memory usage: {(n_subsequences * n_subsequences * 8) / (1024*1024):.2f} MB for distance matrix")
    
    if n_subsequences <= 1:
        print("Warning: Cannot calculate RC with 1 or fewer subsequences. Returning 0.")
        return 0.0

    print("Reshaping subsequences...")
    # Reshape subsequences to have shape (n_subsequences, length_of_subsequence, 1)
    # as required by cdist_dtw
    subsequences_reshaped = subsequences.reshape((subsequences.shape[0], subsequences.shape[1], 1))
    
    print("Calculating DTW distance matrix (this may take a few minutes)...")
    matrix_start_time = time.time()
    # Calculate the pairwise distance matrix using DTW - this is vectorized and much faster
    dist_matrix = cdist_dtw(subsequences_reshaped, subsequences_reshaped)
    matrix_time = time.time() - matrix_start_time
    print(f"Distance matrix calculation completed in {matrix_time:.2f} seconds")
    
    print("Computing minimum and mean distances...")
    # For each subsequence, find Dmin(s) (distance to nearest neighbor)
    np.fill_diagonal(dist_matrix, np.inf)
    d_mins = np.min(dist_matrix, axis=1)

    # For each subsequence, find Dmean(s) (average distance to all others)
    np.fill_diagonal(dist_matrix, 0)
    d_means = np.sum(dist_matrix, axis=1) / (n_subsequences - 1)

    # Calculate the expectation (average) of Dmin and Dmean
    avg_d_min = np.mean(d_mins)
    avg_d_mean = np.mean(d_means)
    
    if avg_d_min == 0:
        print("Warning: Average minimum distance is 0. Returning 0.")
        return 0.0

    # Compute the final Relative Contrast score
    rc_score = avg_d_mean / avg_d_min
    total_time = time.time() - start_time
    print(f"\nCalculation completed in {total_time:.2f} seconds")
    print(f"Average distance to nearest neighbor (E[Dmin]): {avg_d_min:.4f}")
    print(f"Average distance to all neighbors (E[Dmean]): {avg_d_mean:.4f}")
    
    return rc_score

if __name__ == "__main__":
    overall_start_time = time.time()
    print(f"Starting analysis run at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Define input datasets and output directory
    datasets_to_process = [
        r"C:\Users\Kai\Documents\Time_Series_Anomaly_Detection\TSAD_Seminar\Dataset_Creation\prelim_datasets\010_RAMspike_10_Hardware_tr_1200_1st_1210.csv",
        r"C:\Users\Kai\Documents\Time_Series_Anomaly_Detection\TSAD_Seminar\Dataset_Creation\prelim_datasets\011_RAMsleep_11_Hardware_tr_1200_1st_1324.csv",
        r"C:\Users\Kai\Documents\Time_Series_Anomaly_Detection\TSAD_Seminar\Dataset_Creation\prelim_datasets\012_RAMmedium_12_Hardware_tr_1200_1st_1237.csv",
        r"C:\Users\Kai\Documents\Time_Series_Anomaly_Detection\TSAD_Seminar\Dataset_Creation\prelim_datasets\013_RAMmixed_13_Hardware_tr_2000_1st_2083.csv"
    ]
    window_sizes_to_test = [16, 6, 8, 12]
    OUTPUT_DIR = r"C:\Users\Kai\Documents\Time_Series_Anomaly_Detection\TSAD_Seminar\Dataset_Creation\creation_code\Anomaly_difficulty"
    
    # Ensure output directory exists
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    for INPUT_PATH in datasets_to_process:
        for WINDOW_SIZE in window_sizes_to_test:
            iteration_start_time = time.time()
            print("\n" + "="*80)
            print(f"Processing: {os.path.basename(INPUT_PATH)} with window size: {WINDOW_SIZE}")
            print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            print("="*80)
            
            # Read the CSV file
            print("\nReading input file...")
            df = pd.read_csv(INPUT_PATH)
            
            # Assuming the CSV has a 'value' column - adjust if column name is different
            time_series = df.iloc[:, 0].values  # Takes the first column as the time series
            print(f"Time series length: {len(time_series)}")
            
            # Convert to subsequences
            print("\nCreating subsequences...")
            subsequences = series_to_subsequences(time_series, WINDOW_SIZE)
            
            if len(subsequences) == 0:
                print(f"Not enough data points to create subsequences of size {WINDOW_SIZE}. Skipping.")
                continue
                
            print(f"Created {len(subsequences)} subsequences")
            
            # Calculate relative contrast
            rc_score = calculate_relative_contrast(subsequences)
            
            # Create output filename based on input filename
            input_filename = os.path.basename(INPUT_PATH)
            output_filename = f"{WINDOW_SIZE}_rc_score_{input_filename.replace('.csv', '')}.txt"
            output_path = os.path.join(OUTPUT_DIR, output_filename)
            
            # Save the result
            iteration_time = time.time() - iteration_start_time
            print("\nSaving results...")
            with open(output_path, 'w') as f:
                f.write(f"Analysis completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"Total execution time for this run: {iteration_time:.2f} seconds\n")
                f.write(f"Relative Contrast Score: {rc_score}\n")
                f.write(f"Window Size Used: {WINDOW_SIZE}\n")
                f.write(f"Input File: {INPUT_PATH}\n")
                f.write(f"Number of subsequences analyzed: {len(subsequences)}\n")
            
            print(f"\nAnalysis for this configuration complete. Results saved to: {output_path}")
            print(f"Relative Contrast Score: {rc_score}")
            print(f"Execution time for this run: {iteration_time:.2f} seconds")

    total_run_time = time.time() - overall_start_time
    print("\n" + "="*80)
    print(f"All analyses completed at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Total execution time: {total_run_time:.2f} seconds")
    print("="*80)