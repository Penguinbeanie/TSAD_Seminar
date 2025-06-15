import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

# Set figure size for A4 paper (11.69 x 8.27 inches)
plt.rcParams['figure.figsize'] = [11.69, 4]
plt.rcParams['figure.dpi'] = 300

def find_first_anomaly_sequence(df):
    # Find the first occurrence of label 1
    first_anomaly_idx = df[df['Label'] == 1].index[0]
    
    # Find the end of the anomaly sequence
    last_anomaly_idx = first_anomaly_idx
    for i in range(first_anomaly_idx, len(df)):
        if df['Label'].iloc[i] == 1:
            last_anomaly_idx = i
        else:
            break
    
    # Get the start and end indices for the window (15 points before and after the anomaly sequence)
    start_idx = max(0, first_anomaly_idx - 15)
    end_idx = min(len(df), last_anomaly_idx + 16)  # +16 to include 15 points after
    
    return start_idx, end_idx

def plot_sequence(data, labels, title, save_path, actual_indices):
    plt.figure()
    
    # Convert to numpy arrays for proper indexing
    actual_indices = np.array(actual_indices)
    
    # Plot the line connecting all points
    plt.plot(actual_indices, data, '-', color='gray', alpha=0.7, linewidth=1)
    
    # Plot normal points in black
    normal_mask = labels == 0
    if any(normal_mask):
        plt.scatter(actual_indices[normal_mask], data[normal_mask], 
                   color='black', s=30, label='Normal', zorder=3)
    
    # Plot anomalous points in red
    anomaly_mask = labels == 1
    if any(anomaly_mask):
        plt.scatter(actual_indices[anomaly_mask], data[anomaly_mask], 
                   color='red', s=30, label='Anomaly', zorder=3)
    
    plt.title(title)
    plt.xlabel('Dataset Index')
    plt.ylabel('RAM Usage (MB)')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    
    # Save the plot
    plt.savefig(save_path)
    plt.close()

def main():
    base_dir = r"C:\Users\Kai\Documents\Time_Series_Anomaly_Detection\TSAD_Seminar\Dataset_Creation"
    
    # Define anomaly types for each dataset
    anomaly_types = {
        10: "RAM Spike",
        11: "RAM Sleep", 
        12: "RAM Medium",
        13: "RAM Mixed"
    }
    
    # Process datasets 010-013
    for i in range(10, 14):
        dataset_path = os.path.join(base_dir, 'prelim_datasets', 
                                  f'0{i}_RAM{"spike" if i == 10 else "sleep" if i == 11 else "medium" if i == 12 else "mixed"}_{i}_Hardware_tr_{"1200" if i < 13 else "2000"}_1st_{1210 if i == 10 else 1324 if i == 11 else 1237 if i == 12 else 2083}.csv')
        
        df = pd.read_csv(dataset_path)
        start_idx, end_idx = find_first_anomaly_sequence(df)
        
        # Extract the sequence
        sequence_data = df['Data'].iloc[start_idx:end_idx].values
        sequence_labels = df['Label'].iloc[start_idx:end_idx].values
        actual_indices = range(start_idx, end_idx)
        
        # Create and save the plot
        save_path = os.path.join(base_dir, f'anomaly_sequence_{i}.png')
        title = f'First Anomaly Sequence - {anomaly_types[i]} (Dataset 0{i})'
        plot_sequence(sequence_data, sequence_labels, title, save_path, actual_indices)
    
    # Create baseline plot from dataset 010 (indices 46-76)
    dataset_010_path = os.path.join(base_dir, 'prelim_datasets', '010_RAMspike_10_Hardware_tr_1200_1st_1210.csv')
    df_010 = pd.read_csv(dataset_010_path)
    
    baseline_data = df_010['Data'].iloc[46:77].values
    baseline_labels = df_010['Label'].iloc[46:77].values
    baseline_indices = range(46, 77)
    
    # Create and save the baseline plot
    baseline_save_path = os.path.join(base_dir, 'baseline_sequence.png')
    plot_sequence(baseline_data, baseline_labels, 'Baseline Sequence - Normal RAM Usage (Dataset 010)', baseline_save_path, baseline_indices)

if __name__ == "__main__":
    main()
