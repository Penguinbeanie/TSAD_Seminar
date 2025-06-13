import csv
import os

# Define input and output file paths
# Script is in Dataset_Creation/creation_code/RAM_production/
# Data is in Dataset_Creation/prelim_datasets/
SCRIPT_DIR = os.path.dirname(__file__)
BASE_INPUT_OUTPUT_DIR = os.path.join(SCRIPT_DIR, "..", "..", "prelim_datasets")

INPUT_FILENAME = "dataset_3_medium_ram_spike_log.csv"
OUTPUT_FILENAME = "dataset_3_medium_ram_training_data.csv"

INPUT_FILE_PATH = os.path.join(BASE_INPUT_OUTPUT_DIR, INPUT_FILENAME)
OUTPUT_FILE_PATH = os.path.join(BASE_INPUT_OUTPUT_DIR, OUTPUT_FILENAME)

def convert_log_data():
    """
    Reads the log data, processes it according to specified rules,
    and writes the output to a new CSV file for training.
    """
    processed_points = []

    # --- Pass 1: Read data and assign initial labels ---
    try:
        with open(INPUT_FILE_PATH, 'r', newline='') as infile:
            reader = csv.DictReader(infile)
            for idx, row in enumerate(reader):
                ram_usage_mb_str = row.get('ram_usage_mb', '')
                state = row.get('state', '')
                ram_value = None

                if ram_usage_mb_str:
                    try:
                        ram_value = float(ram_usage_mb_str)
                    except ValueError:
                        print(f"Warning: Could not parse RAM value '{ram_usage_mb_str}' at original input row {idx + 2}. RAM set to None for this entry.")
                
                # Assign initial label
                current_label = 0  # Default label
                if state.startswith('working_anomaly_'):
                    current_label = 1
                elif state in ['idle', 'working_normal', 'working_special_prep']:
                    current_label = 0
               

                processed_points.append({
                    'ram_value': ram_value,  # Can be None
                    'label': current_label,    # This will be updated in Pass 2
                    'state': state,
                    # 'original_index': idx # Kept for potential debugging
                })
    except FileNotFoundError:
        print(f"Error: Input file not found at {INPUT_FILE_PATH}")
        return
    except Exception as e:
        print(f"An error occurred during Pass 1 (reading input): {e}")
        return

    if not processed_points:
        print("No data points with RAM usage found to process.")
        return

    # --- Pass 2: Adjust labels for idle states around 'working_anomaly_sleep' ---
    # Identify blocks of 'working_anomaly_sleep'
    sleep_anomaly_blocks = []
    in_block = False
    current_block_start = -1

    for i, point in enumerate(processed_points):
        if point['state'] == 'working_anomaly_sleep':
            if not in_block:
                in_block = True
                current_block_start = i
        else:
            if in_block:
                # Block ended at i-1
                sleep_anomaly_blocks.append((current_block_start, i - 1))
                in_block = False
    if in_block:  # Handle case where file ends with a sleep block
        sleep_anomaly_blocks.append((current_block_start, len(processed_points) - 1))

    # Propagate labels for each identified block
    for start_idx, end_idx in sleep_anomaly_blocks:
        # Propagate backwards from the start of the sleep block
        ptr = start_idx - 1
        while ptr >= 0 and processed_points[ptr]['state'] == 'idle':
            processed_points[ptr]['label'] = 1
            ptr -= 1

        # Propagate forwards from the end of the sleep block
        ptr = end_idx + 1
        while ptr < len(processed_points) and processed_points[ptr]['state'] == 'idle':
            processed_points[ptr]['label'] = 1
            ptr += 1
            
    # --- Pass 3: Write to output CSV ---
    try:
        os.makedirs(os.path.dirname(OUTPUT_FILE_PATH), exist_ok=True)
        with open(OUTPUT_FILE_PATH, 'w', newline='') as outfile:
            writer = csv.writer(outfile)
            writer.writerow(['Data', 'Label'])  # Header
            for point in processed_points:
                if point['ram_value'] is not None: # Only write points that originally had RAM data
                    writer.writerow([point['ram_value'], point['label']])
        print(f"Successfully converted data and saved to {OUTPUT_FILE_PATH}")
    except Exception as e:
        print(f"An error occurred during Pass 3 (writing output): {e}")

if __name__ == '__main__':
    convert_log_data() 