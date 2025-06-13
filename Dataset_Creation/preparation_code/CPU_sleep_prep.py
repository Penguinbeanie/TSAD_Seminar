import csv

def create_labeled_dataset():
    input_csv_path = 'Dataset_Creation/prelim_datasets/execution_log_5.csv'
    output_csv_path = 'Dataset_Creation/prelim_datasets/CPU_5_labeled.csv'

    processed_rows = []
    # This flag tracks if an anomaly cluster just occurred and the next idle sample should be labeled as anomaly.
    # anomaly_cluster_ended_requires_idle_tag = False # Removed this flag

    try:
        with open(input_csv_path, 'r', newline='') as infile:
            reader = csv.DictReader(infile)
            if not reader.fieldnames or 'cpu_percent' not in reader.fieldnames or 'state' not in reader.fieldnames:
                print(f"Error: CSV file {input_csv_path} must contain 'cpu_percent' and 'state' columns.")
                return

            for row in reader:
                current_state = row.get('state', '').strip()
                cpu_percent_str = row.get('cpu_percent', '').strip()

            

                # Try to get CPU data. Only rows with valid CPU data will be in the output.
                try:
                    cpu_data = float(cpu_percent_str)
                except ValueError:
                   
                    continue

                # Determine the label for the current data point
                label = 0  # Default label is 0 (normal/idle)

                if current_state == 'working_anomaly':
                    label = 1
                    
                else:  # For 'working_normal' or any other states that are not 'idle' or 'working_anomaly'
                    label = 0
                    # The flag 'anomaly_cluster_ended_requires_idle_tag' would have been set to False by the logic above. 
                
                processed_rows.append([cpu_data, label])

    except FileNotFoundError:
        print(f"Error: Input file not found at {input_csv_path}")
        return
    except Exception as e:
        print(f"An error occurred: {e}")
        return

    # Write the processed data to the new CSV file
    try:
        with open(output_csv_path, 'w', newline='') as outfile:
            writer = csv.writer(outfile)
            writer.writerow(['Data', 'Label'])  # Write header
            writer.writerows(processed_rows)
        print(f"Successfully created labeled dataset: {output_csv_path}")
    except Exception as e:
        print(f"Error writing output file {output_csv_path}: {e}")

if __name__ == '__main__':
    create_labeled_dataset()