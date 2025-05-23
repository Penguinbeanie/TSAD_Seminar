import pandas as pd
import os
from datetime import timedelta

# Read the CSV files
cpu_data = pd.read_csv('Dataset_Creation/kai-VMware-Virtual-Platform_250523_1704.csv')
execution_log = pd.read_csv('Dataset_Creation/execution_log.csv')

# Convert timestamp columns to datetime
cpu_data['time'] = pd.to_datetime(cpu_data['time'])
execution_log['timestamp'] = pd.to_datetime(execution_log['timestamp'])

# Create a dictionary to map timestamps to states
state_map = {}
anomaly_periods = []  # List to store anomaly periods (start, end)

# First pass: identify anomaly periods
for i in range(len(execution_log)):
    row = execution_log.iloc[i]
    if row['state'] == 'anomaly':
        # Anomaly starts 1 second after detection
        anomaly_start = row['timestamp'] + timedelta(seconds=1)
        # Find the next sleeping state
        anomaly_end = None
        for j in range(i + 1, len(execution_log)):
            if execution_log.iloc[j]['state'] == 'sleeping':
                anomaly_end = execution_log.iloc[j]['timestamp']
                break
        
        # Ensure anomaly runs for at least 2 seconds
        min_anomaly_end = anomaly_start + timedelta(seconds=2)
        if anomaly_end is None or anomaly_end < min_anomaly_end:
            anomaly_end = min_anomaly_end
            
        anomaly_periods.append((anomaly_start, anomaly_end))

# Create new dataframe with Data and Label columns
merged_data = []
for _, row in cpu_data.iterrows():
    timestamp = row['time']
    user_percentage = row['User%']
    
    # Check if timestamp falls within any anomaly period
    is_anomaly = any(start <= timestamp <= end for start, end in anomaly_periods)
    
    # Create label: 0 for normal and sleeping, 1 for anomaly
    label = 1 if is_anomaly else 0
    
    merged_data.append({
        'Data': user_percentage,
        'Label': label
    })

# Create DataFrame and save to CSV
df = pd.DataFrame(merged_data)
output_path = os.path.join('Dataset_Creation', 'merged_test_CPU_1.csv')
df.to_csv(output_path, index=False)