import csv
from datetime import datetime
import os

def parse_nmon_to_csv(nmon_file_path):
    # Get the output CSV path (same name, different extension)
    csv_file_path = os.path.splitext(nmon_file_path)[0] + '.csv'
    
    # Dictionary to store CPU data
    cpu_data = {}
    
    # Read the nmon file
    with open(nmon_file_path, 'r') as f:
        for line in f:
            parts = line.strip().split(',')
            
            # Check if it's a timestamp line
            if parts[0] == 'ZZZZ':
                timestamp = parts[1]  # T0061 format
                date = parts[3]  # 23-MAY-2025 format
                time = parts[2]  # 15:11:44 format
                
                # Convert to datetime
                dt = datetime.strptime(f"{date} {time}", "%d-%b-%Y %H:%M:%S")
                formatted_time = dt.strftime("%Y-%m-%d %H:%M:%S")
                
                # Initialize entry for this timestamp
                cpu_data[timestamp] = {'time': formatted_time}
            
            # Check if it's CPU_ALL data
            elif parts[0] == 'CPU_ALL':
                timestamp = parts[1]
                if timestamp in cpu_data:
                    cpu_data[timestamp]['User%'] = float(parts[2])
    
    # Write to CSV
    with open(csv_file_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['time', 'User%'])
        writer.writeheader()
        
        # Write data in chronological order
        for timestamp in sorted(cpu_data.keys()):
            writer.writerow(cpu_data[timestamp])

# Use the script
nmon_file_path = r"C:\Users\Kai\Documents\Time_Series_Anomaly_Detection\TSAD_Seminar\Dataset_Creation\kai-VMware-Virtual-Platform_250523_1704.nmon"
parse_nmon_to_csv(nmon_file_path)