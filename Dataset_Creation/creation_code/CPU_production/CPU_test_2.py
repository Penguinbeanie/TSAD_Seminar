import numpy as np
import time
from datetime import datetime, timedelta
import random
import csv
import os
import threading
import psutil # Import psutil

# Use a lock for thread-safe writing to the log file
log_lock = threading.Lock()

def log_entry(log_file, entry):
    """Thread-safe logging function - logs a list/tuple entry"""
    with log_lock:
        with open(log_file, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(entry)

# Global variable for continuous state (thread-safe access via logger)
# Possible values: 'idle', 'working_normal', 'working_anomaly'
current_execution_state = 'idle' # Initial state

# Flag for the logging thread
continue_logging = True

def continuous_logging():
    """Function to continuously log the current execution state and CPU usage"""
    # Initialize psutil's cpu_percent calculation - call it once to prime it
    # The first call returns a meaningless 0.0, subsequent calls calculate based on the interval
    psutil.cpu_percent(interval=None, percpu=False)

    last_log_time = datetime.now()
    while continue_logging:
        current_time = datetime.now()
        # Log every second
        if (current_time - last_log_time).total_seconds() >= 1.0:
            try:
                # Get overall system CPU usage (%) non-blocking
                # Use interval=None for non-blocking call after priming
                # percpu=False for total average CPU across all cores
                cpu_percent = psutil.cpu_percent(interval=None, percpu=False)

                # Log the sampled state AND CPU usage
                log_entry(log_file, [current_time.strftime('%Y-%m-%d %H:%M:%S'), current_execution_state, 'SAMPLED_STATE', '', cpu_percent])
                last_log_time = current_time
            except Exception as e:
                # Log any errors getting CPU info
                log_entry(log_file, [datetime.now().strftime('%Y-%m-%d %H:%M:%S'), 'error', 'LOGGER_ERROR', f'Failed to get CPU: {e}', ''])

        time.sleep(0.1) # Small sleep for responsiveness

# Set up the run times
total_runtime = timedelta(minutes=110)
initial_normal_period = timedelta(minutes=15)
start_time = datetime.now()

# List of matrix sizes
normal_sizes = [2800, 3000, 3200, 3400]
anomaly_sizes = [5000] 

# Create or open the CSV file for logging
log_file = 'execution_log_with_cpu.csv' # New file name
with open(log_file, 'w', newline='') as f:
    writer = csv.writer(f)
    # Added columns for event type, details, and CPU %
    writer.writerow(['timestamp', 'state', 'event_type', 'event_details', 'cpu_percent'])

print(f"Starting simulation. Total runtime: {total_runtime}, Initial normal period: {initial_normal_period}")
print(f"Logging to: {log_file}")

# Start the logging thread
logging_thread = threading.Thread(target=continuous_logging, daemon=True)
logging_thread.start()

# Log the start of the script
# Pass '' for cpu_percent as it's an event, not a sampled metric
log_entry(log_file, [datetime.now().strftime('%Y-%m-%d %H:%M:%S'), current_execution_state, 'SCRIPT_START', '', ''])

try:
    # Initial 15-minute normal period
    print(f"Entering initial normal period until {start_time + initial_normal_period}")
    while datetime.now() - start_time < initial_normal_period:
        matrix_size = random.choice(normal_sizes)
        workload_type = 'normal'

        # Log the start of the workload EVENT
        event_details = f'type:{workload_type},size:{matrix_size}'
        # Pass '' for cpu_percent as it's an event
        log_entry(log_file, [datetime.now().strftime('%Y-%m-%d %H:%M:%S'), current_execution_state, 'WORKLOAD_START', event_details, ''])

        # Update the shared state for the continuous logger during the workload
        current_execution_state = 'working_normal'

        # CPU intensive operation
        op_start_time = time.time()
        print(f"  Starting {workload_type} workload (size {matrix_size})...")
        matrix_a = np.random.rand(matrix_size, matrix_size)
        matrix_b = np.random.rand(matrix_size, matrix_size)

        for i in range(5):
            result = np.dot(matrix_a, matrix_b)
            if np.linalg.norm(result) != 0:
                 matrix_a = result / np.linalg.norm(result)
            print(f"    Dot product {i+1}/5 completed.")

        op_end_time = time.time()
        op_duration = op_end_time - op_start_time

        # Log the end of the workload EVENT
        event_details += f',duration:{op_duration:.2f}s'
        # Pass '' for cpu_percent as it's an event
        log_entry(log_file, [datetime.now().strftime('%Y-%m-%d %H:%M:%S'), current_execution_state, 'WORKLOAD_END', event_details, ''])

        # Clean up memory
        del result, matrix_a, matrix_b

        # Reset state to idle during sleep
        current_execution_state = 'idle'
        print(f"  Workload done. Sleeping for 4 seconds.")
        time.sleep(4)
        print(f"  Sleep finished.")


    # Continue with the normal/anomaly pattern for the remaining time
    print(f"Initial period finished. Entering mixed normal/anomaly period until {start_time + total_runtime}")
    while datetime.now() - start_time < total_runtime:
        random_choice = random.randint(1, 50)

        if random_choice == 20:
            matrix_size = random.choice(anomaly_sizes)
            workload_type = 'anomaly'
        else:
            matrix_size = random.choice(normal_sizes)
            workload_type = 'normal'

        # Log the start of the workload EVENT
        event_details = f'type:{workload_type},size:{matrix_size}'
        # Pass '' for cpu_percent as it's an event
        log_entry(log_file, [datetime.now().strftime('%Y-%m-%d %H:%M:%S'), current_execution_state, 'WORKLOAD_START', event_details, ''])

        # Update the shared state for the continuous logger *during* the workload
        current_execution_state = f'working_{workload_type}'

        # CPU intensive operation
        op_start_time = time.time()
        print(f"  Starting {workload_type} workload (size {matrix_size})...")
        matrix_a = np.random.rand(matrix_size, matrix_size)
        matrix_b = np.random.rand(matrix_size, matrix_size)

        for i in range(5):
            result = np.dot(matrix_a, matrix_b)
            if np.linalg.norm(result) != 0:
                matrix_a = result / np.linalg.norm(result)
            print(f"    Dot product {i+1}/5 completed.")

        op_end_time = time.time()
        op_duration = op_end_time - op_start_time

        # Log the end of the workload EVENT
        event_details += f',duration:{op_duration:.2f}s'
        # Pass '' for cpu_percent as it's an event
        log_entry(log_file, [datetime.now().strftime('%Y-%m-%d %H:%M:%S'), current_execution_state, 'WORKLOAD_END', event_details, ''])

        # Clean up memory
        del result, matrix_a, matrix_b

        # Reset state to idle during sleep
        current_execution_state = 'idle'
        print(f"  Workload done. Sleeping for 4 seconds.")
        time.sleep(4)
        print(f"  Sleep finished.")


finally:
    # Log the end of the script
    # Pass '' for cpu_percent as it's an event
    log_entry(log_file, [datetime.now().strftime('%Y-%m-%d %H:%M:%S'), current_execution_state, 'SCRIPT_END', '', ''])
    print("Shutting down logging thread...")
    # Clean up the logging thread
    continue_logging = False
    logging_thread.join(timeout=2) # Give it a moment to finish its last loop
    print("Script completed.")