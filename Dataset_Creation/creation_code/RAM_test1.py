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
current_execution_state = 'idle' # Initial state

# Flag for the logging thread
continue_logging = True

def continuous_logging():
    """Function to continuously log the current execution state and RAM usage,
    with a delay for the first log of a new workload and after workload ends."""
    last_log_time = datetime.min
    previous_state_in_logger = 'uninitialized'
    workload_start_detected_by_logger_time = None
    workload_end_detected_by_logger_time = None  # New variable to track workload end
    WORKLOAD_LOG_DELAY_SECONDS = 0.3
    WORKLOAD_LOG_DELAY_SECONDS_AFTER_END = 0.1

    while continue_logging:
        current_time = datetime.now()
        current_main_thread_state = current_execution_state

        # Detect transitions
        is_currently_working = current_main_thread_state.startswith('working_')
        was_previously_working = previous_state_in_logger.startswith('working_')

        # Detect start of workload
        if is_currently_working and not was_previously_working:
            workload_start_detected_by_logger_time = current_time
            workload_end_detected_by_logger_time = None  # Reset end detection

        # Detect end of workload
        elif not is_currently_working and was_previously_working:
            workload_end_detected_by_logger_time = current_time
            workload_start_detected_by_logger_time = None  # Reset start detection

        # Determine if it's time to log
        time_since_last_log_seconds = (current_time - last_log_time).total_seconds()
        should_log_this_iteration = False

        if time_since_last_log_seconds >= 1.0:  # Basic 1-second interval met
            if is_currently_working and workload_start_detected_by_logger_time:
                # Check delay after workload start
                time_since_workload_detected_by_logger = (current_time - workload_start_detected_by_logger_time).total_seconds()
                if time_since_workload_detected_by_logger >= WORKLOAD_LOG_DELAY_SECONDS:
                    should_log_this_iteration = True
            elif not is_currently_working and workload_end_detected_by_logger_time:
                # Check delay after workload end
                time_since_workload_end = (current_time - workload_end_detected_by_logger_time).total_seconds()
                if time_since_workload_end >= WORKLOAD_LOG_DELAY_SECONDS_AFTER_END:
                    should_log_this_iteration = True
                    workload_end_detected_by_logger_time = None  # Reset after first post-workload log
            else:
                # Regular logging for ongoing states
                should_log_this_iteration = True

        if should_log_this_iteration:
            try:
                # Get RAM usage instead of CPU
                process = psutil.Process(os.getpid())
                ram_usage = process.memory_info().rss / 1024 / 1024  # Convert to MB
                log_entry(log_file, [current_time.strftime('%Y-%m-%d %H:%M:%S'), current_main_thread_state, 'SAMPLED_STATE', '', ram_usage])
                last_log_time = current_time

            except Exception as e:
                log_entry(log_file, [datetime.now().strftime('%Y-%m-%d %H:%M:%S'), 'error', 'LOGGER_ERROR', f'Failed to get RAM: {e}', ''])

        previous_state_in_logger = current_main_thread_state
        time.sleep(0.1)

# Set up the run times
total_runtime = timedelta(minutes=110) # Shortened for testing
initial_normal_period = timedelta(minutes=25) # Shortened for testing
start_time = datetime.now()

# List of matrix sizes - adjusted for RAM consumption
normal_sizes = [3000, 4000, 5000, 6000]  # Larger matrices for more RAM usage
anomaly_sizes = [8000]  # Even larger for anomalies

# Create or open the CSV file for logging
log_file = 'ram_execution_log_1.csv'  # New file name for RAM monitoring
with open(log_file, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['timestamp', 'state', 'event_type', 'event_details', 'ram_usage_mb'])

print(f"Starting RAM usage simulation. Total runtime: {total_runtime}, Initial normal period: {initial_normal_period}")
print(f"Logging to: {log_file}")

# Start the logging thread
logging_thread = threading.Thread(target=continuous_logging, daemon=True)
logging_thread.start()

log_entry(log_file, [datetime.now().strftime('%Y-%m-%d %H:%M:%S'), current_execution_state, 'SCRIPT_START', '', ''])

try:
    # Initial normal period
    print(f"Entering initial normal period until {start_time + initial_normal_period}")
    while datetime.now() - start_time < initial_normal_period:
        matrix_size = random.choice(normal_sizes)
        workload_type = 'normal'

        # State change happens *before* WORKLOAD_START log for main thread events
        current_execution_state = 'working_normal'
        event_details = f'type:{workload_type},size:{matrix_size}'
        log_entry(log_file, [datetime.now().strftime('%Y-%m-%d %H:%M:%S'), current_execution_state, 'WORKLOAD_START', event_details, ''])
        print(f"  MAIN: Starting {workload_type} workload (size {matrix_size}). State: {current_execution_state}")

        op_start_time = time.time()
        # Create multiple matrices to increase RAM usage
        matrices = [np.random.rand(matrix_size, matrix_size) for _ in range(3)]
        result = matrices[0]
        for i in range(5):
            result = np.dot(result, matrices[1])
            if np.linalg.norm(result) != 0:
                result = result / np.linalg.norm(result)
        op_end_time = time.time()
        op_duration = op_end_time - op_start_time

        # WORKLOAD_END logged while state is still 'working_normal'
        event_details += f',duration:{op_duration:.2f}s'
        log_entry(log_file, [datetime.now().strftime('%Y-%m-%d %H:%M:%S'), current_execution_state, 'WORKLOAD_END', event_details, ''])
        del result, matrices

        current_execution_state = 'idle'
        print(f"  MAIN: Workload done. Duration {op_duration:.2f}s. State: {current_execution_state}. Sleeping for 4 seconds.")
        time.sleep(4)

    print(f"Initial period finished. Entering mixed normal/anomaly period until {start_time + total_runtime}")
    while datetime.now() - start_time < total_runtime:
        random_choice = random.randint(1, 50)
        if random_choice == 20:  # High RAM usage anomaly
            matrix_size = random.choice(anomaly_sizes)
            workload_type = 'anomaly'
            current_execution_state = 'working_anomaly'
            
            event_details = f'type:{workload_type},size:{matrix_size}'
            log_entry(log_file, [datetime.now().strftime('%Y-%m-%d %H:%M:%S'), current_execution_state, 'WORKLOAD_START', event_details, ''])
            print(f"  MAIN: Starting {workload_type} workload (size {matrix_size}). State: {current_execution_state}")

            op_start_time = time.time()
            # Create more matrices for anomaly to use more RAM
            matrices = [np.random.rand(matrix_size, matrix_size) for _ in range(4)]
            result = matrices[0]
            for i in range(5):
                result = np.dot(result, matrices[1])
                if np.linalg.norm(result) != 0:
                    result = result / np.linalg.norm(result)
            op_end_time = time.time()
            op_duration = op_end_time - op_start_time
            
            # Clean up matrices
            del result, matrices

        else:  # Normal workload
            matrix_size = random.choice(normal_sizes)
            workload_type = 'normal'
            current_execution_state = 'working_normal'
            
            event_details = f'type:{workload_type},size:{matrix_size}'
            log_entry(log_file, [datetime.now().strftime('%Y-%m-%d %H:%M:%S'), current_execution_state, 'WORKLOAD_START', event_details, ''])
            print(f"  MAIN: Starting {workload_type} workload (size {matrix_size}). State: {current_execution_state}")

            op_start_time = time.time()
            matrices = [np.random.rand(matrix_size, matrix_size) for _ in range(3)]
            result = matrices[0]
            for i in range(5):
                result = np.dot(result, matrices[1])
                if np.linalg.norm(result) != 0:
                    result = result / np.linalg.norm(result)
            op_end_time = time.time()
            op_duration = op_end_time - op_start_time
            
            # Clean up matrices
            del result, matrices

        event_details += f',duration:{op_duration:.2f}s'
        log_entry(log_file, [datetime.now().strftime('%Y-%m-%d %H:%M:%S'), current_execution_state, 'WORKLOAD_END', event_details, ''])

        current_execution_state = 'idle'
        print(f"  MAIN: Workload done. Duration {op_duration:.2f}s. State: {current_execution_state}. Sleeping for 4 seconds.")
        time.sleep(4)

finally:
    current_execution_state = 'ending_script'  # Final state
    log_entry(log_file, [datetime.now().strftime('%Y-%m-%d %H:%M:%S'), current_execution_state, 'SCRIPT_END', '', ''])
    print("Shutting down logging thread...")
    continue_logging = False
    logging_thread.join(timeout=3)  # Give it a moment to finish its last loop and log
    print("Script completed.")