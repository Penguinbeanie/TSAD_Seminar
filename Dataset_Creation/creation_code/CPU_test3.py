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
    """Function to continuously log the current execution state and CPU usage,
    with a delay for the first log of a new workload."""
    psutil.cpu_percent(interval=None, percpu=False) # Prime psutil

    # Initialize last_log_time to allow the first log to occur if conditions met
    last_log_time = datetime.min # Will ensure first log can happen
    # Track the state as seen by the logger in the previous iteration
    previous_state_in_logger = 'uninitialized' # Different from initial 'idle' to detect first change
    # Timestamp when the logger itself detected the start of a new workload
    workload_start_detected_by_logger_time = None
    # The minimum delay after logger detects workload start before logging its CPU
    WORKLOAD_LOG_DELAY_SECONDS = 0.3

    while continue_logging:
        current_time = datetime.now()
        # Snapshot of the global state, which can change asynchronously
        current_main_thread_state = current_execution_state

        # --- Detect transition to a new workload (as seen by this logging thread) ---
        is_currently_working = current_main_thread_state.startswith('working_')
        was_previously_working = previous_state_in_logger.startswith('working_')

        if is_currently_working and not was_previously_working:
            # Transitioned from non-working to working
            workload_start_detected_by_logger_time = current_time
            # print(f"DEBUG LOGGER: New workload '{current_main_thread_state}' detected at {workload_start_detected_by_logger_time}")
        elif not is_currently_working and workload_start_detected_by_logger_time:
            # Transitioned from working to non-working, reset detection time
            workload_start_detected_by_logger_time = None
            # print(f"DEBUG LOGGER: Workload '{previous_state_in_logger}' ended. Resetting detection time.")

        # --- Determine if it's time to log ---
        time_since_last_log_seconds = (current_time - last_log_time).total_seconds()
        should_log_this_iteration = False

        if time_since_last_log_seconds >= 1.0: # Basic 1-second interval met
            if is_currently_working and workload_start_detected_by_logger_time:
                # Currently in a workload, and the logger has a "start detection" time for it.
                # This means this is potentially the first log for this *detected* workload.
                time_since_workload_detected_by_logger = (current_time - workload_start_detected_by_logger_time).total_seconds()
                if time_since_workload_detected_by_logger >= WORKLOAD_LOG_DELAY_SECONDS:
                    should_log_this_iteration = True
                # else: 1s interval met, but 0.3s specific workload delay not met yet. Skip.
                # print(f"DEBUG LOGGER: Workload active. Time since detection: {time_since_workload_detected_by_logger:.2f}s. Delay met: {should_log_this_iteration}")
            else:
                # Not in a 'working' state, or it's a 'working' state but not "newly detected" by logger
                # (i.e., workload_start_detected_by_logger_time is None, meaning it's an ongoing workload
                # for which the initial log already happened or it's an idle state).
                should_log_this_iteration = True
                # print(f"DEBUG LOGGER: Not a new workload or idle. Standard 1s log. State: {current_main_thread_state}")


        if should_log_this_iteration:
            try:
                cpu_percent = psutil.cpu_percent(interval=None, percpu=False)
                log_entry(log_file, [current_time.strftime('%Y-%m-%d %H:%M:%S'), current_main_thread_state, 'SAMPLED_STATE', '', cpu_percent])
                last_log_time = current_time
                # print(f"DEBUG LOGGER: Logged at {current_time}, State: {current_main_thread_state}, CPU: {cpu_percent}")

                # If we just logged for a workload that was newly detected,
                # we don't want the 0.3s delay logic to re-apply for *subsequent* logs of this same workload instance.
                # The `workload_start_detected_by_logger_time` will remain set as long as the state is 'working_',
                # so `time_since_workload_detected_by_logger` will keep increasing, satisfying the 0.3s condition.
                # The main gate becomes `time_since_last_log_seconds >= 1.0`. This is correct.

            except Exception as e:
                log_entry(log_file, [datetime.now().strftime('%Y-%m-%d %H:%M:%S'), 'error', 'LOGGER_ERROR', f'Failed to get CPU: {e}', ''])

        previous_state_in_logger = current_main_thread_state # Update for the next iteration
        time.sleep(0.1) # Polling interval for the logger

# Set up the run times
total_runtime = timedelta(minutes=110) # Shortened for testing
initial_normal_period = timedelta(minutes=15) # Shortened for testing
start_time = datetime.now()

# List of matrix sizes
normal_sizes = [2800, 3000, 3200, 3400]
anomaly_sizes = [5000]

# Create or open the CSV file for logging
log_file = 'execution_log_3.csv' # New file name
with open(log_file, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['timestamp', 'state', 'event_type', 'event_details', 'cpu_percent'])

print(f"Starting simulation. Total runtime: {total_runtime}, Initial normal period: {initial_normal_period}")
print(f"Logging to: {log_file}")

# Start the logging thread
logging_thread = threading.Thread(target=continuous_logging, daemon=True)
logging_thread.start()

log_entry(log_file, [datetime.now().strftime('%Y-%m-%d %H:%M:%S'), current_execution_state, 'SCRIPT_START', '', ''])

try:
    # Initial 15-minute normal period
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
        matrix_a = np.random.rand(matrix_size, matrix_size)
        matrix_b = np.random.rand(matrix_size, matrix_size)
        for i in range(5):
            result = np.dot(matrix_a, matrix_b)
            if np.linalg.norm(result) != 0:
                 matrix_a = result / np.linalg.norm(result)
            # print(f"    Dot product {i+1}/5 completed.")
        op_end_time = time.time()
        op_duration = op_end_time - op_start_time

        # WORKLOAD_END logged while state is still 'working_normal'
        event_details += f',duration:{op_duration:.2f}s'
        log_entry(log_file, [datetime.now().strftime('%Y-%m-%d %H:%M:%S'), current_execution_state, 'WORKLOAD_END', event_details, ''])
        del result, matrix_a, matrix_b

        current_execution_state = 'idle'
        print(f"  MAIN: Workload done. Duration {op_duration:.2f}s. State: {current_execution_state}. Sleeping for 4 seconds.")
        time.sleep(4)
        # print(f"  MAIN: Sleep finished.")

    print(f"Initial period finished. Entering mixed normal/anomaly period until {start_time + total_runtime}")
    while datetime.now() - start_time < total_runtime:
        random_choice = random.randint(1, 50) # Adjusted for potentially longer runs or more anomalies
        if random_choice == 20: # Anomaly
            matrix_size = random.choice(anomaly_sizes)
            workload_type = 'anomaly'
            current_execution_state = 'working_anomaly'
        else: # Normal
            matrix_size = random.choice(normal_sizes)
            workload_type = 'normal'
            current_execution_state = 'working_normal'

        event_details = f'type:{workload_type},size:{matrix_size}'
        log_entry(log_file, [datetime.now().strftime('%Y-%m-%d %H:%M:%S'), current_execution_state, 'WORKLOAD_START', event_details, ''])
        print(f"  MAIN: Starting {workload_type} workload (size {matrix_size}). State: {current_execution_state}")

        op_start_time = time.time()
        matrix_a = np.random.rand(matrix_size, matrix_size)
        matrix_b = np.random.rand(matrix_size, matrix_size)
        for i in range(5):
            result = np.dot(matrix_a, matrix_b)
            if np.linalg.norm(result) != 0:
                matrix_a = result / np.linalg.norm(result)
            # print(f"    Dot product {i+1}/5 completed.")
        op_end_time = time.time()
        op_duration = op_end_time - op_start_time

        event_details += f',duration:{op_duration:.2f}s'
        log_entry(log_file, [datetime.now().strftime('%Y-%m-%d %H:%M:%S'), current_execution_state, 'WORKLOAD_END', event_details, ''])
        del result, matrix_a, matrix_b

        current_execution_state = 'idle'
        print(f"  MAIN: Workload done. Duration {op_duration:.2f}s. State: {current_execution_state}. Sleeping for 4 seconds.")
        time.sleep(4)
        # print(f"  MAIN: Sleep finished.")

finally:
    current_execution_state = 'ending_script' # Final state
    log_entry(log_file, [datetime.now().strftime('%Y-%m-%d %H:%M:%S'), current_execution_state, 'SCRIPT_END', '', ''])
    print("Shutting down logging thread...")
    continue_logging = False
    logging_thread.join(timeout=3) # Give it a moment to finish its last loop and log
    print("Script completed.")