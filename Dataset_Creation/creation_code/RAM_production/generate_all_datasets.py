import numpy as np
import time
from datetime import datetime, timedelta
import random
import csv
import os
import threading
import psutil

# Global lock for thread-safe writing to CSV files
LOG_LOCK = threading.Lock()

# Global dictionary to hold current execution state and continue_logging flag for each active generator
# Key: function_id (e.g., "dataset1"), Value: {"current_execution_state": "idle", "continue_logging": True, "log_file": "path/to/log.csv"}
THREAD_STATES = {}

OUTPUT_BASE_DIR = "/home/kai/Documents/TSAD/TSAD_Seminar/Dataset_Creation/prelim_datasets"

def ensure_output_dir():
    """Ensures the output directory exists."""
    os.makedirs(OUTPUT_BASE_DIR, exist_ok=True)

def log_entry(function_id, event_type, event_details="", ram_usage=""):
    """Thread-safe logging function for a specific dataset generator."""
    state_info = THREAD_STATES.get(function_id)
    if not state_info:
        print(f"Error: No state info found for {function_id} in log_entry.")
        return

    log_file = state_info["log_file"]
    current_state = state_info["current_execution_state"]
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    
    with LOG_LOCK:
        try:
            with open(log_file, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([timestamp, current_state, event_type, event_details, ram_usage])
        except Exception as e:
            # Fallback logging in case file logging fails
            print(f"CRITICAL_LOG_FAIL for {function_id}: {timestamp}, {current_state}, {event_type}, {event_details}, {ram_usage}, ERROR: {e}")


def continuous_logger_thread(function_id):
    """
    Continuously logs the execution state and RAM usage for a specific dataset generator.
    """
    psutil.Process(os.getpid()) # Prime psutil for the process

    last_log_time = datetime.min
    previous_state_in_logger = 'uninitialized'
    workload_start_detected_by_logger_time = None
    workload_end_detected_by_logger_time = None
    WORKLOAD_LOG_DELAY_SECONDS = 0.3
    WORKLOAD_LOG_DELAY_SECONDS_AFTER_END = 0.1

    state_info = THREAD_STATES.get(function_id)
    if not state_info:
        print(f"Error: No state info for {function_id} at logger start.")
        return

    while state_info.get("continue_logging", False): # Check flag from shared state
        current_time = datetime.now()
        # Ensure we fetch the most current state for this specific function
        current_main_thread_state = THREAD_STATES.get(function_id, {}).get("current_execution_state", "unknown")


        is_currently_working = current_main_thread_state.startswith('working_')
        was_previously_working = previous_state_in_logger.startswith('working_')

        if is_currently_working and not was_previously_working:
            workload_start_detected_by_logger_time = current_time
            workload_end_detected_by_logger_time = None
        elif not is_currently_working and was_previously_working:
            workload_end_detected_by_logger_time = current_time
            workload_start_detected_by_logger_time = None

        time_since_last_log_seconds = (current_time - last_log_time).total_seconds()
        should_log_this_iteration = False

        if time_since_last_log_seconds >= 1.0:
            if is_currently_working and workload_start_detected_by_logger_time:
                if (current_time - workload_start_detected_by_logger_time).total_seconds() >= WORKLOAD_LOG_DELAY_SECONDS:
                    should_log_this_iteration = True
            elif not is_currently_working and workload_end_detected_by_logger_time:
                if (current_time - workload_end_detected_by_logger_time).total_seconds() >= WORKLOAD_LOG_DELAY_SECONDS_AFTER_END:
                    should_log_this_iteration = True
                    workload_end_detected_by_logger_time = None
            elif not is_currently_working and not was_previously_working and current_main_thread_state == 'idle': # Ensure idle is logged
                 should_log_this_iteration = True
            elif current_main_thread_state != 'idle': # Log other non-working states if not just logged post-workload
                 should_log_this_iteration = True


        if should_log_this_iteration:
            try:
                process = psutil.Process(os.getpid())
                ram_usage_mb = process.memory_info().rss / 1024 / 1024
                log_entry(function_id, 'SAMPLED_STATE', ram_usage=f'{ram_usage_mb:.2f}')
                last_log_time = current_time
            except Exception as e:
                log_entry(function_id, 'LOGGER_ERROR', f'Failed to get RAM: {e}')
        
        previous_state_in_logger = current_main_thread_state
        time.sleep(0.1)
    
    print(f"Logging thread for {function_id} is stopping.")


def perform_matrix_operation(matrix_size, num_matrices=3, num_operations=5):
    """Performs matrix multiplication and normalization to consume RAM and CPU."""
    op_start_time = time.time()
    try:
        matrices = [np.random.rand(matrix_size, matrix_size) for _ in range(num_matrices)]
        if not matrices: # Should not happen with num_matrices > 0
            return 0
        
        result = matrices[0]
        for _ in range(num_operations): # Renamed loop variable to avoid conflict
            # Ensure there's a second matrix for dot product if num_matrices > 1
            matrix_to_multiply = matrices[1] if num_matrices > 1 else matrices[0] 
            result = np.dot(result, matrix_to_multiply)
            norm = np.linalg.norm(result)
            if norm != 0:
                result = result / norm
            else: # Handle zero norm case to avoid division by zero
                # print(f"Warning: Zero norm encountered for matrix size {matrix_size}. Re-initializing result.")
                result = np.random.rand(matrix_size, matrix_size) # Re-initialize to avoid stuck state
    except MemoryError:
        print(f"MemoryError during matrix operation with size {matrix_size}. Skipping this operation.")
        return time.time() - op_start_time # Return duration so far
    finally:
        # Explicitly delete large objects
        # The 'matrices' and 'result' will be cleaned up when they go out of scope,
        # but del can be used for more immediate cleanup if needed, though often not necessary.
        # Python's garbage collector handles this. Forcing with gc.collect() usually not recommended.
        pass
    
    op_duration = time.time() - op_start_time
    return op_duration

def update_state(function_id, new_state):
    """Updates the execution state for a given function."""
    if function_id in THREAD_STATES:
        THREAD_STATES[function_id]["current_execution_state"] = new_state
    else:
        print(f"Error: Attempted to update state for unknown function_id: {function_id}")

def run_workload_loop(function_id, config):
    """
    Main loop for generating a single dataset.
    Manages normal operation, anomaly injection, and logging.
    """
    log_file_name = f"dataset_{config['id_suffix']}_log.csv"
    log_file_path = os.path.join(OUTPUT_BASE_DIR, log_file_name)

    # Initialize state for this function
    THREAD_STATES[function_id] = {
        "current_execution_state": "idle",
        "continue_logging": True,
        "log_file": log_file_path
    }

    with open(log_file_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['timestamp', 'state', 'event_type', 'event_details', 'ram_usage_mb'])
    
    print(f"Starting dataset generation for {function_id}. Logging to: {log_file_path}")
    update_state(function_id, 'idle')
    log_entry(function_id, 'SCRIPT_START', f"Dataset {config['id_suffix']}")

    # Start dedicated logging thread
    logging_thread = threading.Thread(target=continuous_logger_thread, args=(function_id,), daemon=True)
    logging_thread.start()

    start_time = datetime.now()
    total_runtime = timedelta(minutes=config['total_runtime_min'])
    initial_normal_period = timedelta(minutes=config['initial_normal_min'])
    
    normal_sizes = config['normal_sizes']
    anomaly_config = config['anomaly_config']
    base_sleep_time = config.get('base_sleep_time_s', 4)

    # Memory preparation if specified
    if config.get('prep_memory_anomaly_size'):
        prep_size = config['prep_memory_anomaly_size']
        num_matrices_prep = config.get('prep_num_matrices', 4) # Typically higher for anomaly
        print(f"  {function_id} MAIN: Preparing memory with anomaly matrix size {prep_size}...")
        update_state(function_id, 'working_special_prep')
        log_entry(function_id, 'WORKLOAD_START', f'type:prep_anomaly,size:{prep_size}')
        duration = perform_matrix_operation(prep_size, num_matrices=num_matrices_prep)
        log_entry(function_id, 'WORKLOAD_END', f'type:prep_anomaly,size:{prep_size},duration:{duration:.2f}s')
        update_state(function_id, 'idle')
        print(f"  {function_id} MAIN: Memory preparation done. Duration {duration:.2f}s.")

    try:
        # Initial normal period
        print(f"  {function_id} MAIN: Entering initial normal period ({config['initial_normal_min']} min)")
        while datetime.now() - start_time < initial_normal_period:
            matrix_size = random.choice(normal_sizes)
            current_sleep_time = base_sleep_time
            
            update_state(function_id, 'working_normal')
            event_details = f'type:normal,size:{matrix_size}'
            log_entry(function_id, 'WORKLOAD_START', event_details)
            # print(f"  {function_id} MAIN: Starting normal workload (size {matrix_size}). State: {THREAD_STATES[function_id]['current_execution_state']}")
            
            duration = perform_matrix_operation(matrix_size)
            
            event_details += f',duration:{duration:.2f}s'
            log_entry(function_id, 'WORKLOAD_END', event_details)
            
            update_state(function_id, 'idle')
            # print(f"  {function_id} MAIN: Workload done. Duration {duration:.2f}s. State: {THREAD_STATES[function_id]['current_execution_state']}. Sleeping for {current_sleep_time}s.")
            time.sleep(current_sleep_time)

        # Mixed normal/anomaly period
        print(f"  {function_id} MAIN: Initial normal period finished. Entering mixed normal/anomaly period.")
        anomaly_decision_range = config.get('anomaly_decision_range', 50) # e.g. 1 to 50 or 1 to 150

        while datetime.now() - start_time < total_runtime:
            workload_type = 'normal'
            matrix_size = random.choice(normal_sizes)
            current_sleep_time = base_sleep_time
            num_matrices_op = 3 # Default for normal
            
            # Anomaly decision
            # For function 4, anomaly_config will be a list of choices. Others a dict.
            chosen_anomaly_type = None
            if isinstance(anomaly_config, list): # Function 4 case
                if random.randint(1, anomaly_decision_range) == 1: # Adjust trigger as needed
                    chosen_anomaly_def = random.choice(anomaly_config)
                    workload_type = chosen_anomaly_def['type']
                    chosen_anomaly_type = workload_type # for logging details
            elif anomaly_config.get('size') or anomaly_config.get('sleep_increase_s'): # Func 1, 2, 3
                if random.randint(1, anomaly_decision_range) == 1: # Adjust trigger
                    workload_type = anomaly_config['type']
                    chosen_anomaly_type = workload_type


            if workload_type == 'ram_spike': # Function 1 type anomaly
                matrix_size = random.choice(anomaly_config['size'])
                num_matrices_op = anomaly_config.get('num_matrices', 4) # More matrices for RAM spike
                update_state(function_id, 'working_anomaly_ram_spike')
                event_details = f'type:anomaly_ram_spike,size:{matrix_size}'
            elif workload_type == 'sleep_increase': # Function 2 type anomaly
                current_sleep_time += anomaly_config['sleep_increase_s']
                # Matrix size remains normal for this anomaly type
                update_state(function_id, 'working_anomaly_sleep')
                event_details = f'type:anomaly_sleep_increase,sleep_time:{current_sleep_time}s'
                log_entry(function_id, 'WORKLOAD_START', event_details)
                # Skip matrix operation for sleep anomaly - just sleep
                duration = current_sleep_time
                event_details_end = event_details + f',duration:{duration:.2f}s,anomaly_trigger:sleep_increase'
                log_entry(function_id, 'WORKLOAD_END', event_details_end)
                update_state(function_id, 'idle')
                time.sleep(current_sleep_time)
                continue  # Skip the rest of the loop
            elif workload_type == 'medium_ram_spike': # Function 3 type anomaly
                matrix_size = random.choice(anomaly_config['size'])
                num_matrices_op = anomaly_config.get('num_matrices', 3) # Default or specific
                update_state(function_id, 'working_anomaly_medium_ram_spike')
                event_details = f'type:anomaly_medium_ram_spike,size:{matrix_size}'
            else: # Normal workload
                update_state(function_id, 'working_normal')
                event_details = f'type:normal,size:{matrix_size}'

            log_entry(function_id, 'WORKLOAD_START', event_details)
            # print(f"  {function_id} MAIN: Starting {workload_type} workload (details: {event_details}). State: {THREAD_STATES[function_id]['current_execution_state']}")
            
            duration = perform_matrix_operation(matrix_size, num_matrices=num_matrices_op)
            
            event_details_end = event_details + f',duration:{duration:.2f}s'
            if chosen_anomaly_type: # Add anomaly type to end log if it was an anomaly
                 event_details_end += f',anomaly_trigger:{chosen_anomaly_type}'
            log_entry(function_id, 'WORKLOAD_END', event_details_end)
            
            update_state(function_id, 'idle')
            # print(f"  {function_id} MAIN: Workload done. Duration {duration:.2f}s. State: {THREAD_STATES[function_id]['current_execution_state']}. Sleeping for {current_sleep_time}s.")
            time.sleep(current_sleep_time)

    except KeyboardInterrupt:
        print(f"  {function_id} MAIN: Keyboard interrupt detected. Shutting down this generator.")
    finally:
        update_state(function_id, 'ending_script')
        log_entry(function_id, 'SCRIPT_END', f"Dataset {config['id_suffix']} generation ended.")
        
        if function_id in THREAD_STATES:
            THREAD_STATES[function_id]["continue_logging"] = False
        
        print(f"  {function_id} MAIN: Waiting for logging thread to finish...")
        if logging_thread.is_alive():
            logging_thread.join(timeout=5)
        if logging_thread.is_alive():
            print(f"  {function_id} WARNING: Logging thread did not terminate cleanly.")
        
        # Clean up global state entry for this function to free memory
        if function_id in THREAD_STATES:
            del THREAD_STATES[function_id]
        print(f"Dataset generation {function_id} completed.")


# --- Dataset Specific Configurations ---

def generate_dataset_1():
    """Anomaly: Larger matrix multiplication."""
    config = {
        'id_suffix': '1_ram_spike',
        'total_runtime_min': 90,
        'initial_normal_min': 20,
        'normal_sizes': [3000, 3500, 4500, 5000],
        'anomaly_config': {'type': 'ram_spike', 'size': [6000], 'num_matrices': 4}, # More matrices for higher RAM
        'prep_memory_anomaly_size': 6000, # Prepare with anomaly size
        'prep_num_matrices': 4,
        'anomaly_decision_range': 50,
    }
    run_workload_loop('dataset1', config)

def generate_dataset_2():
    """Anomaly: Increased sleep timer instead of larger matrix."""
    config = {
        'id_suffix': '2_sleep_anomaly',
        'total_runtime_min': 90,
        'initial_normal_min': 20,
        'normal_sizes': [3000, 3500, 4500, 5000],
        # Matrix size for "anomaly" event is normal, but sleep increases
        'anomaly_config': {'type': 'sleep_increase', 'sleep_increase_s': 2},
        'base_sleep_time_s': 4,
        'anomaly_decision_range': 50,
        # No specific RAM prep needed as anomaly isn't RAM intensive itself
    }
    run_workload_loop('dataset2', config)

def generate_dataset_3():
    """Anomaly: Matrix calculation of medium size."""
    config = {
        'id_suffix': '3_medium_ram_spike',
        'total_runtime_min': 90,
        'initial_normal_min': 20,
        'normal_sizes': [3000, 3500, 4500, 5000],
        'anomaly_config': {'type': 'medium_ram_spike', 'size': [4000], 'num_matrices': 3},
        'anomaly_decision_range': 50,
        # No specific prep needed, or prep with its own anomaly size if desired
        # 'prep_memory_anomaly_size': 4000,
    }
    run_workload_loop('dataset3', config)

def generate_dataset_4():
    """All three previous anomalies included, one selected at random. Longer runtime."""
    config = {
        'id_suffix': '4_mixed_anomalies',
        'total_runtime_min': 160,
        'initial_normal_min': 35,
        'normal_sizes': [3000, 3500, 4500, 5000],
        'anomaly_config': [ # List of anomaly definitions to choose from
            {'type': 'ram_spike', 'size': [6000], 'num_matrices': 4},
            {'type': 'sleep_increase', 'sleep_increase_s': 2}, # Uses normal matrix size
            {'type': 'medium_ram_spike', 'size': [4000], 'num_matrices': 3},
        ],
        'base_sleep_time_s': 4,
        'prep_memory_anomaly_size': 6000, # Prepare with the largest RAM anomaly
        'prep_num_matrices': 4,
        'anomaly_decision_range': 150, # Increased range for same anomaly frequency
    }
    run_workload_loop('dataset4', config)

def main():
    """Main function to trigger dataset generations."""
    ensure_output_dir()
    print(f"Global output directory set to: {OUTPUT_BASE_DIR}")


    
    print("Starting Dataset 1 Generation...")
    generate_dataset_1()
    print("\nDataset 1 Generation Finished.\n")
    
    print("Starting Dataset 2 Generation...")
    generate_dataset_2()
    print("\nDataset 2 Generation Finished.\n")

    print("Starting Dataset 3 Generation...")
    generate_dataset_3()
    print("\nDataset 3 Generation Finished.\n")
    
    print("Starting Dataset 4 Generation...") # As per user focus
    generate_dataset_4()
    print("\nDataset 4 Generation Finished.\n")

    # Example: If you want to run only one for testing:
    # generate_dataset_1()

    print("All dataset generation processes have been initiated (or completed if run sequentially).")

if __name__ == '__main__':
    # Note: The brief print statements within the main() function for starting/finishing each dataset
    # are placeholders. The actual detailed logging to CSV and console happens within each generate_dataset_*
    # and its helper run_workload_loop.
    # Consider enabling one dataset generation at a time for testing, or manage system load if running all.
    
    # For demonstration, I'm only calling one function.
    # You can uncomment others as needed. Be mindful of total execution time.
    
    main() 