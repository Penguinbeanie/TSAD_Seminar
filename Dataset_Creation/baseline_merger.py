import pandas as pd

# Load the datasets
df_data = pd.read_csv("Dataset_Creation/kai-VMware-Virtual-Platform_250523_2019.csv")
df_log = pd.read_csv("Dataset_Creation/execution_log.csv")

# Convert timestamp columns to datetime objects
df_data["time"] = pd.to_datetime(df_data["time"])
df_log["timestamp"] = pd.to_datetime(df_log["timestamp"])

# Merge the dataframes based on the timestamp
merged_df = pd.merge(df_data, df_log, left_on="time", right_on="timestamp", how="left")

# Create the initial 'Label' column
# Maps "normal" to 0 and "anomaly" to 1.
# Timestamps not in df_log (NaN state) are treated as "normal" by default.
merged_df["Label"] = merged_df["state"].fillna("normal").map({"normal": 0, "anomaly": 1})

# Identify rows that directly follow an anomaly
# Shift the 'Label' column down by one. If the previous row's label was 1,
# the current row (after shifting) will show 1. fill_value=0 handles the first row.
directly_after_anomaly = merged_df['Label'].shift(1, fill_value=0)

# Update the 'Label' column:
# A row becomes an anomaly if it was originally an anomaly OR if it directly follows an anomaly.
# Using bitwise OR for conciseness: 0|0=0, 0|1=1, 1|0=1, 1|1=1
merged_df['Label'] = merged_df['Label'] | directly_after_anomaly.astype(int)

# Select and rename columns for the final output
output_df = merged_df[["User%", "Label"]]
output_df = output_df.rename(columns={"User%": "Data"})

# Save the result
output_df.to_csv("Dataset_Creation/merged_test_1.csv", index=False)

print("Successfully created Dataset_Creation/merged_test_1.csv. Anomalies propagate to the immediately following row.")