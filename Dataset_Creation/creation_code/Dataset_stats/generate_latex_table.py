import pandas as pd
import os

def generate_latex_table(csv_path, output_path=None):
    """
    Generates a LaTeX table from a CSV file.

    Args:
        csv_path (str): The path to the input CSV file.
        output_path (str, optional): The path to save the .tex file. 
                                     If None, prints to console. 
                                     Defaults to None.
    """
    print(f"Attempting to read CSV from: {csv_path}")
    if not os.path.exists(csv_path):
        print(f"Error: File does not exist at the specified path: {csv_path}")
        return

    try:
        df = pd.read_csv(csv_path)
        print("Successfully read CSV file.")
    except Exception as e:
        print(f"Error reading or processing CSV file: {e}")
        return

    # Select the required columns
    columns = ['Model', 'AUC-PR', 'AUC-ROC', 'VUS-PR', 'VUS-ROC']
    if not all(col in df.columns for col in columns):
        print(f"Error: CSV must contain the columns: {', '.join(columns)}")
        return
        
    df_table = df[columns]
    print("Successfully selected columns for the table.")

    # Start LaTeX table
    latex_string = "\\begin{table}[htbp]\n"
    latex_string += "\\centering\n"
    
    # Extract dataset name for caption
    dataset_name = os.path.basename(csv_path).replace('_', '\\_').replace('.csv', '')
    latex_string += f"\\caption{{Model Performance Comparison on {dataset_name}}}\n"
    latex_string += f"\\label{{tab:model_performance_{os.path.basename(csv_path).split('_')[1]}}}\n"
    
    latex_string += "\\begin{tabular}{|l|c|c|c|c|}\n"
    latex_string += "\\hline\n"
    latex_string += "\\textbf{Model} & \\textbf{AUC-PR} & \\textbf{AUC-ROC} & \\textbf{VUS-PR} & \\textbf{VUS-ROC} \\\\\n"
    latex_string += "\\hline\n"

    # Add data rows
    for _, row in df_table.iterrows():
        model = str(row['Model']).replace('_', '\\_')
        auc_pr = f"{row['AUC-PR']:.4f}"
        auc_roc = f"{row['AUC-ROC']:.4f}"
        vus_pr = f"{row['VUS-PR']:.4f}"
        vus_roc = f"{row['VUS-ROC']:.4f}"
        latex_string += f"{model} & {auc_pr} & {auc_roc} & {vus_pr} & {vus_roc} \\\\\n"

    # End LaTeX table
    latex_string += "\\hline\n"
    latex_string += "\\end{tabular}\n"
    latex_string += "\\end{table}\n"
    print("Successfully generated LaTeX string.")

    if output_path:
        try:
            with open(output_path, 'w') as f:
                f.write(latex_string)
            print(f"LaTeX table saved to {output_path}")
        except Exception as e:
            print(f"Error writing to output file: {e}")
    else:
        print("\n--- Generated LaTeX Table ---\n")
        print(latex_string)
        print("\n---------------------------\n")

if __name__ == '__main__':
    # The script is in TSAD_Seminar/Dataset_Creation/creation_code/Dataset_stats
    # The CSV is in TSAD_Seminar/Evaluation/Created_Benchmarking_Datasets/dataset_based
    # So we need to go up 3 directories and then down into the Evaluation folder.
    script_dir = os.path.dirname(__file__)
    csv_file = os.path.join(script_dir, '..', '..', '..', 'Evaluation', 'Created_Benchmarking_Datasets', 'dataset_based', '013_RAMmixed_13_Hardware_tr_2000_1st_2083.csv')
    
    # Normalize the path to resolve the '..' correctly
    csv_file = os.path.normpath(csv_file)

    # Define the output path for the .tex file (optional)
    output_file = os.path.join(script_dir, '013_RAMmixed_13_Hardware.tex')
    output_file = os.path.normpath(output_file)

    generate_latex_table(csv_file, output_file) 