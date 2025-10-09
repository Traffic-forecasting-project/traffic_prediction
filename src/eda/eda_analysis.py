''' 
__author__ = "Georges Nassopoulos"
__copyright__ = None
__version__ = "1.0.2"
__email__ = "georges.nassopoulos@gmail.com"
__status__ = "Dev"
__desc__ = "Comprehensive EDA script with logging for both raw and processed traffic data."
'''

import os
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from io import StringIO

## Imports for "microservice" et "legacy" structures respectively
try:
    from Services.FastAPI.src.logging_utils import get_logger, log_execution_time_and_path
    from Services.utils.utils import save_summary_stats
    from Services.FastAPI.src.constants import (
        TARGETS,
        LIVE_DATA_DIR,
        EDA_OUTPUT_DIR,
        PROCESSED_DATA_DIR 
    )
except:
    from src.core.logging_utils import get_logger, log_execution_time_and_path
    from src.utils.utils import save_summary_stats
    from src.core.constants import (
        TARGETS,
        LIVE_DATA_DIR,
        EDA_OUTPUT_DIR,
        PROCESSED_DATA_DIR 
    )

## Initialize logger
logger = get_logger(__name__)

def plot_histograms(df: pd.DataFrame, label: str, output_dir: str) -> None:
    """
        Generate and save histograms for all numerical columns in a DataFrame

        Args:
            df (pd.DataFrame): Input data
            label (str): Dataset label used in filenames
            output_dir (str): Directory where plots will be saved
    """
    
    ## Select numeric columns only
    num_cols = df.select_dtypes(include=np.number).columns

    ## Generate and save a histogram for each numeric column
    for col in num_cols:
        plt.figure(figsize=(6, 4))
        sns.histplot(df[col], bins=30, kde=True)
        plt.title(f"{label} - {col}")
        plt.xlabel(col)
        plt.ylabel("Frequency")
        plt.tight_layout()

        ## Save figure
        plot_path = os.path.join(output_dir, f"{label}__{col}_hist.png")
        plt.savefig(plot_path)
        plt.close()

def plot_correlation_matrix(df: pd.DataFrame, label: str, output_dir: str) -> None:
    """
        Compute and save the correlation matrix heatmap for numerical columns

        Args:
            df (pd.DataFrame): Input data
            label (str): Dataset label for the title
            output_dir (str): Directory to save the plot
    """
    
    ## Extract numeric columns only
    num_cols = df.select_dtypes(include=np.number)

    ## Only compute correlation if more than one numeric column
    if num_cols.shape[1] >= 2:
        corr = num_cols.corr()

        ## Replace NaN with 0 to avoid seaborn MaskedConstant warning
        corr_safe = corr.fillna(0)

        plt.figure(figsize=(10, 8))
        sns.heatmap(corr_safe, annot=True, fmt=".2f", cmap="coolwarm")
        
        plt.title(f"Correlation Matrix - {label}")
        plt.tight_layout()

        ## Save heatmap
        plt.savefig(os.path.join(output_dir, f"{label}__correlation.png"))
        plt.close()

def load_and_analyze_group(label: str, files: list[str], summary_path: str) -> None:
    """
        Load multiple CSV files, concatenate them, and run EDA:
        summary stats, histograms, and correlation matrix

        Args:
            label (str): Name of the dataset group
            files (list[str]): List of CSV file paths to process
            summary_path (str): Path to the summary CSV file

        Returns:
            None
    """
    
    try:
        ## Concatenate all CSVs into a single DataFrame
        df = pd.concat([pd.read_csv(f, low_memory=False, on_bad_lines='skip') for f in files], ignore_index=True)

        ## Generate and save statistics and visualizations
        save_summary_stats(df, label, summary_path)
        plot_histograms(df, label, EDA_OUTPUT_DIR)
        plot_correlation_matrix(df, label, EDA_OUTPUT_DIR)

        logger.info(f"\t[EDA] Analysis completed for {label} ({len(df)} rows).")
    except Exception as e:
        logger.error(f"\t[EDA] Failed to process {label}: {e}")

def validate_dataset(df: pd.DataFrame) -> None:
    """
        Validate a dataset by reporting potential data quality issues without modifying the DataFrame

        Args:
            df (pd.DataFrame): The DataFrame to validate

        Returns:
            None. Prints validation report to console
    """
    
    logger.info("\n\t Dataset Validation Report\n" + "="*30)

    ## Check for columns with all null values
    null_cols = df.columns[df.isnull().all()]
    if len(null_cols):
        logger.info(f"\n\t Columns with all null values ({len(null_cols)}): {list(null_cols)}")

    ## Check for columns with 100% identical values
    constant_cols = [col for col in df.columns if df[col].nunique(dropna=False) == 1]
    if constant_cols:
        logger.info(f"\n\t Columns with only one unique value ({len(constant_cols)}): {constant_cols}")

    ## Check for columns with high null proportion
    high_null_cols = df.columns[df.isnull().mean() > 0.8]
    if len(high_null_cols):
        logger.info(f"\n\t  Columns with more than 80% missing values ({len(high_null_cols)}): {list(high_null_cols)}")

    ## Check for numeric columns with very low variance
    low_variance_cols = [col for col in df.select_dtypes(include='number') if df[col].std() < 1e-3]
    if low_variance_cols:
        logger.info(f"\n\t Numeric columns with near-zero variance: {low_variance_cols}")

    logger.info("\n\t Validation completed.\n")

def generate_eda_report(csv_path: str, target: str = "mean_delay"):
    """
        Generate an Exploratory Data Analysis (EDA) report for a given dataset
        This includes text-based summaries and visualizations

        Args:
            csv_path (str): Path to the CSV file containing the dataset
            target (str): Name of the target variable (default is "mean_delay")

        Output:
            Saves a text summary and plots in a dedicated folder
    """
    
    ## Load dataset
    df = pd.read_csv(csv_path, low_memory=False)
    base_name = os.path.splitext(os.path.basename(csv_path))[0]
    output_folder = f"{EDA_OUTPUT_DIR}/eda_outputs_{base_name}"
    os.makedirs(output_folder, exist_ok=True)

    ## ====== 1. General Information ======
    with open(os.path.join(output_folder, "eda_text_summary.txt"), "w", encoding="utf-8") as f:
        f.write(f"EDA Report for file: {csv_path}\n")
        f.write("="*60 + "\n\n")

        ## Dataset structure
        f.write("Dataset Info:\n")
        buffer = StringIO()
        df.info(buf=buffer, verbose=True, memory_usage=True)
        f.write(buffer.getvalue() + "\n\n")

        ## Missing values per column
        f.write("Missing Values (%):\n")
        f.write((df.isnull().mean() * 100).sort_values(ascending=False).to_string() + "\n\n")

        ## Basic statistics for numeric columns
        f.write("Descriptive Statistics:\n")
        f.write(df.describe().T.to_string() + "\n\n")

        ## ====== 2. Outlier Detection ======
        f.write("Outliers (above 95th percentile):\n")
        high_values = df[target][df[target] > df[target].quantile(0.95)]
        f.write(f"Count: {len(high_values)} (over {len(df)} rows)\n")
        f.write(f"Max: {high_values.max():.2f}, 95th percentile: {df[target].quantile(0.95):.2f}\n\n")

        ## ====== 3. Correlation Matrix ======
        num_cols = df.select_dtypes(include=[np.number])
        corr_matrix = num_cols.corr()

        f.write("Top 10 features correlated with target:\n")
        top_corr = corr_matrix[target].abs().sort_values(ascending=False).drop(target).head(10)
        f.write(top_corr.to_string() + "\n\n")

        f.write("Full Correlation Matrix (numeric only):\n")
        f.write(corr_matrix.to_string() + "\n")

    ## Run validation logic if any
    validate_dataset(df)

    ## ====== 4. Visual Correlation Heatmap ======
    plt.figure(figsize=(12, 10))
    # sns.heatmap(corr_matrix[top_corr.index.tolist() + [target]], annot=True, fmt=".2f", cmap="coolwarm")
    corr_data = corr_matrix[top_corr.index.tolist() + [target]].fillna(0)
    sns.heatmap(corr_data, annot=True, fmt=".2f", cmap="coolwarm")        
    # heatmap_data = corr_matrix[top_corr.index.tolist() + [target]].copy()
    # heatmap_data = heatmap_data.fillna(0)  # ou: .replace([np.inf, -np.inf], 0)
    # sns.heatmap(heatmap_data, annot=True, fmt=".2f", cmap="coolwarm")

    plt.title(f"Correlation Heatmap with {target}")
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, "heatmap_correlation.png"))
    plt.close()

    ## ====== 5. Distribution Plot of Target Variable ======
    plt.figure(figsize=(8, 5))
    sns.histplot(df[target], bins=40, kde=True)
    plt.title(f"Distribution of Target: {target}")
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, f"distribution_{target}.png"))
    plt.close()

    logger.info(f"\t [EDA] Report generated for {target} in folder: {output_folder}")

@log_execution_time_and_path
def run_eda_pipeline(
    strategy: str = "incident_analysis",
    live_data_dir: str = os.path.join("data", "live"),
    processed_data_dir: str = os.path.join("data", "processed"),
    eda_output_dir: str = os.path.join("eda")
) -> None:
    """
        Run EDA analysis using the specified strategy and directories.

        Args:
            strategy (str): 'traffic_analysis' or 'incident_analysis'
            live_data_dir (str): Path to directory containing live CSV files
            processed_data_dir (str): Path to processed feature-engineered data
            eda_output_dir (str): Path where EDA results will be saved
    """
    try:

        ## Path to store all global stats
        summary_path = os.path.join(eda_output_dir, "eda_summary.csv")

        ## Perform EDA on raw live data grouped by strategy
        for strategy in ["incident_analysis", "traffic_analysis"]:
            pattern = os.path.join(live_data_dir, f"live_data_{strategy}*")
            files = glob.glob(pattern)

            if files:
                load_and_analyze_group(f"live_{strategy}", files, summary_path)
            else:
                logger.warning(f"[EDA] No files found for strategy: {strategy}")

        ## Perform EDA on all feature-engineered exports
        export_files = glob.glob(os.path.join(processed_data_dir, "*.csv"))
        for f in export_files:
            label = os.path.splitext(os.path.basename(f))[0]
            load_and_analyze_group(f"exports_{label}", [f], summary_path)

        ## Final summary logs
        logger.info(f"\t All visualizations saved to: {eda_output_dir}/")
        logger.info(f"\t Summary statistics CSV: {summary_path}")

        ## Loop through each target and generate EDA report
        for target in TARGETS:
            tmp_var = processed_data_dir.replace("/DataCollection/data/DataPreparation","/DataPreparation/data")
            csv_file = f"{tmp_var}/df_features_incident_analysis_{target}.csv"
            if os.path.exists(csv_file):
                logger.info(f"\t Generating EDA report for {target}...")
                generate_eda_report(csv_file, target)
            else:
                logger.info(f"\t File not found: {csv_file}")

    except Exception as e:
        logger.warning(f"Error while EDA analysis for strategy '{strategy}': {e}")
       
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run EDA analysis pipeline.")
    parser.add_argument("-s", "--strategy", type=str, default="incident_analysis")
    parser.add_argument("-l", "--live-dir", type=str, default=os.path.join("data", "live"))
    parser.add_argument("-p", "--processed-dir", type=str, default=os.path.join("data", "processed"))
    parser.add_argument("-e", "--eda-dir", type=str, default=os.path.join("eda"))

    args = parser.parse_args()

    run_eda_pipeline(
        strategy=args.strategy,
        live_data_dir=args.live_dir,
        processed_data_dir=args.processed_dir,
        eda_output_dir=args.eda_dir
    )