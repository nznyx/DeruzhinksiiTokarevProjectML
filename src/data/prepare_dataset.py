import os
import sys
import logging

# Ensure src module is importable if running directly from this file's directory or root
# This allows 'from src.data.loader ...' to work if we run 'python src/data/prepare_dataset.py' from root
# provided we add root to sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(os.path.dirname(current_dir)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# If running from root as 'python src/data/prepare_dataset.py', os.getcwd() is root.
# If running from src/data, os.getcwd() is src/data.
# We need to ensure we can import 'src'.
# A robust way is adding the project root to sys.path.
sys.path.append(os.getcwd())

from src.data.loader import MoscowHousingLoader

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def main():
    logger.info("Starting Moscow Housing Data Loader...")
    
    # Determine data directories relative to project root
    # We assume the script is run from project root, or we find root relative to script
    # Let's find project root based on this file's location: src/data/prepare_dataset.py -> ../../..
    script_path = os.path.abspath(__file__)
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(script_path)))
    
    data_raw_dir = os.path.join(project_root, "data", "raw")
    data_processed_dir = os.path.join(project_root, "data", "processed")
    
    logger.info(f"Project root detected at: {project_root}")
    logger.info(f"Reading data from: {data_raw_dir}")
    
    # Initialize loader
    loader = MoscowHousingLoader(data_dir=data_raw_dir)
    
    # Load and merge data
    logger.info("Loading datasets...")
    merged_df = loader.load_all()
    
    # Basic statistics
    logger.info("Dataset Statistics:")
    logger.info(f"Shape: {merged_df.shape}")
    logger.info(f"Columns: {merged_df.columns.tolist()}")
    
    # Check for missing values
    missing_summary = merged_df.isnull().sum()
    logger.info("Missing values per column:")
    for col, val in missing_summary.items():
        logger.info(f"  {col}: {val}")
        
    # Save processed data
    output_path = os.path.join(data_processed_dir, "moscow_housing_merged.csv")
    logger.info(f"Saving merged dataset to {output_path}...")
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    merged_df.to_csv(output_path, index=False)
    
    logger.info("Done!")

if __name__ == "__main__":
    main()

