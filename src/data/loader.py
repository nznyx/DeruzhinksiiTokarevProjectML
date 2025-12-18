import os
import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Optional
from src.data.schemas import DATASET_MAPPINGS, DATASET_FILES, DATASET_YEARS

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class MoscowHousingLoader:
    def __init__(self, data_dir: str = "data/raw"):
        """
        Initialize the loader with the directory containing raw CSV files.
        """
        self.data_dir = data_dir
        # Schema with location features
        self.unified_schema = [
            'price', 'total_area', 'rooms',
            'floor', 'total_floors', 'subway_dist', 'year'
        ]
        # Walking speed: 5.5 km/h (average walking pace) = 91.67 m/min
        self.walking_speed_m_per_min = 5500 / 60  # 91.67 m/min

    def _get_file_path(self, dataset_key: str) -> str:
        filename = DATASET_FILES.get(dataset_key)
        if not filename:
            raise ValueError(f"Unknown dataset key: {dataset_key}")
        return os.path.join(self.data_dir, filename)

    def _standardize_df(self, df: pd.DataFrame, dataset_key: str) -> pd.DataFrame:
        """
        Renames columns to the unified schema and adds the year column.
        Converts min_to_metro to subway_dist if needed.
        """
        mapping = DATASET_MAPPINGS.get(dataset_key, {})
        
        # Rename columns
        df = df.rename(columns=mapping)
        
        # Add year from metadata if available and not present in dataframe
        if 'year' not in df.columns:
            year = DATASET_YEARS.get(dataset_key)
            if year:
                df['year'] = year
        
        # Convert min_to_metro to subway_dist (meters)
        if 'min_to_metro' in df.columns and 'subway_dist' not in df.columns:
            # Фильтрация нереалистичных значений (>60 минут считаем ошибкой)
            min_to_metro_clean = pd.to_numeric(df['min_to_metro'], errors='coerce')
            min_to_metro_clean = min_to_metro_clean.where(min_to_metro_clean <= 60, np.nan)
            # Convert minutes to meters (walking speed: 5.5 km/h = 91.67 m/min)
            df['subway_dist'] = min_to_metro_clean * self.walking_speed_m_per_min
        
        # Ensure all target columns exist (fill with NaN if missing)
        for col in self.unified_schema:
            if col not in df.columns:
                df[col] = np.nan
                
        # Select only unified columns
        df = df[self.unified_schema]
        
        return df

    def _load_generic(self, dataset_key: str, sep: str = ',') -> Optional[pd.DataFrame]:
        """
        Generic loader for datasets that just need column mapping.
        """
        file_path = self._get_file_path(dataset_key)
        if not os.path.exists(file_path):
            logger.warning(f"File not found: {file_path}. Skipping {dataset_key}.")
            return None
        
        try:
            df = pd.read_csv(file_path, sep=sep)
            logger.info(f"Loaded {dataset_key}: {df.shape[0]} rows")
            return self._standardize_df(df, dataset_key)
        except Exception as e:
            logger.error(f"Error loading {dataset_key}: {e}")
            return None

    def load_all(self) -> pd.DataFrame:
        """
        Loads and merges all available datasets.
        """
        dfs = []
        
        # 1. Prices of Moscow apartments (2024)
        df = self._load_generic("moscow_apartments_2024")
        if df is not None: dfs.append(df)
        
        # 2. Moscow Apartment Listings (2020)
        df = self._load_generic("moscow_listings_2020")
        if df is not None: dfs.append(df)
        
        # 3. Price of flats in Moscow (2018)
        df = self._load_generic("moscow_flats_2018")
        if df is not None: dfs.append(df)
        
        # 4. Moscow Housing Price Dataset
        df = self._load_generic("moscow_housing_price_dataset")
        if df is not None: dfs.append(df)

        if not dfs:
            logger.warning("No datasets loaded.")
            return pd.DataFrame(columns=self.unified_schema)
        
        full_df = pd.concat(dfs, ignore_index=True)
        
        # Basic type conversion
        # 'price' to float
        full_df['price'] = pd.to_numeric(full_df['price'], errors='coerce')
        
        # 'year' to int
        full_df['year'] = pd.to_numeric(full_df['year'], errors='coerce')

        logger.info(f"Total merged data: {full_df.shape[0]} rows")
        return full_df

if __name__ == "__main__":
    loader = MoscowHousingLoader()
    df = loader.load_all()
    print(df.info())
    print(df.head())
