#!/bin/bash
set -e

# Ensure we are in project root
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
cd "$SCRIPT_DIR"

if [ ! -d "venv" ]; then
    echo "Virtual environment not found. Please run ./setup_venv.sh first."
    exit 1
fi

source venv/bin/activate

# Check if kaggle is installed
if ! command -v kaggle &> /dev/null; then
    echo "kaggle command not found. Installing kaggle..."
    pip install kaggle
fi

# Check for ~/.kaggle/kaggle.json
if [ ! -f ~/.kaggle/kaggle.json ]; then
    echo "Error: ~/.kaggle/kaggle.json not found."
    echo "Please download your Kaggle API token from https://www.kaggle.com/me/account and place it in ~/.kaggle/kaggle.json"
    echo "Then run: chmod 600 ~/.kaggle/kaggle.json"
    exit 1
fi

echo "Downloading datasets to data/raw/..."
mkdir -p data/raw
cd data/raw

# 1. Prices of Moscow apartments (2024)
echo "Downloading Prices of Moscow apartments (2024)..."
kaggle datasets download -d ivan314sh/prices-of-moscow-apartments --unzip -o
# The file inside is likely 'moscow_flats_dataset_eng.csv' or similar, based on ls output 'moscow_flats_dataset_eng.csv' and 'moscow_flats_dataset.csv'
# It seems 'prices-of-moscow-apartments' dataset contains 'moscow_flats_dataset.csv' and 'moscow_flats_dataset_eng.csv'.
# We will use 'moscow_flats_dataset_eng.csv' if available, otherwise Russian one. 
# But wait, looking at file list: 'moscow_flats_dataset_eng.csv' is there.
# Let's rename it to what schemas expects: prices_moscow_2024.csv
if [ -f "moscow_flats_dataset_eng.csv" ]; then
    mv moscow_flats_dataset_eng.csv prices_moscow_2024.csv
elif [ -f "moscow_flats_dataset.csv" ]; then
    mv moscow_flats_dataset.csv prices_moscow_2024.csv
fi
# Clean up the other one if both exist
rm -f moscow_flats_dataset.csv moscow_flats_dataset_eng.csv 2>/dev/null || true


# 2. Moscow Apartment Listings (2020)
echo "Downloading Moscow Apartment Listings (2020)..."
kaggle datasets download -d alexeyleshchenko/moscow-apartment-listings --unzip -o
# Expected: moscow_apartment_listings.csv (based on ls output)
# Schema expects: moscow_apartment_listings_2020.csv
if [ -f "moscow_apartment_listings.csv" ]; then
    mv moscow_apartment_listings.csv moscow_apartment_listings_2020.csv
fi

# 3. Price of flats in Moscow (2018)
echo "Downloading Price of flats in Moscow (2018)..."
kaggle datasets download -d hugoncosta/price-of-flats-in-moscow --unzip -o
# Expected: flats_moscow.csv (based on ls output)
# Schema expects: price_flats_moscow_2018.csv
if [ -f "flats_moscow.csv" ]; then
    mv flats_moscow.csv price_flats_moscow_2018.csv
fi

# 4. Moscow Housing Price Dataset
echo "Downloading Moscow Housing Price Dataset..."
kaggle datasets download -d egorkainov/moscow-housing-price-dataset --unzip -o
# Expected: data.csv (based on ls output)
# Schema expects: moscow_housing_price.csv
if [ -f "data.csv" ]; then
    mv data.csv moscow_housing_price.csv
fi

echo "Download complete."
