"""
Configuration for dataset column mappings.
Each key in DATASET_MAPPINGS represents a dataset identifier.
The value is a dictionary mapping the RAW column name in the CSV to the UNIFIED column name.

Target Schema (Extended with location features):
- price (float): Listing price
- total_area (float): Total apartment area in sq meters
- rooms (int): Number of rooms
- floor (int): Floor number
- total_floors (int): Total floors in building
- subway_dist (float): Distance to nearest subway in meters
- year (int): Year of the dataset (extracted or injected)

Note: For subway_dist, if time is provided (minutes), convert using 5.5 km/h walking speed (91.67 m/min)
"""

DATASET_MAPPINGS = {
    "moscow_apartments_2024": {
        # data/raw/prices_moscow_2024.csv
        "price": "price",
        "total_area": "total_area",
        "floor": "floor",
        "number_of_floors": "total_floors",
        "number_of_rooms": "rooms",
        "min_to_metro": "min_to_metro",  # Will convert to subway_dist (time->distance)
    },
    "moscow_listings_2020": {
        # data/raw/moscow_apartment_listings_2020.csv
        "price": "price",
        "footage": "total_area",
        "rooms": "rooms",
        "floor": "floor",
        "max_floor": "total_floors",
        "dist_to_subway": "subway_dist",  # Already in meters
    },
    "moscow_flats_2018": {
        # data/raw/price_flats_moscow_2018.csv
        "price": "price",
        "totsp": "total_area",
        "floor": "floor",
        "metrdist": "min_to_metro",  # В минутах, будет конвертировано в метры
    },
    "moscow_housing_price_dataset": {
        # data/raw/moscow_housing_price.csv
        "Price": "price",
        "Area": "total_area",
        "Number of rooms": "rooms",
        "Floor": "floor",
        "Number of floors": "total_floors",
        "Minutes to metro": "min_to_metro",  # Will convert to subway_dist
    }
}

# Define filenames expected in data/raw
DATASET_FILES = {
    "moscow_apartments_2024": "prices_moscow_2024.csv",
    "moscow_listings_2020": "moscow_apartment_listings_2020.csv",
    "moscow_flats_2018": "price_flats_moscow_2018.csv",
    "moscow_housing_price_dataset": "moscow_housing_price.csv"
}

# Define fixed year for datasets that don't have a date column
DATASET_YEARS = {
    "moscow_apartments_2024": 2024,
    "moscow_listings_2020": 2020,
    "moscow_flats_2018": 2018,
    "moscow_housing_price_dataset": 2023 
}
