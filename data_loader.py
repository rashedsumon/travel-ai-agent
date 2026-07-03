import os
import glob
import kagglehub
import pandas as pd

def load_all_datasets():
    """
    Downloads datasets seamlessly via kagglehub and loads them into memory.
    """
    print("Downloading datasets from Kaggle...")
    
    # 1. Worldwide Travel Cities (Ratings & Climate)
    path_cities = kagglehub.dataset_download("furkanima/worldwide-travel-cities-ratings-and-climate")
    # Dynamically find the CSV file in the downloaded path
    cities_file = glob.glob(os.path.join(path_cities, "*.csv"))[0]
    df_cities = pd.read_csv(cities_file)
    
    # 2. Travel Planner Dataset
    path_planner = kagglehub.dataset_download("arkakarmoker/travel-planner-dataset")
    # Looks for any CSV in the destination directory
    planner_file = glob.glob(os.path.join(path_planner, "*.csv"))[0]
    df_planner = pd.read_csv(planner_file)
    
    # 3. Tourism Recommendation Dataset
    path_rec = kagglehub.dataset_download("lucasbrownkk/tourism-recommendation-dataset")
    rec_file = glob.glob(os.path.join(path_rec, "*.csv"))[0]
    df_rec = pd.read_csv(rec_file)
    
    # 4. Travel Agency Data
    path_agency = kagglehub.dataset_download("nhatnguyentran/travel-agency-data")
    agency_file = glob.glob(os.path.join(path_agency, "*.csv"))[0]
    df_agency = pd.read_csv(agency_file)
    
    return df_cities, df_planner, df_rec, df_agency

if __name__ == "__main__":
    # Test execution
    c, p, r, a = load_all_datasets()
    print("All datasets loaded successfully!")
    print(f"Cities shape: {c.shape}, Planner shape: {p.shape}")