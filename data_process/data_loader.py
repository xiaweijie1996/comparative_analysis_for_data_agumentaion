import yaml
import pandas as pd
import numpy as np

# Read the configuration file
with open("data_process/config.yaml", "r") as file:
    config = yaml.safe_load(file)

# Define a function to load the cleaned dataset
def catch_the_wind(path=config["Data"]["path"]):
    # Read the cleaned dataset
    data = pd.read_csv(path, index_col=0)
    print(wind_data.head())
    
    # Abstrat the wind related data
    wind_data = data[["ref_datetime", "valid_datetime", "WindSpeed", "Wind_MWh_credit"]]
    print(wind_data.head())
    
    pass

if __name__ == "__main__":
    catch_the_wind()
    