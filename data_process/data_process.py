from huggingface_hub import snapshot_download
import pandas as pd
import numpy as np
import xarray as xr

# Define a function to load the dataset from the huggingface hub
def load_dataset(
    local_dir = "data_process",
    repo_id = 'Weijie1996/wind_solar_dataset',  
):
    # check if the dataset is already downloaded 
    try:
        snapshot_download(repo_id=repo_id,local_dir=local_dir,repo_type="dataset")
    except:
        print("Dataset is already downloaded")

def process_data():
    # Load the dataset 20200920_20231027
    dwd_Hornsea1 = xr.open_dataset("data_process/fulldataset/dwd_icon_eu_20200920_20231027/dwd_icon_eu_hornsea_1_20200920_20231027.nc")
    dwd_Hornsea1_features = dwd_Hornsea1["WindSpeed:100"].mean(dim=["latitude","longitude"]).to_dataframe().reset_index()
    dwd_Hornsea1_features["ref_datetime"] = dwd_Hornsea1_features["ref_datetime"].dt.tz_localize("UTC")
    dwd_Hornsea1_features["valid_datetime"] = dwd_Hornsea1_features["ref_datetime"] + pd.TimedeltaIndex(dwd_Hornsea1_features["valid_datetime"],unit="hours")

    dwd_solar = xr.open_dataset("data_process/fulldataset/dwd_icon_eu_20200920_20231027/dwd_icon_eu_pes10_20200920_20231027.nc")
    dwd_solar_features = dwd_solar["SolarDownwardRadiation"].mean(dim="point").to_dataframe().reset_index()
    dwd_solar_features["ref_datetime"] = dwd_solar_features["ref_datetime"].dt.tz_localize("UTC")
    dwd_solar_features["valid_datetime"] = dwd_solar_features["ref_datetime"] + pd.TimedeltaIndex(dwd_solar_features["valid_datetime"],unit="hours")

    energy_data = pd.read_csv("data_process/fulldataset/Energy_Data_20200920_20231027.csv")
    energy_data["dtm"] = pd.to_datetime(energy_data["dtm"])
    energy_data["Wind_MWh_credit"] = 0.5*energy_data["Wind_MW"] - energy_data["boa_MWh"]
    energy_data["Solar_MWh_credit"] = 0.5*energy_data["Solar_MW"]

    modelling_table = dwd_Hornsea1_features.merge(dwd_solar_features,how="outer",on=["ref_datetime","valid_datetime"])
    modelling_table = modelling_table.set_index("valid_datetime").groupby("ref_datetime").resample("30T").interpolate("linear")
    modelling_table = modelling_table.drop(columns="ref_datetime",axis=1).reset_index()
    modelling_table = modelling_table.merge(energy_data,how="inner",left_on="valid_datetime",right_on="dtm")
    modelling_table = modelling_table[modelling_table["valid_datetime"] - modelling_table["ref_datetime"] < np.timedelta64(50,"h")]
    modelling_table.rename(columns={"WindSpeed:100":"WindSpeed"},inplace=True)
    
    # Columns to be used for modelling
    columns = ['ref_datetime','valid_datetime','WindSpeed','SolarDownwardRadiation','Wind_MWh_credit','Solar_MWh_credit']
    modelling_table = modelling_table[columns]
    
    # Save energy data to clean_data folder as csv
    modelling_table.to_csv("cleaned_data/Data_20200920_20231027.csv",index=False)
    

if __name__ == "__main__":
    print("1)\U0001F600 Data processing started ")
    print("2)\U0001F605 Downloading the dataset from huggingface hub")
    load_dataset()
    print("3)\U0001F923 Processing the data ")
    process_data()
    print("4)\U0001F911 Data processing completed, the data is saved in cleaned_data folder ")