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
    
    # Abstrat the wind related data
    wind_data = data[["valid_datetime", "WindSpeed", "Wind_MWh_credit"]]

    # Transform the datetime
    wind_data["valid_datetime"] = pd.to_datetime(wind_data["valid_datetime"])
    wind_data["valid_datetime"] = wind_data["valid_datetime"].dt.dayofyear
    
    # Aggregate the data by ref_datetime
    wind_data = wind_data.groupby("ref_datetime").agg({"valid_datetime":'mean', "WindSpeed":list, "Wind_MWh_credit":list}).reset_index()
    
    # Rename the columns
    windspeed_dataframe = wind_data['WindSpeed'].apply(pd.Series).add_prefix('WindSpeed_')
    day_dataframe = wind_data['valid_datetime']
    wind_credit_dataframe = wind_data['Wind_MWh_credit'].apply(pd.Series).add_prefix('Wind_MWh_credit_')
    
    # Concatenate the dataframes
    full_input_data = pd.concat([day_dataframe, windspeed_dataframe], axis=1)
    full_output_data = wind_credit_dataframe
    
    return full_input_data, full_output_data

def split_train_test_val(full_input_data, full_output_data,
                         split_ratios = config["Data"]["train_val_test_split"]):
    # Split the data into train, validation and test sets based on the split ratios (default 70-15-15)
    train_ratio, val_ratio, test_ratio = split_ratios
    train_size = int(len(full_input_data) * train_ratio)
    val_size = int(len(full_input_data) * val_ratio)
    test_size = len(full_input_data) - train_size - val_size
    
    # Split the full input data
    train_input = full_input_data[:train_size]
    val_input = full_input_data[train_size:train_size + val_size]
    test_input = full_input_data[-test_size:]
    
    # Split the full output data
    train_output = full_output_data[:train_size]
    val_output = full_output_data[train_size:train_size + val_size]
    test_output = full_output_data[-test_size:]
    
    # Define dictionary to store the data
    data_dict = {
        "train_input": train_input,
        "val_input": val_input,
        "test_input": test_input,
        "train_output": train_output,
        "val_output": val_output,
        "test_output": test_output
    }
    # print('Train input shape:', train_input.shape)
    # print('Train output shape:', train_output.shape)
    # print('Validation input shape:', val_input.shape)
    # print('Validation output shape:', val_output.shape)
    # print('Test input shape:', test_input.shape)
    # print('Test output shape:', test_output.shape)
    
    return data_dict

def train_data_reserve(data_dict, reserve_ratio):
    # Only use reserve_ratio of the training data for training
    # Split the data into train and reserve sets based on the reserve ratio
    reserve_size = int(len(data_dict["train_input"]) * reserve_ratio)
    
    # Split the train data
    reserve_input = data_dict["train_input"][:reserve_size]
    reserve_output = data_dict["train_output"][:reserve_size]
    
    # Update the train data
    data_dict["train_input"] = data_dict["train_input"][reserve_size:]
    data_dict["train_output"] = data_dict["train_output"][reserve_size:]
    
    # Define dictionary to store the data
    data_dict["reserve_input"] = reserve_input
    data_dict["reserve_output"] = reserve_output
    
    # print('Train input shape:', data_dict["train_input"].shape)
    # print('Train output shape:', data_dict["train_output"].shape)
    
    return data_dict

class Dataloader():
    def __init__(self, data_dict, data_type, 
                 batch_size = config["Data"]["batch_size"]):
        self.data_dict = data_dict
        self.batch_size = batch_size
        self.enm_index = 0
        self.data_type = data_types
    
    def get_batch(self):
        # Allow emueration over the data, until the end of the data
        input_data = self.data_dict[self.data_type + "_input"]
        output_data = self.data_dict[self.data_type + "_output"]
        
        if self.enm_index + self.batch_size > len(input_data):
            self.enm_index = 0
        batch_input = input_data[self.enm_index:self.enm_index + self.batch_size]
        batch_output = output_data[self.enm_index:self.enm_index + self.batch_size]
        self.enm_index += self.batch_size
        
        return batch_input, batch_output
    

if __name__ == "__main__":
    # Catch the wind data
    full_input_data, full_output_data = catch_the_wind()
    print('--------------------------------------')
    
    # Split the data into train, validation and test sets
    data_dict = split_train_test_val(full_input_data, full_output_data)
    print('--------------------------------------')
    
    # Split the training data into training and reserve sets
    data_dict_reserve = train_data_reserve(data_dict, reserve_ratio=0.5)
    
    # Define the data types
    data_types = 'train'
    
    # Create a dataloader object
    dataloader = Dataloader(data_dict, data_types)
    
    # iterate over the data
    for i in range(100):
        batch_input, batch_output = dataloader.get_batch()
        print('Batch input shape:', batch_input.shape)
        print('Batch output shape:', batch_output.shape)
        print(i)
        print('--------------------------------------')

    
    