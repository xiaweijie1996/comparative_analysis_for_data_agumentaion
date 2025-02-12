# Wind and Solar Data Generation

This repository contains models for generating and processing wind and solar data.

## Install Required Packages

Install the required Python packages using `requirements.txt`:
```bash
pip install -r requirements.txt
```

## Download the Data

Run the file `data_process/data_process.py`. The data will be saved in the `cleaned_data` folder.

```bash
python data_process/data_process.p
```

The folder `original_data_split` contains multiple `data_dict_x.pickle` files, where `x` represents the percentage of the training dataset included. For instance, `data_dict_0.1.pickle` contains 10% of the training data, while `data_dict_1.0.pickle` includes the entire training dataset.

## Experiment Explanation
*Data Augmentation*: To simulate different levels of data availability, we train generative models using `data_dict_x.pickle` files with varying values of `x`. The generated synthetic data is then leveraged for training higher-level models, such as those used for forecasting.Folder `data_augmentation` contains algorithms used for data augmentation.


*Prediction*: The augmented data is consequently used for training higher-level models. Folder `exp_pred` contains algorithms used for prediction.

 

## Original Dataset

The original dataset can be accessed from the following sources:

- [Hugging Face](https://huggingface.co/datasets/Weijie1996/wind_solar_dataset)
- [IEEE Dataport](https://ieee-dataport.org/competitions/hybrid-energy-forecasting-and-trading-competition#files)