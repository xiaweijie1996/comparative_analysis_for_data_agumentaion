# Wind and Solar Data Generation

This repository contains models for generating and processing wind and solar data.

## Install Required Packages

Install the required Python packages using `requirements.txt`:
```bash
pip install -r requirements.txt
```

## Load the Data

Run the file `data_process/data_process.py`. The data will be saved in the `cleaned_data` folder.

```bash
python data_process/data_process.p
```

## Examples
[svr_predictor.py](https://github.com/xiaweijie1996/comparative_analysis_for_data_agumentaion/blob/main/exp_pred/svr/svr_predictor.py) contains a example of how to use the current pipline.


## Original Dataset

The original dataset can be accessed from the following sources:

- [Hugging Face](https://huggingface.co/datasets/Weijie1996/wind_solar_dataset)
- [IEEE Dataport](https://ieee-dataport.org/competitions/hybrid-energy-forecasting-and-trading-competition#files)