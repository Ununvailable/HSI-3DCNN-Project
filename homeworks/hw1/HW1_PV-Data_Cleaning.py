# -*- coding: utf-8 -*-
import pandas as pd

original_file_name = "PV_farmt_20220101_20230420_original.csv"
cleaned_file_name = "PV_farmt_20220101_20230420_cleaned.csv"

file_dir = r"hsi_datasets/hw1/"

pv_dataframe = pd.read_csv(file_dir + original_file_name, sep = ";")
print(pv_dataframe)

pv_dataframe.fillna(0)

pv_dataframe.to_csv(file_dir + cleaned_file_name)