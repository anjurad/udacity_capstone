# import libraries
import os
import pandas as pd
import numpy as np

# read in data
df = pd.read_csv("./data/aug_train.csv")

# make dataset smaller
df_small = df[:1000]

# replace NaN with Null for .csv
df_small = df_small.where(pd.notnull(df_small), "")

# vars
download_folder = "data_blob_upload"
download_file = "job_leaver_aug_small.csv"

# create
os.makedirs(download_folder, exist_ok=True)
download_location = os.path.join(download_folder, download_file)

# export the data to root
df_small.to_csv(download_location, index=False)
