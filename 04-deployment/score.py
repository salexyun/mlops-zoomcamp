#!/usr/bin/env python
# coding: utf-8

import sys

import numpy as np
import pandas as pd
import pickle

from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error


categorical = ['PULocationID', 'DOLocationID']


def read_data(filename):
    df = pd.read_parquet(filename)
    
    df['duration'] = df.tpep_dropoff_datetime - df.tpep_pickup_datetime
    df['duration'] = df.duration.dt.total_seconds() / 60

    df = df[(df.duration >= 1) & (df.duration <= 60)].copy()

    df[categorical] = df[categorical].fillna(-1).astype('int').astype('str')
    
    return df


def apply_model(input_file, dv, model):
    df = read_data(input_file)
    dicts = df[categorical].to_dict(orient='records')
    X_val = dv.transform(dicts)
    y_pred = model.predict(X_val)

    print(f"Mean predicted duration: {np.mean(y_pred)}")
    print(f"Standard deviation: {np.std(y_pred)}")

    return df, y_pred


def save_results(df, y_pred, year, month, output_file):
    df['ride_id'] = f'{year:04d}/{month:02d}_' + df.index.astype('str')
    df_result = pd.DataFrame()
    df_result['ride_id'] = df['ride_id']
    df_result['predicted'] = y_pred

    df_result.to_parquet(
        output_file,
        engine='pyarrow',
        compression=None,
        index=False
    )


def run():
    taxi_type = sys.argv[1] # yellow
    year = int(sys.argv[2]) # 2023
    month = int(sys.argv[3]) # 3

    input_file = f"https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_{year:04d}-{month:02d}.parquet"
    output_file = f"{taxi_type}_{year:04d}-{month:02d}.parquet"

    print("Loading the model")
    with open('model.bin', 'rb') as f_in:
        dv, model = pickle.load(f_in)

    print("Applying the model") 
    df, y_pred = apply_model(input_file, dv, model)

    print(f"Saving results to {output_file}")
    save_results(df, y_pred, year, month, output_file)



if __name__ == "__main__":
    run()