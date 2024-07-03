#!/usr/bin/env python
# coding: utf-8

import sys
import pickle
import pandas as pd

from datetime import datetime


def read_data(data, columns):
    df = pd.DataFrame(data, columns=columns)
    
    df['duration'] = df.tpep_dropoff_datetime - df.tpep_pickup_datetime
    df['duration'] = df.duration.dt.total_seconds() / 60

    df = df[(df.duration >= 1) & (df.duration <= 60)].copy()

    df[columns] = df[columns].fillna(-1).astype('int').astype('str')
    
    return df


def prepare_data(df, columns):
    """Performs transformation."""
    df['duration'] = df.tpep_dropoff_datetime - df.tpep_pickup_datetime
    df['duration'] = df.duration.dt.total_seconds() / 60

    df = df[(df.duration >= 1) & (df.duration <= 60)].copy()

    df[columns] = df[columns].fillna(-1).astype('int').astype('str')

    return df


def dt(hour, minute, second=0):
    """Helper function."""
    return datetime(2023, 1, 1, hour, minute, second)


def test_preprocessing_logic():
    data = [
        (None, None, dt(1, 1), dt(1, 10)),
        (1, 1, dt(1, 2), dt(1, 10)),
        (1, None, dt(1, 2, 0), dt(1, 2, 59)),
        (3, 4, dt(1, 2, 0), dt(2, 2, 1)),      
    ]
    columns = ['PULocationID', 'DOLocationID', 'tpep_pickup_datetime', 'tpep_dropoff_datetime']

    actual_df = read_data(data, columns)

    df = pd.DataFrame(data, columns=columns)
    expected_df = prepare_data(df, columns)
    print(expected_df.info())

    assert actual_df.equals(expected_df)