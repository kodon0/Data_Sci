#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul  1 15:03:38 2020

@author: kieranodonnell
"""

# This script returns the monthly real time electricyt prices for a given US State (Texas in this case)

import eia
import pandas as pd
import numpy as np

def retrieve_time_series(api, series_ID):
    """
    Return the time series dataframe, based on API and unique Series ID
    api: API that we're connected to
    series_ID: string. Name of the series that we want to pull from the EIA API
    """
    #Retrieve Data By Series ID 
    series_search = api.data_by_series(series=series_ID)
    ##Create a pandas dataframe from the retrieved time series
    df = pd.DataFrame(series_search)
    return df
    

def main():
    """
    Run main script
    """
    #Create EIA API using  specific API key
    api_key = "716c2a14953da546decd4f96c16aba8d"
    api = eia.API(api_key)
    
    #Pull the electricity price data
    series_ID='ELEC.PRICE.TX-ALL.M'
    electricity_df=retrieve_time_series(api, series_ID)
    electricity_df.reset_index(level=0, inplace=True)
    #Rename the columns for easer analysis
    electricity_df.rename(columns={'index':'Date',
            electricity_df.columns[1]:'Electricity_Price'}, 
            inplace=True)
    print(electricity_df)

if __name__== "__main__":
    main()
