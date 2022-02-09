#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thursday August  20 11:28:59 2020

@author: kieranodonnell
"""
import pandas as pd
import numpy as np
import time
from datetime import datetime, timedelta
from pandas.tseries.offsets import MonthEnd
from elexon import ElexonRawClient

class PriceDataAccesser():

    '''This class takes in a user's api_key = api_key, start = start date in YYYY-mm-dd,
    end = end date in YYYY-mm-dd and Elexon data code in XXXXXX. Generated data will have 30 min increments.
    Suggest using the 1st day of the month in string format. Builds on ElexonRawClient https://github.com/MichaelKavanagh/elexon.
    Tested with 'DERSYSDATA' - i.e.pricing data and imbalances. This is due to how Exelon API uses to and from dates!

    Methods include:

    .date_ranger(self)
    .price_getter(self)
    .dataframe_maker(self)

    Need to call .price_getter before .dataframe_maker

    by K O'Donnell'''

    global df

    def __init__(self, api_key, start, end, elexon_code):


        self.api_key = api_key
        self.start = start
        self.end = end
        self.elexon_code = elexon_code

    def date_ranger(self):

        '''This method simply generates a the list of dates within user defined start and end'''

        for beg in pd.date_range(self.start, self.end, freq='MS'):
            print(beg.strftime("%Y-%m-%d"), (beg + MonthEnd(1)).strftime("%Y-%m-%d"))

    def price_getter(self):

        '''This method pulls data from Exelon. Need to use api key etc as defined previously'''
        self.prices_since_date = []
        self.api = ElexonRawClient(self.api_key)
        # Put in start of and end of month (i.e 31st etc)
        for date in pd.date_range(self.start, self.end, freq='MS'):
            date = self.api.request(self.elexon_code,FromSettlementDate = date.strftime("%Y-%m-%d"),
                                         ToSettlementDate = (date + MonthEnd(1)).strftime("%Y-%m-%d"))
            self.prices_since_date.append(date)

    def dataframe_maker(self):

        '''This method changes the pulled data into a Pandas Dataframe, adds datetimeindex,
        removes settlemnt date as it is redundant at this stage.
        Example usage: desired_dataframe = dataaccesser.dataframe_maker()'''

        for i in range(1,len(self.prices_since_date)):
            s = self.prices_since_date[0]
            s += self.prices_since_date[i]
            self.df = pd.DataFrame(s)
            self.df = self.df.drop('settlementDate', axis = 1)
            self.dates = pd.date_range(start=self.start,periods=len(self.df), freq='30T')
            self.df.index = self.dates

        return self.df

class DemandDataAccesser():

    '''This class takes in a user's api_key = api_key, start = start date in YYYY-mm-dd,
    end = end date in YYYY-mm-dd and Elexon data code in XXXXXX. Generated data will have 30 min increments.
    Suggest using the 1st day of the month in string format. Builds on ElexonRawClient https://github.com/MichaelKavanagh/elexon.
    Tested with 'SYSDEM' - i.e.demand data and imbalances. This is due to how Exelon API uses to and from dates!

    Methods include:

    .date_ranger(self)
    .demand_getter(self)
    .dataframe_maker(self)

    Need to call .demand_getter before .dataframe_maker. Use 'SYSDEM' Elexon code for demand.

    Output is a tuple: choose[0] for ITSDO and choose[1] for TSDF. Other details can be removed using normal Pandas.

    by K O'Donnell'''


    global df_dem, df_ITSDO, df_TSDF

    def __init__(self, api_key, start, end, elexon_code):

        self.api_key = api_key
        self.start = start
        self.end = end
        self.elexon_code = elexon_code

    def date_ranger(self):

        '''This method simply generates a the list of dates within user defined start and end'''

        for beg in pd.date_range(self.start, self.end, freq='MS'):
            print(beg.strftime("%Y-%m-%d"), (beg + MonthEnd(1)).strftime("%Y-%m-%d"))

    def demand_getter(self):

        '''This method pulls data from Exelon. Need to use api key etc as defined previously'''
        self.prices_since_date = []
        self.api = ElexonRawClient(self.api_key)
        # Put in start of and end of motnh (i.e 31st etc)
        for date in pd.date_range(self.start, self.end, freq='MS'):
            date = self.api.request(self.elexon_code,FromDate = date.strftime("%Y-%m-%d"),
                                         ToDate = (date + MonthEnd(1)).strftime("%Y-%m-%d"))
            self.prices_since_date.append(date)

    def dataframe_maker(self):

        '''This method changes the pulled data into a Pandas Dataframe, adds datetimeindex,
        removes settlemnt date as it is redundant at this stage.
        Example usage: desired_dataframe = dataaccesser.dataframe_maker()'''

        for i in range(1,len(self.prices_since_date)):
            s = self.prices_since_date[0]
            s += self.prices_since_date[i]
            self.df_dem = pd.DataFrame(s)
            self.df_ITSDO = self.df_dem[self.df_dem['recordType']=='ITSDO']
            self.df_TSDF = self.df_dem[self.df_dem['recordType']=='TSDF']
            self.df = self.df_dem.drop('startTimeOfHalfHrPeriod', axis = 1)
            self.dates_ITSDO = pd.date_range(start=self.start,periods=len(self.df_ITSDO), freq='30T')
            self.dates_TSDF = pd.date_range(start=self.start,periods=len(self.df_TSDF), freq='30T')
            self.df_ITSDO.index = self.dates_ITSDO
            self.df_TSDF.index = self.dates_TSDF
            self.df_ITSDO = self.df_ITSDO.drop('startTimeOfHalfHrPeriod', axis = 1)
            self.df_TSDF = self.df_TSDF.drop('startTimeOfHalfHrPeriod', axis = 1)

        return self.df_ITSDO, self.df_TSDF, self.df_dem
