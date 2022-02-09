#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul  7 13:53:14 2020

@author: kieranodonnell
"""

# Hourly 
# URL = https://my.elexys.be/MarketInformation/SpotBelpex.aspx
import requests
from bs4 import BeautifulSoup
import pandas as pd


url = 'https://my.elexys.be/MarketInformation/SpotBelpex.aspx'
page = requests.get(url)
page_content = page.content
soup = BeautifulSoup(page_content,'html.parser')
tabl = soup.find_all("table", {"class" : "dxgvControl_Office2010Blue dxgv"})

data_raw = []
for t in tabl:
    rows = t.find_all("tr")
    for row in rows:
        data_raw.append(row.get_text(separator = '').split("€"))
        
data_cleaned = []
for i in range(len(data_raw)):
    if len(data_raw[i]) == 2:
        data_cleaned.append(data_raw[i])
    
    
df = pd.DataFrame.from_records(data_cleaned)
df.info()