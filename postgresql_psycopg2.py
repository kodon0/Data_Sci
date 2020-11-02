#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 22 13:58:47 2020

@author: kieranodonnell
"""

# After installing with pip install psycopg2
import psycopg2 as pg2
conn = pg2.connect(database='dvdrental', user='postgres',password='pwd')
# Establish connection and start cursor to be ready to query
cur = conn.cursor()
# Pass in a PostgreSQL query as a string
cur.execute("SELECT * FROM payment")

# To save and index results, assign it to a variable
data = cur.fetchmany(10)

query1 = '''
        CREATE TABLE new_table (
            userid integer
            , tmstmp timestamp
            , type varchar(10)
        );
        '''

cur.execute(query1)

# Close after
conn.close
