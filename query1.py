# -*- coding: utf-8 -*-
"""
Created on Thu Aug  9 23:05:31 2018

@author: JG
"""

import os
import sys
import logging
import pandas as pd
import pandas.io.sql as psql


try:
    import psycopg2 as pg
    import psycopg2.extras
except:
    print( "Install psycopg2")
    exit(123)
    
PG_CONN_STRING = "dbname='postgres' port='5432' user='postgres' password='phludphlud'"
dbconn = pg.connect(PG_CONN_STRING)
cursor = dbconn.cursor()
cursor.execute("""
               SELECT COUNT(*) FROM business_view
               WHERE business_view.state = 'ON'
               """)

print(cursor.fetchall()[0])

df = pd.read_sql_query('SELECT * from business_view', con=dbconn)
print(df['state'].head())
print(df.shape)

# dbconn.commit()
# print("Committed.")