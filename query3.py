# -*- coding: utf-8 -*-
"""
Created on Thu Aug  9 23:05:31 2018

@author: JG
"""

import os
import sys
import logging

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
               SELECT count(r.review_id), b.name FROM 
               (SELECT * FROM business_view 
               WHERE business_view.city = 'Toronto'
               ) b
               INNER JOIN review_view r
               ON b.business_id = r.business_id
               GROUP BY b.name
               order by count DESC
               LIMIT 10;
               """)
print(cursor.fetchall())
# print("Fetched")

# dbconn.commit()