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
               DROP MATERIALIZED VIEW IF EXISTS business_view               
               """)
cursor.execute("""
               DROP MATERIALIZED VIEW IF EXISTS review_view               
               """)
cursor.execute("""
               CREATE MATERIALIZED  VIEW business_view AS
               SELECT 
               id as id,
               (data->>'business_id') as business_id,
               (data->>'name') as name,
               (data->>'city') as city,
               (data->>'state') as state,
               (data->>'stars') as stars
               
               FROM business        
               ;
               CREATE INDEX i_business_city ON business_view(city);
               CREATE INDEX i_business_state ON business_view(state);
               CREATE INDEX i_business_stars ON business_view(stars);
               """)
cursor.execute("""
               CREATE MATERIALIZED VIEW review_view AS
               SELECT 
               id as id,
               (data->>'business_id') as business_id,
               (data->>'user_id') as user_id,
               (data->>'review_id') as review_id,
               (data->>'stars') as stars,
               (data->>'text') as text,
               (data->>'date') as date
               
               FROM review        
               ;
               
               CREATE INDEX i_review_stars ON review_view(stars);
               CREATE INDEX i_review_date ON review_view(date);
               """)

cursor.execute("""
               CREATE MATERIALIZED VIEW users_view AS
               SELECT 
               id as id,
               (data->>'name') as name,
               (data->>'user_id') as user_id,
               (data->>'review_count') as review_count,
               (data->>'average_stars') as average_stars
               
               
               FROM users        
               ;
               
               CREATE INDEX i_users_stars ON review_view(stars);
               
               """)


dbconn.commit()
print("Committed.")