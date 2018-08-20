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
               SELECT r.text FROM 
               business_view b
               INNER JOIN review_view r
               ON b.business_id = r.business_id
               WHERE b.name = 'Chipotle Mexican Grill'
               
               
               """)
reviews = cursor.fetchall()

word_count = {}

for i in reviews:
    text = i[0]
    text = text.translate({ord(c): None for c in '!@#$.,"'})
    text = text.split()
    for word in text:
        if word in word_count:
            word_count[word] += 1
        else:
            word_count[word] = 1

print(sorted(word_count, key=word_count.__getitem__))
# print("Fetched")

# dbconn.commit()