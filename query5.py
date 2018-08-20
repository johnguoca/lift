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
               WITH mag_reviewers as
               (
               SELECT r.user_id as u_id FROM review_view r
               LEFT JOIN users_view u
               ON r.user_id = u.user_id
               WHERE r.business_id = '4JNXUYY8wbaaDmk3BPzlWw'
               ) 
               
               
               
               SELECT COUNT (DISTINCT u_id) FROM mag_reviewers;
               
                              
               """)

number_mag_reviewers = cursor.fetchone()[0]
print("The number of total Mon Ami Gabi reviewers is {}.".format(number_mag_reviewers))
cursor.execute("""
               WITH mag_reviewers as
               (
               SELECT r.user_id as u_id, u.name as name FROM review_view r
               LEFT JOIN users_view u
               ON r.user_id = u.user_id
               WHERE r.business_id = '4JNXUYY8wbaaDmk3BPzlWw'
               ) 
               SELECT count(*) FROM mag_reviewers
               INNER JOIN
               (
               SELECT count(u.user_id), u.user_id AS u_id
               FROM users_view u
               LEFT JOIN 
               (SELECT r.user_id AS u_id FROM review_view r
               LEFT JOIN business_view b
               ON r.business_id = b.business_id  
               WHERE b.state = 'ON'
               ) e1
               ON u.user_id = e1.u_id
               GROUP BY u.user_id
               HAVING count(u.name) > 10
               ) ont_reviews
               ON mag_reviewers.u_id = ont_reviews.u_id
               
               """)

number_mag_reviewers_ON10 = cursor.fetchone()[0]
print("The number of Mon Ami Gabi reviewers who also reviewed at least 10 restaurants in Ontario is {}.".format(number_mag_reviewers_ON10))

# print("Fetched")

# dbconn.commit()