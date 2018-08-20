# -*- coding: utf-8 -*-
"""
Created on Sat Jul 21 16:33:49 2018

@author: JG
"""

#!/usr/bin/env python

import os
import sys
import logging

try:
    import psycopg2 as pg
    import psycopg2.extras
except:
    print( "Install psycopg2")
    exit(123)

try:
    import progressbar
except:
    print( "Install progressbar2")
    exit(123)

import json

import logging
logger = logging.getLogger()

PG_CONN_STRING = "dbname='postgres' port='5432' user='postgres' password='phludphlud'"

data_dir = r"C:\Users\JG\Desktop\yelp_dataset (2)"
dbconn = pg.connect(PG_CONN_STRING)

logger.info("Loading data from '{}'".format(data_dir))

cursor = dbconn.cursor()

counter = 0
empty_files = []

class ProgressInfo:

    def __init__(self, dir):
        files_no = 0
        for root, dirs, files in os.walk(dir):
            for file in files:
                if file.endswith(".json"):
                    files_no += 1
        self.files_no = files_no
        print( "Found {} files to process".format(self.files_no))
        self.bar = progressbar.ProgressBar(maxval=self.files_no,
                                           widgets=[' [', progressbar.Timer(), '] [', progressbar.ETA(), '] ', progressbar.Bar(),])

    def update(self, counter):
        self.bar.update(counter)

pi = ProgressInfo(os.path.expanduser(data_dir))

# =============================================================================
# for root, dirs, files in os.walk(os.path.expanduser(data_dir)):
#     for f in files:
#         fname = os.path.join(root, f)
#         print(fname)
# =============================================================================
        
fname = r"C:\Users\JG\Desktop\yelp_dataset (2)\yelp_academic_dataset_tip.json"
import json
lines = []
for line in open(fname, 'r', encoding="utf-8"):
    print(line)
#    data = js.read()
#    import json
#    dd = json.loads(data)
    lines.append(json.loads(line))
    cursor.execute("""
                              INSERT INTO tip(data)
                              VALUES (%s)
                             
                          """, (line, ))
# =============================================================================
#          if not fname.endswith(".json"):
#              continue
#          with open(fname) as js:
#              data = js.read()
#              if not data:
#                  empty_files.append(fname)
#                  continue
#              import json
#              dd = json.loads(data)
#              counter += 1
#              pi.update(counter)
#              cursor.execute("""
#                              INSERT INTO stats_data(data)
#                              VALUES (%s)
#                              ON CONFLICT ON CONSTRAINT no_overlapping_jsons DO NOTHING
#                          """, (data, ))
# =============================================================================




print( "")

logger.debug("Refreshing materialized views")
# cursor.execute("""REFRESH MATERIALIZED VIEW sessions""");
cursor.execute("""ANALYZE""");

dbconn.commit()

logger.info("Loaded {} files".format(counter))
logger.info("Found {} empty files".format(len(empty_files)))
if empty_files:
    logger.info("Empty files:")
    for f in empty_files:
        logger.info(" >>> {}".format(f))