#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep 18 21:00:56 2021

@author: lakshmipeesapati
"""

import sys
import re
import os

import numpy as np
from numpy import dot
from numpy.linalg import norm
from pyspark import SparkContext
from pyspark.sql import SparkSession
from pyspark.sql import SQLContext


# In[3]:

output_str = ""
sc = SparkContext('local', 'Spark SQL')
sqlContext = SQLContext(sc)
output_str = ""


# In[4]:


#wikiPagesFile=sys.argv[1]
wikiCategoryFile=sys.argv[1]


# In[5]:



wikiCategoryLinks = sc.textFile(wikiCategoryFile)
#wikiCategoryLinks.take(2)

wikiCats=wikiCategoryLinks.map(lambda x: x.split(",")).map(lambda x: (x[0].replace('"', ''), x[1].replace('"', '') ))

df =sqlContext.createDataFrame(wikiCats, ['id', 'cat'])
#print("Count of records = ",df.count())

df_cat_count = df.groupBy('cat').count()
output_str = output_str + "Count of categories: " + str(df_cat_count.count()) + "\n**************\n"
output_str = output_str + "Average number of categories is: " + str(df_cat_count.agg({'count': 'avg'}).collect()) + "\n**************\n"
output_str = output_str + "Minimum number of categories is: " + str(df_cat_count.agg({'count': 'min'}).collect()) + "\n**************\n"
output_str = output_str + "Maximum number of categories is: " + str(df_cat_count.agg({'count': 'max'}).collect()) + "\n**************\n"

output_str = output_str + "Top Categories are: \n" + str(df_cat_count.orderBy("count", ascending=False).limit(10).collect())
print(output_str)

#path = os.path.join(sys.argv[2])
#with open(path, "w") as txtFile:
#    txtFile.write(output_str)
sc.stop()
