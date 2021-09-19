#!/usr/bin/env python
# coding: utf-8

# In[2]:


import sys
import re
import os

import numpy as np
from numpy import dot
from numpy.linalg import norm
from pyspark import SparkContext
from pyspark.sql import SparkSession


# In[3]:


sc = SparkContext()
output_str = ""


# In[4]:


wikiPagesFile=sys.argv[1]
wikiCategoryFile=sys.argv[2]


# In[5]:



wikiCategoryLinks = sc.textFile(wikiCategoryFile)
#wikiCategoryLinks.take(2)

wikiCats=wikiCategoryLinks.map(lambda x: x.split(",")).map(lambda x: (x[0].replace('"', ''), x[1].replace('"', '') ))


# In[6]:


wikiPages = sc.textFile(wikiPagesFile)
#wikiPages.take(2)


# In[7]:


#wikiCats.take(1)


# In[8]:


spark = SparkSession.builder.master("local[*]").getOrCreate()


# In[9]:


df = spark.read.csv(wikiPagesFile)
#df.take(1)


# In[10]:


#wikiPages.take(1)


# In[11]:


numberOfDocs = wikiPages.count()
#print(numberOfDocs)


# In[12]:


validLines = wikiPages.filter(lambda x : 'id' in x and 'url=' in x)
keyAndText = validLines.map(lambda x : (x[x.index('id="') + 4 : x.index('" url=')], x[x.index('">') + 2:][:-6]))
#keyAndText.take(1)


# In[13]:


def buildArray(listOfIndices):
    returnVal = np.zeros(5000)
    for index in listOfIndices:
        returnVal[index] = returnVal[index] + 1
    mysum = np.sum(returnVal)
    returnVal = np.divide(returnVal, mysum)
    return returnVal

def build_zero_one_array (listOfIndices):
    returnVal = np.zeros (5000)
    for index in listOfIndices:
        if returnVal[index] == 0: returnVal[index] = 1
    return returnVal

def stringVector(x):
    returnVal = str(x[0])
    for j in x[1]:
        returnVal += ',' + str(j)
    return returnVal

def cousinSim (x,y):
    normA = np.linalg.norm(x)
    normB = np.linalg.norm(y)
    return np.dot(x,y)/(normA*normB)


# In[14]:


# Now, we split the text in each (docID, text) pair into a list of words
# After this step, we have a data set with
# (docID, ["word1", "word2", "word3", ...])
# We use a regular expression here to make
# sure that the program does not break down on some of the documents
regex = re.compile('[^a-zA-Z]')
keyAndListOfWords = keyAndText.map(lambda x : (str(x[0]), regex.sub(' ', x[1]).lower().split()))


# In[15]:


# Now get the top 20,000 words... first change (docID, ["word1", "word2",␣,→"word3", ...])
# to ("word1", 1) ("word2", 1)...
allWords = keyAndListOfWords.flatMap(lambda x: x[1]).map(lambda x: (x,1))
#allWords.take(10)


# In[16]:


# Now, count all of the words, giving us ("word1", 1433), ("word2", 3423423),␣,→etc.
allCounts = allWords.reduceByKey(lambda x,y: x+y)
#allCounts.take(10)


# In[17]:


# Get the top 20,000 words in a local array in a sorted format based on␣,→frequency
# If you want to run it on your laptio, it may a longer time for top 20k words.
topWords = allCounts.top(5000, lambda x:x[1])
#
#print("Top Words in Corpus:", allCounts.top(10, key=lambda x: x[1]))


# In[18]:


# We'll create a RDD that has a set of (word, dictNum) pairs
# start by creating an RDD that has the number 0 through 20000
# 20000 is the number of words that will be in our dictionary
topWordsK = sc.parallelize(range(5000))

# Now, we transform (0), (1), (2), ... to ("MostCommonWord", 1)
# ("NextMostCommon", 2), ...
# the number will be the spot in the dictionary used to tell us
# where the word is located
dictionary = topWordsK.map (lambda x : (topWords[x][0], x))
#print("Word Postions in our Feature Matrix. Last 20 words in 20k positions: ",dictionary.top(20, lambda x : x[1]))


# In[19]:


################### TASK 2 ##################
# Next, we get a RDD that has, for each (docID, ["word1", "word2", "word3", ...,→]),
# ("word1", docID), ("word2", docId), ...
allWordsWithDocID = keyAndListOfWords.flatMap(lambda x: ((j, x[0]) for j in x[1]))
# Now join and link them, to get a set of ("word1", (dictionaryPos, docID)) pairs
allDictionaryWords = allWordsWithDocID.join(dictionary)

# Now, we drop the actual word itself to get a set of (docID, dictionaryPos) pairs
justDocAndPos = allDictionaryWords.map(lambda x : x[1])

# Now get a set of (docID, [dictionaryPos1, dictionaryPos2, dictionaryPos3...]) pairs
allDictionaryWordsInEachDoc = justDocAndPos.groupByKey()
#allDictionaryWordsInEachDoc.take(1)


# In[20]:


# The following line this gets us a set of
# (docID, [dictionaryPos1, dictionaryPos2, dictionaryPos3...]) pairs
# and converts the dictionary positions to a bag-of-words numpy array...
allDocsAsNumpyArrays = allDictionaryWordsInEachDoc.map(lambda x: (x[0],buildArray(x[1])))
allDocsAsNumpyArrays_3 = allDocsAsNumpyArrays.take(3)
print(allDocsAsNumpyArrays_3)
output_str = output_str+str(allDocsAsNumpyArrays_3)
#sc.parallelize(allDocsAsNumpyArrays.take(3)).saveAsTextFile(sys.argv[2])                


# In[21]:


# Now, create a version of allDocsAsNumpyArrays where, in the array,
# every entry is either zero or one.
# A zero means that the word does not occur,
# and a one means that it does.
zeroOrOne = allDictionaryWordsInEachDoc.map(lambda x: (x[0],build_zero_one_array(x[1])))
#print(zeroOrOne.take(3))     


# In[22]:


# Now, add up all of those arrays into a single array, where the
# i^th entry tells us how many
# individual documents the i^th word in the dictionary appeared in
dfArray = zeroOrOne.reduce(lambda x1, x2: ("", np.add(x1[1], x2[1])))[1]


# In[23]:


# Create an array of 20,000 entries, each entry with the value numberOfDocs (number of docs)
multiplier = np.full(5000, numberOfDocs)
# Get the version of dfArray where the i^th entry is the inverse-document frequency for the
# i^th word in the corpus
idfArray = np.log(np.divide(np.full(5000, numberOfDocs), dfArray))
# Finally, convert all of the tf vectors in allDocsAsNumpyArrays to tf * idf vectors
allDocsAsNumpyArraysTFidf = allDocsAsNumpyArrays.map(lambda x: (x[0], np.multiply(x[1], idfArray)))
allDocsAsNumpyArraysTFidf_2 = allDocsAsNumpyArraysTFidf.take(2)
print(allDocsAsNumpyArraysTFidf_2)
output_str = output_str+"\n********************************"
output_str = output_str + str(allDocsAsNumpyArraysTFidf_2)


# In[24]:


#wikiCats.take(1)


# In[25]:


# Now, we join it with categories, and map it after join so that we have only the wikipageID
# This joun can take time on your laptop.
# You can do the join once and generate a new wikiCats data and store it. Our WikiCategories includes all categories
# of wikipedia.
featuresRDD = wikiCats.join(allDocsAsNumpyArraysTFidf).map(lambda x: (x[1][0], x[1][1]))
# Cache this important data because we need to run kNN on this data set.
featuresRDD.cache()
#featuresRDD.take(10)


# In[26]:


# Let us count and see how large is this data set.
#featuresRDD.count()


# In[53]:


# Finally, we have a function that returns the prediction for the label of a string, using a kNN algorithm
def getPrediction (textInput, k):
    # Create an RDD out of the textIput
    myDoc = sc.parallelize (('', textInput))
    # Flat map the text to (word, 1) pair for each word in the doc
    wordsInThatDoc = myDoc.flatMap (lambda x : ((j, 1) for j in regex.sub(' ',x).lower().split()))
    # This will give us a set of (word, (dictionaryPos, 1)) pairs
    allDictionaryWordsInThatDoc = dictionary.join (wordsInThatDoc).map (lambda x: (x[1][1], x[1][0])).groupByKey ()
    
    # Get tf array for the input string
    myArray = buildArray(allDictionaryWordsInThatDoc.top (1)[0][1])
    
    
    #Get the tf * idf array for the input string
    myArray = np.multiply (myArray, idfArray)
    
    # Get the distance from the input text string to all database documents, using cosine similarity (np.dot() )
    distances = featuresRDD.map (lambda x : (x[0], np.dot (x[1], myArray)))
    
    # distances = allDocsAsNumpyArraysTFidf.map (lambda x : (x[0], cousinSim(x[1],myArray)))
    # get the top k distances
    topK = distances.top (k, lambda x : x[1])
    
    # and transform the top k distances into a set of (docID, 1) pairs
    docIDRepresented = sc.parallelize(topK).map (lambda x : (x[0], 1))
    
    # now, for each docID, get the count of the number of times this document ID appeared in the top k
    numTimes = docIDRepresented.reduceByKey(lambda x,y : x+y)
    
    # Return the top 1 of them.
    # Ask yourself: Why we are using twice top() operation here?
    return numTimes.top(k, lambda x: x[1])
    
    


# In[55]:

output_str = output_str+"\n********************************"
pred1 = getPrediction('Sport Basketball Volleyball Soccer', 10)
output_str = output_str + str(pred1)
print(pred1)


# In[56]:

output_str = output_str+"\n********************************"
pred2 = getPrediction('Sport Basketball Volleyball Soccer', 10)
output_str = output_str + str(pred2)
print(pred2)


# In[57]:

output_str = output_str+"\n********************************"
pred3 = getPrediction('How many goals Vancouver score last year?', 10)
output_str = output_str + str(pred3)
print(pred3)


# In[ ]:

print(output_str)
#sc.parallelize(output_str).saveAsTextFile(sys.argv[3])
##with open(path, "w") as txtFile:
#    txtFile.write(output_str)
sc.stop()




