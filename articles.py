import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import random
import re
import nltk
from collections import Counter

articleNames = ['Protest 1 NYTimes', 'Protest 2 BBC', 'Trump 1 BBC', 'Trump 2 NYTimes', 'brexit 1', 'brexit 2', 'rugby 1', 'rugby 2']

articles = pd.DataFrame()
articles['name'] = articleNames

links = []
articleBodies = []
for articleName in articleNames:
    with open('data/'+articleName+'.txt', 'r') as file:
        link = file.readline()
        content = file.read().lower()

        links.append(link)
        articleBodies.append(content)

articles['link'] = links
articles['content'] = articleBodies

with open('top1000EnglishWords.txt', 'r') as file:
    frequentEnglishWords = set(file.read().splitlines())

# wordFeatures = set()
wordFeatures = []
for (index, content) in articles.content.items():
    content = re.sub("\d+", "", content) # Remove numbers from content
    words = nltk.word_tokenize(content) # Split string into words

    # Remove single/double character words and most frequent english words
    words = [word for word in words if word not in frequentEnglishWords and len(word) > 3]

    c = Counter(words)
    for (word, count) in c.most_common(20):
        wordFeatures.append(word)

    # wordFeatures += words
    # wordFeatures.update(words)

# Remove words if they only occur in a single article
wordFeatures = [word for word in wordFeatures if wordFeatures.count(word) > 1]
wordFeatures = set(wordFeatures)
# print(len(wordFeatures))

X = pd.DataFrame()

for word in wordFeatures:
    featureValueVector = []
    for (bla, content) in articles.content.items():
        wordCount = content.count(word)
        featureValueVector.append(wordCount)

    X[word] = featureValueVector

# print(X.head())
from sklearn.cluster import KMeans

result = KMeans(n_clusters=4, n_init=50).fit(X)

articles['label'] = result.labels_
print(articles[['name', 'label']])

# Calculate each data sample's distance to each centroid
X_dist = result.transform(X)

# Iteratively show a bar chart of x=sample, y=distance for each centroid
for centroid in range(len(X_dist[0])):
    distances = []
    for rowIndex in range(len(X_dist)):
        distances.append(X_dist[rowIndex][centroid])

    plt.figure(figsize=(16,10))
    plt.bar(articles['name'], distances)
    plt.title("Distance of each Article from Cluster Centroid #" + str(centroid))
    plt.show()
