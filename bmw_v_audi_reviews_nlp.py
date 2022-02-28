# -*- coding: utf-8 -*-
"""
Created on Sun Feb 20 18:54:14 2022

@author: lafisher
"""

# Imports necessary Python libraries
import pandas as pd
import re
from sklearn.decomposition import NMF
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import plotly.graph_objects as go
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

## read and clean
audi = pd.read_csv("audi_clean.csv", encoding='cp1252')
audi = audi.iloc[:, 2:8]
bmw = pd.read_csv("bmw_clean.csv", lineterminator='\n', encoding='cp1252')
bmw = bmw.iloc[:, 2:8]

bmw_review_list = list(bmw['Review'])
bmw_review = [b for b in bmw_review_list if isinstance(b, str)]

audi_review_list = list(audi['Review'])
audi_review = [b for b in audi_review_list if isinstance(b, str)]

## Text cleaning
audi_review = [each_string.lower() for each_string in audi_review]
bmw_review = [each_string.lower() for each_string in bmw_review]
audi_review = [re.sub(
    r"(@\[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)|^rt|http.+?", "", a) for a in audi_review]
bmw_review = [re.sub(
    r"(@\[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)|^rt|http.+?", "", b) for b in bmw_review]

all_stopwords = ['weve', 'got', 'sure', 'hows', 'going', 'were''must', 'becomes', 'fire', 'several', 'very', 'on', 'him', 'next', 'perhaps', 'alone', 'hereupon', 'be', 'until', 'via', 'go', 'whom', 'none', 'except', 'per', 'as', 'whither', 'just', 'never', 'whereupon', 'hereafter', 'both', 'twenty', 'indeed', 'for', 'anyhow', 'made', 'km', 'being', 'yet', 'about', 'wherein', 'onto', 'whatever', 'front', 'forty', 'whereby', 'seems', 'over', 'two', 'through', 'bottom', 'side', 'something', 'any', 'many', 'was', 'whence', 'but', 'itself', 'these', 'more', 'only', 'find', 'done', 'hers', 'thence', 'without', 'if', 'along', 'amount', 'why', 'after', 'of', 'thick', 'further', 'rather', 'when', 'am', 'between', 'had', 'are', 'keep', 'eight', 'system', 'yourselves', 'de', 'its', 'take', 'while', 'nothing', 'etc', 'out', 'anyway', 'thereafter', 'whole', 'since', 'or', 'show', 'ltd', 'eg', 'describe', 'also', 'same', 'six', 'thereupon', 'under', 'yours', 'thin', 'really', 'behind', 'hasnt', 'at', 'thereby', 'within', 'inc', 'a', 'cry', 'might', 'wherever', 'sometime', 'herself', 'beyond', 'which', 'enough', 'again', 'cant', 'because', 'than', 'whenever', 'to', 'fill', 'from', 'third', 'whereas', 'in', 'it', 'co', 'once', 'ie', 'before', 'sometimes', 'therein', 'would', 'during', 'someone', 'they', 'last', 'seemed', 'sincere', 'less', 'five', 'computer', 'few', 'will', 'move', 'put', 'against', 'you', 'latterly', 'by', 'full', 'where', 'each', 'whether', 'throughout', 'all', 'did', 'became', 'himself', 'four', 'themselves', 'ten',
                 'were', 'others', 'couldnt', 'eleven', 'too', 'namely', 'off', 'now', 'nine', 'whose', 'always', 'please', 'fifteen', 'yourself', 'everything', 'call', 'no', 'amoungst', 'ours', 'another', 'around', 'above', 'her', 'have', 'neither', 'beside', 'quite', 'become', 'regarding', 'mill', 'own', 'somewhere', 'me', 'say', 'here', 'using', 'she', 'thus', 'myself', 'may', 'kg', 'upon', 'there', 'see', 'however', 'an', 'us', 'nobody', 'moreover', 'across', 'otherwise', 'ourselves', 'into', 'amongst', 'don', 'nor', 'not', 'already', 'least', 'empty', 'else', 'various', 'other', 'nowhere', 'anywhere', 'is', 'well', 'twelve', 'becoming', 'that', 'such', 'hereby', 'former', 'nevertheless', 'he', 'bill', 'toward', 'hundred', 'cannot', 'seem', 'doesn', 'back', 'everyone', 'though', 'fifty', 'could', 'sixty', 'serious', 'top', 'almost', 'the', 'although', 'doing', 'among', 'con', 'we', 'therefore', 'first', 'below', 'anyone', 'down', 'used', 'those', 'herein', 'often', 'elsewhere', 'and', 'then', 'somehow', 'detail', 'every', 'do', 'towards', 'give', 'thru', 'part', 'whereafter', 'still', 'either', 'due', 'how', 'seeming', 'what', 'interest', 'i', 'found', 'make', 'your', 'three', 'even', 'whoever', 'his', 'unless', 'mine', 'afterwards', 'noone', 'latter', 'most', 'formerly', 'can', 'anything', 'name', 'together', 'who', 'our', 'with', 'get', 'didn', 'this', 'has', 'up', 'should', 'them', 'mostly', 'one', 'some', 're', 'hence', 'ever', 'meanwhile', 'besides', 'beforehand', 'much', 'un', 'my', 'their', 'everywhere', 'been', 'so', 'does']
audi_review = [word for word in audi_review if word.lower()
               not in all_stopwords]
bmw_review = [word for word in bmw_review if word.lower() not in all_stopwords]


## AUDI OVERALL
# use tfidf to create trigrams
vect = TfidfVectorizer(stop_words='english', ngram_range=(3, 3))
# Fit and transform
audi_tfidf = vect.fit_transform(audi_review)
# Save the feature names for later to create topic summaries
tfidf_fn = vect.get_feature_names()


# Applying Non-Negative Matrix Factorization

nmf = NMF(n_components=1, solver="mu")
W = nmf.fit_transform(audi_tfidf)
H = nmf.components_


def show_topics(vectorizer=vect, lda_model=nmf, n_words=20):
    keywords = np.array(vectorizer.get_feature_names())
    topic_keywords = []
    for topic_weights in lda_model.components_:
        top_keyword_locs = (-topic_weights).argsort()[:n_words]
        topic_keywords.append(keywords.take(top_keyword_locs))
    return topic_keywords


topic_keywords_audi = show_topics(vectorizer=vect, lda_model=nmf, n_words=5)

## BMW OVERALL
# Fit and transform
bmw_tfidf = vect.fit_transform(bmw_review)
words = np.array(vect.get_feature_names())

# Applying Non-Negative Matrix Factorization

nmf = NMF(n_components=1, solver="mu")
W = nmf.fit_transform(bmw_tfidf)
H = nmf.components_

topic_keywords_bmw = show_topics(vectorizer=vect, lda_model=nmf, n_words=5)


## CLEAN AND MODEL AUDI 5 STAR
audi5 = audi[audi.Rating == 5]
audi5_review_list = list(audi5['Review'])
audi5_review = [b for b in audi5_review_list if isinstance(b, str)]
audi5_review = [each_string.lower() for each_string in audi5_review]
audi5_review = [re.sub(
    r"(@\[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)|^rt|http.+?", "", a) for a in audi5_review]

audi5_review = [word for word in audi5_review if word.lower()
                not in all_stopwords]

# use tfidf
# Fit and transform
audi5_tfidf = vect.fit_transform(audi5_review)
words = np.array(vect.get_feature_names())
# Applying Non-Negative Matrix Factorization
nmf = NMF(n_components=1, solver="mu")
W = nmf.fit_transform(audi5_tfidf)
H = nmf.components_

topic_keywords_audi_5 = show_topics(vectorizer=vect, lda_model=nmf, n_words=5)

## CLEAN AND MODEL AUDI 1 STAR
audi1 = audi[audi.Rating == 1]
audi1_review_list = list(audi1['Review'])
audi1_review = [b for b in audi1_review_list if isinstance(b, str)]


audi1_review = [each_string.lower() for each_string in audi1_review]
audi1_review = [re.sub(
    r"(@\[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)|^rt|http.+?", "", a) for a in audi1_review]

audi1_review = [word for word in audi1_review if word.lower()
                not in all_stopwords]

# use tfidf
# Fit and transform
audi1_tfidf = vect.fit_transform(audi1_review)
words = np.array(vect.get_feature_names())
# Applying Non-Negative Matrix Factorization

nmf = NMF(n_components=1, solver="mu")
W = nmf.fit_transform(audi1_tfidf)
H = nmf.components_
topic_keywords_audi_1 = show_topics(vectorizer=vect, lda_model=nmf, n_words=5)

##CLEAN AND MODEL BMW 5 STAR
bmw5 = bmw[bmw.Rating == 5]

bmw5_review_list = list(bmw5['Review'])
bmw5_review = [b for b in bmw5_review_list if isinstance(b, str)]


bmw5_review = [each_string.lower() for each_string in bmw5_review]
bmw5_review = [re.sub(
    r"(@\[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)|^rt|http.+?", "", a) for a in bmw5_review]

bmw5_review = [word for word in bmw5_review if word.lower()
               not in all_stopwords]


# Fit and transform
bmw5_tfidf = vect.fit_transform(bmw5_review)
words = np.array(vect.get_feature_names())
# Applying Non-Negative Matrix Factorization

nmf = NMF(n_components=1, solver="mu")
W = nmf.fit_transform(bmw5_tfidf)
H = nmf.components_

topic_keywords_bmw_5 = show_topics(vectorizer=vect, lda_model=nmf, n_words=5)


##CLEAN AND MODEL BMW 1 STAR


bmw1 = bmw[bmw.Rating == 1]

bmw1_review_list = list(bmw1['Review'])
bmw1_review = [b for b in bmw1_review_list if isinstance(b, str)]


bmw1_review = [each_string.lower() for each_string in bmw1_review]
bmw1_review = [re.sub(
    r"(@\[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)|^rt|http.+?", "", a) for a in bmw1_review]

bmw1_review = [word for word in bmw1_review if word.lower()
               not in all_stopwords]

# use tfidf
# Fit and transform
bmw1_tfidf = vect.fit_transform(bmw1_review)
words = np.array(vect.get_feature_names())
# Applying Non-Negative Matrix Factorization

nmf = NMF(n_components=1, solver="mu")
W = nmf.fit_transform(bmw1_tfidf)
H = nmf.components_
topic_keywords_BMW_1 = show_topics(vectorizer=vect, lda_model=nmf, n_words=5)


##TRIGRMAS

def get_top_n_bigram(corpus, n=None):
    vec = CountVectorizer(ngram_range=(3, 3), stop_words='english').fit(corpus)
    bag_of_words = vec.transform(corpus)
    sum_words = bag_of_words.sum(axis=0)
    words_freq = [(word, sum_words[0, idx])
                  for word, idx in vec.vocabulary_.items()]
    words_freq = sorted(words_freq, key=lambda x: x[1], reverse=True)
    return words_freq[:n]


common_words_audi = get_top_n_bigram(audi_review, 20)
common_words_bmw = get_top_n_bigram(bmw_review, 20)






