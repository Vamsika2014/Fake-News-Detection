# Fake-News-Detection
For importing libraries:
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import re
import nltk
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import HashingVectorizer
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
To read a data set from csv file:
df
pd.read_csv ('fake news/train.csv')
Tokenization:
from
nltk.tokenize import word_tokenize
text =="Hello everyone. Welcome to pbl lab we are explainng
about fake news detection"
word_tokenize (text)
CONVERTING LABELS:
df.label = df.label.astype(str)
df.label = df.label.str.strip
dict = { 'REAL' : '1' , 'FAKE' : '
df ['label'] = df ['label']. dict df.head
x_df = df['total']
y_df = df ['
Vectorization:
from
sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from
sklearn.feature_extraction.text import TfidfVectorizer
count_vectorizer
= CountVectorizer
count_vectorizer.fit_transform x_df
freq_term_matrix = count_vectorizer.transform x_df
tfidf = TfidfTransformer(norm = "l2")
tfidf.fit freq_term_matrix
tf_idf_matrix = tfidf.fit_transform freq_term_matrix
print(tf_idf_matrix)
Modeling:
x_train
, x_test , y_train , y_test = train_test_split tf_idf_matrix,y_df ,
random_state =
Naïve bayes:
from
sklearn.naive_bayes import MultinomialNB
NB = MultinomialNB()
NB.fit x_train , y_train
Accuracy = NB.score x_test , y_test
print(Accuracy*100)
