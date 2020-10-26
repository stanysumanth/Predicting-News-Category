
# importing relevant libraries
import pandas as pd
import numpy as np
import re

import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import accuracy_score, classification_report

# reading the data
data = pd.read_csv('news-data.csv')
data.head()

set(data['category'])

data['category'].value_counts()

# cleaning the text data 
nltk.download('stopwords')
all_stopwords = set(stopwords.words('english'))
stemmer = PorterStemmer()

def clean_text(text):
    """
       text: string
       return: modified initial string
    """
    text = re.sub('[^a-zA-Z]', ' ', text)
    text = text.lower()
    text = text.split()
    text = ' '.join([stemmer.stem(word)for word in text if word not in all_stopwords])
    return text

# cleaning the textual data
data['text'] = data['text'].apply(clean_text)

# seperating the texts and categories
articles = data['text']
categories = data['category']

# tokeninzing the textual data
count_vectorizer = CountVectorizer(stop_words='english', max_features=2000)

articles_count = count_vectorizer.fit_transform(articles)

# splitting the articles and categories into training and testing data
X_train, X_test, y_train, y_test = train_test_split(articles_count, categories, test_size=0.20, random_state=42)

# fitting the kernel SVM Model and training the model
model = AdaBoostClassifier(n_estimators=100)
model.fit(X_train, y_train)

# evaluating the model performance by predicting the test set results
pred = model.predict(X_test)
print(accuracy_score(y_test, pred))

print(classification_report(y_test,pred))