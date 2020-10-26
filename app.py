import pandas as pd
import numpy as np
import re

import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

from flask import Flask, request, jsonify, render_template
import pickle

app = Flask(__name__)

count_vectorizer = pickle.load(open('count_vectorizer.pickle', 'rb'))
model = pickle.load(open('model.pickle', 'rb'))

nltk.download('stopwords')
all_stopwords = set(stopwords.words('english'))
stemmer = PorterStemmer()


@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    new_data = [str(x) for x in request.form.values()]
    my_string = new_data[0]
    my_string = re.sub('[^a-zAz]', ' ', my_string)
    my_string = my_string.lower()
    my_string = my_string.split()
    my_string = ' '.join([stemmer.stem(word)for word in my_string if word not in all_stopwords])
    new_vector = count_vectorizer.transform([my_string])
    pred = model.predict(new_vector)

    return render_template('index.html', prediction_text='Category of the news should be {}'.format(pred[0].upper()))



if __name__ == "__main__":
    app.run(debug=True)