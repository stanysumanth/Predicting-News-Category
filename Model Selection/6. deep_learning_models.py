
import pandas as pd
import numpy as np
import re

import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer 

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

data = pd.read_csv('/content/drive/My Drive/datascience/Machine Learning/Board Infinity/portfolio/news-data.csv')
data.head()

# le = LabelEncoder()
# data['category'] = le.fit_transform(data['category'])

nltk.download('stopwords')
all_stopwords = set(stopwords.words('english'))

def clean_text(text):
    """
        text: a string        
        return: modified initial string
    """
    text = re.sub('[^a-zA-Z]', ' ', text)
    text = text.lower()
    text = text.split()
    stemmer = PorterStemmer()
    text = ' '.join([stemmer.stem(word)for word in text if word not in all_stopwords])
    return text

data['text'] = data['text'] .apply(clean_text)

data.head()

article = data['text']
label = data['category']

train_X, X_test, train_y, y_test  = train_test_split(article, label, test_size = 0.2, random_state = 0)
X_train, X_val, y_train, y_val = train_test_split(train_X, train_y, test_size = 0.15, random_state = 0)

print(X_train.shape)
print(y_train.shape)
print(X_val.shape)
print(y_val.shape)
print(X_test.shape)
print(y_test.shape)

num_words = 2000
oov_token = '<UNK>'
pad_type = 'post'
trunc_type = 'post'

# Tokenize our training data
tokenizer = Tokenizer(num_words=num_words, oov_token=oov_token)
tokenizer.fit_on_texts(X_train)

# Get our training data word index
word_index = tokenizer.word_index

# Encode training data sentences into sequences
train_sequences = tokenizer.texts_to_sequences(X_train)

# Get max training sequence length
maxlength = max([len(x) for x in train_sequences])

# Pad the training sequences
train_padded = pad_sequences(train_sequences, padding=pad_type, truncating=trunc_type, maxlen=maxlength)

# Output the results of our work
print("Word index:\n", word_index)
print("\nTraining sequences:\n", train_sequences)
print("\nPadded training sequences:\n", train_padded)
print("\nPadded training shape:", train_padded.shape)
print("Training sequences data type:", type(train_sequences))
print("Padded Training sequences data type:", type(train_padded))

# for x, y in zip(X_train, train_padded):
#   print('{} -> {}'.format(x, y))

# print("\nWord index (for reference):", word_index)

# tokenizing validation data
validation_sequences = tokenizer.texts_to_sequences(X_val)
validation_padded = pad_sequences(validation_sequences, padding=pad_type, truncating=trunc_type,  maxlen=maxlength)

print("Validation sequences:\n", validation_sequences)
print("\nPadded validation sequences:\n", validation_padded)
print("\nPadded validation shape:",validation_padded.shape)

# tokenizing test data
test_sequences = tokenizer.texts_to_sequences(X_test)
test_padded = pad_sequences(test_sequences, padding=pad_type, truncating=trunc_type, maxlen=maxlength)

print("Testing sequences:\n", test_sequences)
print("\nPadded testing sequences:\n", test_padded)
print("\nPadded testing shape:",test_padded.shape)

print(set(label))

label_tokenizer = Tokenizer()
label_tokenizer.fit_on_texts(y_train)

training_label_seq = np.array(label_tokenizer.texts_to_sequences(y_train))
validation_label_seq = np.array(label_tokenizer.texts_to_sequences(y_val))
testing_label_seq = np.array(label_tokenizer.texts_to_sequences(y_test))

print(training_label_seq[0])
print(training_label_seq[1])
print(training_label_seq[2])
print(training_label_seq.shape)
print()
print(validation_label_seq[0])
print(validation_label_seq[1])
print(validation_label_seq[2])
print(validation_label_seq.shape)
print()
print(testing_label_seq[0])
print(testing_label_seq[1])
print(testing_label_seq[2])
print(testing_label_seq.shape)

model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Dense(10, activation='relu'))
model.add(tf.keras.layers.Dense(10, activation='relu'))
model.add(tf.keras.layers.Dense(5, activation='softmax'))
model.compile(optimizer = 'adam', loss = 'sparse_categorical_crossentropy' , metrics = ['accuracy'])

num_epochs = 10
train_pred = model.fit(X_train, y_train, epochs=num_epochs, validation_data=(X_val, y_val), verbose=2)

