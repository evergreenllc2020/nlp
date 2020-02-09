from __future__ import print_function
from flask import Flask, request, render_template
import os
import sys
import numpy as np
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from keras.layers import Dense, Input, GlobalMaxPooling1D, Dropout
from keras.layers import Conv1D, MaxPooling1D, Embedding
from keras.models import Model
from keras.initializers import Constant
from matplotlib import pyplot
from keras import backend as K
import pandas as pd
from sklearn.utils import shuffle
import keras.backend.tensorflow_backend as tb
from tensorflow.keras.models import model_from_json


tb._SYMBOLIC_SCOPE.value = True


# setting up template directory
TEMPLATE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'templates')

app = Flask(__name__, template_folder=TEMPLATE_DIR, static_folder=TEMPLATE_DIR)
#app = Flask(__name__)




@app.route("/")
def hello():
	return TEMPLATE_DIR


BASE_DIR = '/Volumes/My Passport for Mac/data'
GLOVE_DIR = os.path.join(BASE_DIR, 'glove.6B')
TEXT_DATA_DIR = os.path.join(BASE_DIR, 'imdb_movie_reviews')
MOVIE_REVIEW_FILE_NAME = "imdb_master.csv"
MAX_SEQUENCE_LENGTH = 1000
MAX_NUM_WORDS = 20000
EMBEDDING_DIM = 100
VALIDATION_SPLIT = 0.2

def read_csv(filepath):
     if os.path.splitext(filepath)[1] != '.csv':
          return  # or whatever
     seps = [',', ';', '\t']                    # ',' is default
     encodings = [None, 'utf-8', 'ISO-8859-1']  # None is default
     for sep in seps:
         for encoding in encodings:
              try:
                  return pd.read_csv(filepath, encoding=encoding, sep=sep)
              except Exception:  # should really be more specific 
                  pass
     raise ValueError("{!r} is has no encoding in {} or seperator in {}"
                      .format(filepath, encodings, seps))


# second, prepare text samples and their labels
print('Processing text dataset')
index_to_label_dict = {}
texts = []  # list of text samples
labels_index = {}  # dictionary mapping label name to numeric id
labels = []  # list of label ids

input_df = read_csv(os.path.join(TEXT_DATA_DIR, MOVIE_REVIEW_FILE_NAME))
review_df1 = input_df[['review','label']]
review_df = review_df1[review_df1['label']!='unsup']


review_df = shuffle(review_df)
print("Here are Few Samples in data")
print(review_df.head)

print("Here total number of posistive and negative samples")
print(review_df.groupby(['label']).count())

print("Converting pandas dataframe into lists")
texts = review_df['review'].values.tolist()
labels = []
labels_text = []
labels_text_unique = review_df.label.unique().tolist()
labels_text = review_df['label'].values.tolist()

idxCounter = 0
for label in labels_text_unique:
    labels_index[label] = idxCounter
    index_to_label_dict[idxCounter] = label
    idxCounter = idxCounter + 1;

idxCounter = 0    
for label in labels_text:
    if(idxCounter % 100 == 0):
	    print("processing row " + str(idxCounter))
    labels.append(labels_index[label])
    idxCounter = idxCounter + 1;
    

print("Labels Array")
print(len(labels))
print("Labels Dictionary")
print(labels_index)
print("Done")

print("loading model .....")
# load json and create model
json_file = open('/Volumes/My Passport for Mac/model/sentiment_analysis/model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("/Volumes/My Passport for Mac/model/sentiment_analysis/model.h5")
print("Loaded model from disk")
# evaluate loaded model on test data
print("compiling model")
loaded_model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
print("done")

#finally, vectorize the text samples into a 2D integer tensor
print("Running tokenizer")
tokenizer = Tokenizer(num_words=MAX_NUM_WORDS)
tokenizer.fit_on_texts(texts)
sequences = tokenizer.texts_to_sequences(texts)
print("done")


def getSentiments(review_data):
	reviewList = []
	reviewList.append(review_data)
	test_sequences = tokenizer.texts_to_sequences(reviewList)
	test_data = pad_sequences(test_sequences, maxlen=MAX_SEQUENCE_LENGTH)
	nn_output = loaded_model.predict(test_data)
	print(nn_output)
	i=0
	sentiments = {}
	for idx in np.argmax(nn_output, axis=1):
		print("Category: ", index_to_label_dict[idx])
		print("text: " , reviewList[i])
		print("=====================================")
		sentiments[index_to_label_dict[idx]] = reviewList[i]
		i = i + 1
	return sentiments


@app.route("/sentiments",	 methods=['GET', 'POST'])
def rec():
	query = '' 
	if(request.method == "POST"):
		print("inside post")
		review = request.form.get('review')
		print(review)
		sentiments = getSentiments(review)
		return render_template('classifysentiments.html', review=review, sentiments=sentiments)
	else:
		return render_template('classifysentiments.html', review="" ,sentiments=None)
	

if __name__ == "__main__":
    app.run(debug=True)


