from __future__ import print_function
from flask import Flask, request, render_template
import os



import os
import sys
import numpy as np
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from keras.layers import Dense, Input, GlobalMaxPooling1D
from keras.layers import Conv1D, MaxPooling1D, Embedding
from keras.models import Model
from keras.initializers import Constant
from keras.models import model_from_json


# setting up template directory
TEMPLATE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'templates')

app = Flask(__name__, template_folder=TEMPLATE_DIR, static_folder=TEMPLATE_DIR)
#app = Flask(__name__)




@app.route("/")
def hello():
	return TEMPLATE_DIR

BASE_DIR = '/Volumes/My Passport for Mac/data'
GLOVE_DIR = os.path.join(BASE_DIR, 'glove.6B')
TEXT_DATA_DIR = os.path.join(BASE_DIR, '20_newsgroup')
MAX_SEQUENCE_LENGTH = 1000
MAX_NUM_WORDS = 20000
EMBEDDING_DIM = 100
VALIDATION_SPLIT = 0.2

# second, prepare text samples and their labels
print('Processing text dataset')

texts = []  # list of text samples
labels_index = {}  # dictionary mapping label name to numeric id
index_to_label_dict = {}
labels = []  # list of label ids
for name in sorted(os.listdir(TEXT_DATA_DIR)):
    path = os.path.join(TEXT_DATA_DIR, name)
    if os.path.isdir(path):
        label_id = len(labels_index)
        labels_index[name] = label_id
        index_to_label_dict[label_id] = name
        for fname in sorted(os.listdir(path)):
            if fname.isdigit():
                fpath = os.path.join(path, fname)
                args = {} if sys.version_info < (3,) else {'encoding': 'latin-1'}
                with open(fpath, **args) as f:
                    t = f.read()
                    i = t.find('\n\n')  # skip header
                    if 0 < i:
                        t = t[i:]
                    texts.append(t)
                labels.append(label_id)

print('Found %s texts.' % len(texts))
#print(texts.shape)
print(labels[0])

print("loading model .....")
# load json and create model
json_file = open('/Volumes/My Passport for Mac/model/model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("/Volumes/My Passport for Mac/model/model.h5")
print("Loaded model from disk")
 
# evaluate loaded model on test data
loaded_model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
print("done")

#finally, vectorize the text samples into a 2D integer tensor
tokenizer = Tokenizer(num_words=MAX_NUM_WORDS)
tokenizer.fit_on_texts(texts)
sequences = tokenizer.texts_to_sequences(texts)




def getNewsClassification(news_data):
	newsList = []
	newsList.append(news_data)
	test_sequences = tokenizer.texts_to_sequences(newsList)
	test_data = pad_sequences(test_sequences, maxlen=MAX_SEQUENCE_LENGTH)
	nn_output = loaded_model.predict(test_data)
	print(nn_output)
	i=0
	newsClassifications = {}
	for idx in np.argmax(nn_output, axis=1):
		print("Category: ", index_to_label_dict[idx])
		print("text: " , newsList[i])
		print("=====================================")
		newsClassifications[index_to_label_dict[idx]] = newsList[i]
		i = i + 1
	return newsClassifications


@app.route("/news",	 methods=['GET', 'POST'])
def rec():
	query = '' 
	if(request.method == "POST"):
		print("inside post")
		print(str(request.form.get('news')))
		news = request.form.get('news')
		print(news)
		newsClassifications = getNewsClassification(news)
		return render_template('classifynews.html', news=news, newsClassifications=newsClassifications)
	else:
		return render_template('classifynews.html', news="" ,newsClassifications=None)
	

if __name__ == "__main__":
    app.run(debug=True)


