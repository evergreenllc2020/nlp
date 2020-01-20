from flask import Flask, request, render_template
import os

# setting up template directory
TEMPLATE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'templates')

app = Flask(__name__, template_folder=TEMPLATE_DIR, static_folder=TEMPLATE_DIR)
#app = Flask(__name__)

import pandas as pd
import numpy as np
from numpy import int64

import requests
import IPython.display as Disp

import sklearn
from sklearn.decomposition import TruncatedSVD





@app.route("/")
def hello():
	return TEMPLATE_DIR



print("building recommendation engine")
print("reading data")
books_df = pd.read_csv("./dataset/books.csv")
ratings_df = pd.read_csv("./dataset/ratings.csv", encoding='UTF-8',  dtype={'user_id': int,'book_id':int, 'rating':int} )
books_df_2 = books_df[['book_id', 'books_count', 'original_publication_year', 'average_rating','original_title','image_url','authors']]
combined_books_df = pd.merge(ratings_df, books_df, on='book_id')
print("creating pivot table")
ct_df = combined_books_df.pivot_table(values='rating', index='user_id', columns='original_title', fill_value=0)
X = ct_df.values.T
print("Creating SVD")
SVD  = TruncatedSVD(n_components=20, random_state=17)
result_matrix = SVD.fit_transform(X)
print("building correlation")
corr_mat = np.corrcoef(result_matrix)
book_names = ct_df.columns
book_list = list(book_names)
isInitialized = True
print(book_list.index("The Hunger Games"))
print("done building recommendation engine")
print("ready for recommendation engine")
#hunger_game_index = book_list.index('The Hunger Games')
#corr_hunger_games = corr_mat[hunger_game_index]
#list(book_names[(corr_hunger_games<1.0) & (corr_hunger_games>0.8)])


def getRecommendations(bookName):
	book_name_index = book_list.index(bookName)
	corr_book = corr_mat[book_name_index]
	recList = list(book_names[(corr_book<1.0) & (corr_book>0.9)])
	max=5
	if(len(recList)<5):
		max=len(recList)
	return books_df_2[books_df_2.original_title.isin(recList)]



@app.route("/rec",	 methods=['GET', 'POST'])
def rec():
	query = '' 
	if(request.method == "POST"):
		print("inside post")
		print(str(request.form.get('query')))
		query = request.form.get('query')
		#print("the book name is " + query)
		recommendations = getRecommendations(query)
		#print(query)
		return render_template('rec.html', query=query, recommendations=recommendations.to_html())
	else:
		return render_template('rec.html', query="" ,recommendations="<<unknown>>")
	

if __name__ == "__main__":
    app.run(debug=True)


