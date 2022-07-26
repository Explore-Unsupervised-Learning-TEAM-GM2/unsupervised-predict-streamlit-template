"""

    Collaborative-based filtering for item recommendation.

    Author: Explore Data Science Academy.

    Note:
    ---------------------------------------------------------------------
    Please follow the instructions provided within the README.md file
    located within the root of this repository for guidance on how to use
    this script correctly.

    NB: You are required to extend this baseline algorithm to enable more
    efficient and accurate computation of recommendations.

    !! You must not change the name and signature (arguments) of the
    prediction function, `collab_model` !!

    You must however change its contents (i.e. add your own collaborative
    filtering algorithm), as well as altering/adding any other functions
    as part of your improvement.

    ---------------------------------------------------------------------

    Description: Provided within this file is a baseline collaborative
    filtering algorithm for rating predictions on Movie data.

"""
# Streamlit dependencies
from sqlalchemy import true
import streamlit as st

# Script dependencies
#import surprise
import pandas as pd
import numpy as np
import pickle
import copy
from surprise import Reader, Dataset
from surprise import SVD, NormalPredictor, BaselineOnly, KNNBasic, NMF
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer
#from sympy import false

# Importing data
movies_df = pd.read_csv('resources/data/movies.csv',sep = ',')
ratings_df = pd.read_csv('resources/data/ratings.csv')
ratings_df.drop(['timestamp'], axis=1,inplace=True)
ratings_df = ratings_df

# We make use of an SVD model trained on a subset of the MovieLens 10k dataset.
model=pickle.load(open('resources/models/SVD.pkl', 'rb'))

def prediction_item(item_id):
    """Map a given favourite movie to users within the
       MovieLens dataset with the same preference.

    Parameters
    ----------
    item_id : int
        A MovieLens Movie ID.

    Returns
    -------
    list
        User IDs of users with similar high ratings for the given movie.

    """
    # Data preprosessing
    reader = Reader(rating_scale=(0, 5))
    load_df = Dataset.load_from_df(ratings_df,reader)
    a_train = load_df.build_full_trainset()

    predictions = []
    for ui in a_train.all_users():
        predictions.append(model.predict(iid=item_id,uid=ui, verbose = False))
    return predictions

def pred_movies(movie_list):
    """Maps the given favourite movies selected within the app to corresponding
    users within the MovieLens dataset.

    Parameters
    ----------
    movie_list : list
        Three favourite movies selected by the app user.

    Returns
    -------
    list
        User-ID's of users with similar high ratings for each movie.

    """
    # Store the id of users
    id_store=[]
    # For each movie selected by a user of the app,
    # predict a corresponding user within the dataset with the highest rating
    for i in movie_list:
        predictions = prediction_item(item_id = i)
        predictions.sort(key=lambda x: x.est, reverse=True)
        # Take the top 10 user id's from each movie with highest rankings
        for pred in predictions[:10]:
            id_store.append(pred.uid)
    # Return a list of user id's
    return id_store

# !! DO NOT CHANGE THIS FUNCTION SIGNATURE !!
# You are, however, encouraged to change its content.  
def collab_model(movie_list,top_n=10):
    """Performs Collaborative filtering based upon a list of movies supplied
       by the app user.

    Parameters
    ----------
    movie_list : list (str)
        Favorite movies chosen by the app user.
    top_n : type
        Number of top recommendations to return to the user.

    Returns
    -------
    list (str)
        Titles of the top-n movie recommendations to the user.

    """
    import scipy as sp

  
    movie_ids = pred_movies(movie_list)
    df_init_users = ratings_df[ratings_df['userId']==movie_ids[0]]
    
    for i in movie_ids:
        df_init_users=df_init_users.append(ratings_df[ratings_df['userId']==i])
    
    

    print(f'number of ratings: {df_init_users.shape[0]}')
    # create the predictions
    pred_series= []
    for movie_id, name, user_id in zip(movie_ids, movies_df['title'],df_init_users['userId']):
        rating_real =ratings_df.query(f'movieId == {movie_id}')['rating'].values[0] if movie_id in df_init_users['movieId'].values else 0
    # generate the prediction
    # generate the prediction
        rating_pred = model.predict(movie_id,user_id, verbose=False)
    # add the prediction to the list of predictions
        pred_series.append([name, rating_pred.est, rating_real])
        predicted = pd.DataFrame(pred_series, columns = ['title','rating', 'actual_rating']) #actual rating used for checking previous ratings for movie id
        predicted = predicted.sort_values(by = ['rating'], ascending = False)
  

    #title list
    title_list = predicted['title'].tolist()
    recommended_movies = []

    for i in title_list[:top_n]:
        recommended_movies.append(i)
    return recommended_movies
 