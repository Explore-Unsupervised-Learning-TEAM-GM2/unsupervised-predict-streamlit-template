"""

    Helper functions for data loading and manipulation.

    Author: Explore Data Science Academy.

"""
# Data handling dependencies
import pandas as pd
import numpy as np

def load_movie_titles(path_to_movies):
    """Load movie titles from database records.

    Parameters
    ----------
    path_to_movies : str
        Relative or absolute path to movie database stored
        in .csv format.

    Returns
    -------
    list[str]
        Movie titles.

    """
    df = pd.read_csv(path_to_movies)
    df = df.dropna()
    movie_list = df['title'].to_list()
    return movie_list

def load_movie_df(path_to_movies):
    """Load movie dataframe from database records.

    Parameters
    ----------
    path_to_movies : str
        Relative or absolute path to movie database stored
        in .csv format.

    Returns
    -------
    Dataframe
        Movie Dataframe

    """
    df = pd.read_csv(path_to_movies)
    df = df.dropna()
    movies = df
    return movies

def load_movie_budget(path_to_movies):
    """Load movie budget from database records.

    Parameters
    ----------
    path_to_movies : str
        Relative or absolute path to movie database stored
        in .csv format.

    Returns
    -------
    list[str]
        Movie budget.

    """
    df = pd.read_csv(path_to_movies)
    df = df.dropna()
    movie_budget = df['budget']
    return movie_budget

def load_movie_runtime(path_to_movies):
    """Load movie runtime from database records.

    Parameters
    ----------
    path_to_movies : str
        Relative or absolute path to movie database stored
        in .csv format.

    Returns
    -------
    list[str]
        Movie runtime.

    """
    df = pd.read_csv(path_to_movies)
    df = df.dropna()
    movie_runtime = df['runtime'].to_list()
    return movie_runtime

def load_movie_ratings(path_to_movies):
    """Load movie rating from database records.

    Parameters
    ----------
    path_to_movies : str
        Relative or absolute path to movie database stored
        in .csv format.

    Returns
    -------
    list[str]
        Movie runtime.

    """
    df = pd.read_csv(path_to_movies)
    df = df.dropna()
    movie_ratings = df[['userId','movieId','rating']].dropna()
    return movie_ratings