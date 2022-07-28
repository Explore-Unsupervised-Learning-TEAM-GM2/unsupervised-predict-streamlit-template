"""

    Streamlit webserver-based Recommender Engine.

    Author: Explore Data Science Academy.

    Note:
    ---------------------------------------------------------------------
    Please follow the instructions provided within the README.md file
    located within the root of this repository for guidance on how to use
    this script correctly.

    NB: !! Do not remove/modify the code delimited by dashes !!

    This application is intended to be partly marked in an automated manner.
    Altering delimited code may result in a mark of 0.
    ---------------------------------------------------------------------

    Description: This file is used to launch a minimal streamlit web
	application. You are expected to extend certain aspects of this script
    and its dependencies as part of your predict project.

	For further help with the Streamlit framework, see:

	https://docs.streamlit.io/en/latest/

"""
# Streamlit dependencies
from turtle import title
import streamlit as st
from PIL import Image

# Data handling dependencies
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import re
import matplotlib as mpl
import sympy
from sympy import continued_fraction_reduce
mpl.rcParams['figure.dpi'] = 130


# Custom Libraries
from utils.data_loader import load_movie_titles
from utils.data_loader import load_movie_budget
from utils.data_loader import load_movie_runtime
from utils.data_loader import load_movie_ratings
from utils.data_loader import load_movie_df
from recommenders.collaborative_based import collab_model
from recommenders.content_based import content_model

# Data Loading
movies = load_movie_df('resources/data/movies.csv')
title_list = movies['title']
movies = movies.dropna() 
ratings = load_movie_ratings('resources/data/ratings.csv')

imdb_data_budget = load_movie_budget('resources/data/imdb_data.csv')
imdb_data_runtime = load_movie_runtime('resources/data/imdb_data.csv')


# App declaration
def main():

    # DO NOT REMOVE the 'Recommender System' option below, however,
    # you are welcome to add more options to enrich your app.
    background = Image.open('resources/imgs/wise.png')
    col1, col2, col3 = st.columns([2, 3, 2])
    col2.image(background, use_column_width=True)
    
    page_options = ["Recommender System","Solution Overview","Movie Statistics", "About Us"]   

    # -------------------------------------------------------------------
    # ----------- !! THIS CODE MUST NOT BE ALTERED !! -------------------
    # -------------------------------------------------------------------
    page_selection = st.sidebar.selectbox("Choose Option", page_options)
    if page_selection == "Recommender System":
        # Header contents
        st.write('# Movie Recommender Engine')
        st.write('### EXPLORE Data Science Academy Unsupervised Predict')
        st.image('resources/imgs/Image_header.png',use_column_width=True)
        # Recommender System algorithm selection
        sys = st.radio("Select an algorithm",
                       ('Content Based Filtering',
                        'Collaborative Based Filtering'))

        # User-based preferences
        st.write('### Enter Your Three Favorite Movies')
        movie_1 = st.selectbox('Fisrt Option',title_list[14930:15200])
        movie_2 = st.selectbox('Second Option',title_list[25055:25255])
        movie_3 = st.selectbox('Third Option',title_list[21100:21200])
        fav_movies = [movie_1,movie_2,movie_3]
       

        # Perform top-10 movie recommendation generation
        if sys == 'Content Based Filtering':
            if st.button("Recommend"):
                try:
                    with st.spinner('Crunching the numbers...'):
                        top_recommendations = content_model(movie_list=fav_movies,
                                                            top_n=10)
                    st.title("We think you'll like:")
                    for i,j in enumerate(top_recommendations):
                        st.subheader(str(i+1)+'. '+j)
                except:
                    st.error("Oops! Looks like this algorithm does't work.\
                              We'll need to fix it!")


        if sys == 'Collaborative Based Filtering':
            if st.button("Recommend"):
                try:
                    with st.spinner('Crunching the numbers...'):
                        top_recommendations = collab_model(movie_list=fav_movies,
                                                           top_n=10)
                    st.title("We think you'll like:")
                    for i,j in enumerate(top_recommendations):
                        st.subheader(str(i+1)+'. '+j)
                except:
                    st.error("Oops! Looks like this algorithm does't work.\
                              We'll need to fix it!")
               

    # -------------------------------------------------------------------

    # ------------- SAFE FOR ALTERING/EXTENSION -------------------
    if page_selection == "Solution Overview":
        st.title("Solution Overview")
        st.write('Recommender systems help users select similar items when engaging with onine services \n'
                'These systems would provide suggestions that might interest the user leading to service improvement and customers satisfaction. \n'
                'The suggestion method is based on content and collaborative filtering approach, that captures correlation between user preferences and item features.n')
        
        st.title('Content Based Filtering')
        st.write("- Content-based filtering depends on similarities between the attributes of the items.\n"
                "- It recommends products to customers based on those that other customers has previously rated as the highest.\n"
                "- Each rated item has a profile.\n"
                "- The algorithm compares item features and pick the items with the highest score.\n"
                "- The algorithm then recommends an item that is the most similarities to the given object.\n"
                "- This algorithm does not consider user preferences, it simply considers item features.")

        st.title('Collaborative Based Filtering')
        st.write("- Collaborative filtering is based on how other users have reacted to the same items.\n"
                " - It is dependent on user preferences rather than the features of the item.\n" 
                "- The algorithm uses survey data from similar users.\n"
                "- The dat that has been colleced is in a table alongside the user's ratings.\n"
                "- The algorithm makes a  prediction based on what similar users liked.\n"
                "- The main drawback to this algorithm is that you need data before you can offer recommendations.\n"
                "- The alogorithm suffers from cold start, where it will make erroneous predictions or repeatedly predicting the same items if there is no data available.")


    #-----------------------------------------------------------------------------------------------------------------------------------------------------------
    if page_selection == "About Us":
        #About us page
        st.title('Wisetech (Pty) Ltd')

        st.write('In today’s technology driven world, recommender systems \
            are socially and economically critical to ensure that individuals can make optimised choices \
                surrounding the content they engage with on a daily basis. One application where this is especially \
                    true is movie recommendations; where intelligent algorithms can help viewers find great titles from tens of thousands of options.Wise was built with the latest AI technology by our team of problem solvers and innovate engineers,with the purpose of assisting streamers to find the latest hpyed content on the net without pondering too much on what binge next.')       

        st.title("Our Mission:")
        st.write("To build cutting edge technology that brings convinience to everyday life.")

        st.title("Our Vision")
        st.write("Develop models that makes accurate predictions with the use of past data.")

        

        #Team image
        team = Image.open('resources/imgs/wiseteam.png')
      
        st.image(team, width=700)
        
        

        #Contact            
        st.title('Contact Us')
        st.subheader('Head Office')
        st.markdown('Adress')
        st.markdown('South Africa')
        st.markdown('123 smith st, Johannesburg, 2100')
        st.markdown ('Telephone')
        st.markdown('(+27)11 940 7892')

        st.markdown ('E-mail')
        st.markdown ('sales@wisetech.com')
        st.markdown('support@wisetech.com')
        

    #---------------------------------------------------------------------------------------------------------------------------------------------------------
    if page_selection == "Movie Statistics":
        
        st.markdown("<h1 style = 'text-align: center;'>Release Year</h1>", unsafe_allow_html=True)

        import plotly.express as px
        from plotly.graph_objs import Layout
        #creating a new column for year
        
        movies['year'] = [x[-1].strip('()') for x in movies.title.str.split(" ")]
        
        num_pattern = r'^$|[a-zA-Z]|Τσιτσάνης|101次求婚|2006–2007|выбывание|پدر|Начальник|Джа|Девочки|первого'
        movies["year"] = movies["year"].replace(to_replace = num_pattern, value = np.nan, regex = True)
        year = [int(x) for x in movies["year"].dropna()]
        year = pd.DataFrame(year, columns = ["Year"])

        #Ploting year data
        fig = px.histogram(year, x = 'Year', histnorm='percent')
        fig.update_layout(title= 'Release Year',title_x=0.5)
        fig.update_layout(plot_bgcolor='pink')
        fig.update_layout(xaxis_range =[1900,2022])
        st.plotly_chart(fig, use_container_width=True)
        st.write(f'Our model was trained on few 90s movie classics and a significant number 21st century movies')
        

        #movie ratings distribution Plot
        st.markdown("<h1 style = 'text-align: center;'>Movie Rating</h1>", unsafe_allow_html=True)
        fig = px.box(ratings, x = 'rating')
        fig.update_layout(title= 'Movie Ratings',title_x=0.5)
        fig.update_layout(plot_bgcolor='pink')
        st.plotly_chart(fig, use_container_width=True)
        st.write(f'Average rating  database: {round(np.mean(ratings["rating"]),2)} with 75% of the rating greater than 3.')
        
        #movie runtime Distribution 
        st.markdown("<h1 style = 'text-align: center;'>Movie Runtime</h1>", unsafe_allow_html=True)
        runtime = pd.DataFrame(imdb_data_runtime, columns=['runtime (min)'])
        fig = px.histogram(runtime, x = 'runtime (min)', histnorm='percent')
        fig.update_layout(title= 'Runtime',title_x=0.5)
        fig.update_layout(plot_bgcolor='pink')
        fig.update_layout(xaxis_range =[0,250])
        st.plotly_chart(fig, use_container_width=True)
        st.write(f'Average rating  database: {round(np.mean(imdb_data_runtime),2)}min with a symetrical distribution')
    #---------------------------------------------------------------------------------------------------------------------------------------------


    # You may want to add more sections here for aspects such as an EDA,
    # or to provide your business pitch.

if __name__ == '__main__':
    main()
