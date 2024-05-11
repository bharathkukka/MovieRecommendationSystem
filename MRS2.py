import numpy as np
import pandas as pd
# import ast
import nltk
import re
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer
pd.set_option('display.width', 1000)
pd.set_option('display.max_columns', 25)
# here we have written these to display the dataframes in a propper manner
RawMovies = pd.read_csv("/Users/bharathgoud/PycharmProjects/MovieRecomndationSystem/Recommender_System/TMDB dataset/tmdb_5000_movies.csv")
RawCredits = pd.read_csv("/Users/bharathgoud/PycharmProjects/MovieRecomndationSystem/Recommender_System/TMDB dataset/tmdb_5000_credits.csv")
# print(RawMovies.head(1))
# print(RawMovies.shape)
# print(RawCredits.head(1))
# print(RawCredits.shape)
''' here we need to merge these 2 dataframes into a single dataframe , so i will use pd.merge()
we use "on" = common_column'''
MergeDF = pd.merge(RawMovies, RawCredits, on='title')
# print(MergeDF.shape)
# print(MergeDF.head(1))
# print(MergeDF.info())

'''.info() method in pandas is used to provides a concise summary of a DataFrame, 
including its index dtype, column dtypes, non-null values, and memory usage.  '''
''' now here we have 23 columns but in that there are many columns  that was not require to me ,
so i will create a newdf for storing required columns '''

RequiredColumnDF = MergeDF[['movie_id', 'title', 'overview', 'keywords', 'genres', 'cast', 'crew']]

'''The inner set of square brackets ['movie_id', 'title','overview','keywords','genres','cast','crew']
 creates a list of column names we want to select from the DataFrame.
The outer set of square brackets MergeDF[...] takes this list of column names and tells 
pandas to select those columns from the DataFrame MergeDF.'''
# print(RequiredColumnDF.shape)
# print(RequiredColumnDF.head(1))
# print(RequiredColumnDF.overview[1])
null_values = RequiredColumnDF.isnull().sum()
# print(null_values) # here in overview there are 3 null values so lets remove null values in overview column
# we can also handel missing values in many other approaches
RequiredColumnDF = RequiredColumnDF.dropna(subset=['overview']).copy()
'''When inplace=True is specified, the operation is performed on the DataFrame itself,
 and the changes are made in place. This means that the original DataFrame is modified,
  and there is no need to create a new DataFrame to store the result.'''
null_values_after = RequiredColumnDF.isnull().sum()
# print(null_values_after) #  here we have 0 null values
# lets check for the duplicates in the df
duplicates = RequiredColumnDF.duplicated().sum()
# print(duplicates) # no duplicates
genre = RequiredColumnDF.iloc[0].genres
'''iloc is a method in pandas used for integer-location based indexing. 
It is primarily used to select rows and columns by their integer position within the DataFrame.'''
#print(genre) # here we need to change the formate of the genres , it was in this formate [{"id": 28,
# "name": "Action"}, {"id": 12, "name": "Adventure"}, {"id": 14, "name": "Fantasy"}, {"id": 878, "name": "Science
# Fiction"}] we want in this formate ['Action','Adventure','Fantasy','ScienceFiction']

import ast

# "Abstract Syntax Trees" provides functions to parse Python source code
# into abstract syntax trees (ASTs) and to compile ASTs into Python bytecode.
# let's create a function/method that will give a list what we want
def convert(input_string):
    updated = []
    for i in ast.literal_eval(input_string):  #ast.literal_eval(). This converts the string into a Python list.
        updated.append(i['name'])
    return updated

RequiredColumnDF['genres'] = RequiredColumnDF['genres'].apply(convert)
#g2=RequiredColumnDF.iloc[0].genres
#print(g2)
#genres_data = RequiredColumnDF['genres']
#print(genres_data)
# lets apply same for keywords
#keywords_data = g2=RequiredColumnDF.iloc[0].keywords
#print(keywords_data)
RequiredColumnDF["keywords"] = RequiredColumnDF["keywords"].apply(convert)
keywords_data =RequiredColumnDF.iloc[0].keywords
#print(keywords_data) # output after ['culture clash', 'future', 'space war', 'space colony', 'society', 'space travel', 'fu.......]

'''what is the other columns
overview_data = g2=RequiredColumnDF.iloc[0].overview
print(overview_data)
title_data = RequiredColumnDF['title']
print(title_data) '''


# print(RequiredColumnDF.head()) # cast&crew reformate i only consider  1st 3 names in cast & only director name in crew
# print(RequiredColumnDF['cast'][0]) # here we have a lot info about one cast we take only name of the cast
def convert2(input_string_cast):
    updated = []
    counter = 0
    for i in ast.literal_eval(input_string_cast):
        if counter != 3:
            updated.append(i['name'])
            counter += 1
        else:
            break
    return updated

RequiredColumnDF['cast'] = RequiredColumnDF['cast'].apply(convert2)


#print(RequiredColumnDF['cast'][0])
# lets display only director name in crew column
# print(RequiredColumnDF['crew'][0]) # i need only of the crew  director name , i will not consider id, job and everything

def fetch_directorName(input_string_crew):
    Dname = []
    for i in ast.literal_eval(input_string_crew):
        if i['job'] == 'Director':
            Dname.append(i['name'])
            break
    return Dname

RequiredColumnDF['crew'] = RequiredColumnDF['crew'].apply(fetch_directorName)
# print(RequiredColumnDF['crew'][0]) #output 0[James Cameron]
#print(RequiredColumnDF.head())

# overview is in a string formate but we need it in a list formate representing the features as a list of column names
# provides clarity, flexibility, and compatibility, making it a common practice in ML
RequiredColumnDF['overview']=RequiredColumnDF['overview'].apply(lambda x: x.split())
#lambda function that takes a string x as input and splits it into a list of words using the split() method
#By default, the split() method splits the string at whitespace characters and returns a list of substrings
#print(RequiredColumnDF['overview'][0]) #here we have overview in string formate
#print(RequiredColumnDF.head())

#now overview,genres,cast,crew,keywords in list formate ,there is a problem with spaces like science fiction here i need to remove the space between them then i can refer it as a single word
RequiredColumnDF['genres'] = RequiredColumnDF['genres'].apply(lambda x: [i.replace(" ", "") for i in x])
RequiredColumnDF['keywords'] = RequiredColumnDF['keywords'].apply(lambda x: [i.replace(" ", "") for i in x])
RequiredColumnDF['cast'] = RequiredColumnDF['cast'].apply(lambda x: [i.replace(" ", "") for i in x])
RequiredColumnDF['crew'] = RequiredColumnDF['crew'].apply(lambda x: [i.replace(" ", "") for i in x])
RequiredColumnDF['overview'] = RequiredColumnDF['overview'].apply(lambda x: [i.replace(" ", "") for i in x])
#print(RequiredColumnDF.head())
RequiredColumnDF['tags'] = RequiredColumnDF['genres']+RequiredColumnDF['cast']+RequiredColumnDF['crew']+RequiredColumnDF['keywords']+RequiredColumnDF['overview']
# print(RequiredColumnDF.head()) # here we will have 8 columns by adding tags as a new column . now i should create new df that is having movie id, movie name , tags

MoviesDF=RequiredColumnDF[['movie_id','title','tags']]
MoviesDF.loc[:, 'tags'] = MoviesDF['tags'].apply(lambda x: " ".join(x)) # here we converted tags which is in list into string with spaces between words
# we need to convert tags into lower case
MoviesDF.loc[:, 'tags'] = MoviesDF['tags'].apply(lambda x: x.lower())
#print(MoviesDF.head())
# Data Preprocessing is completed  still here
# data exploration & feature engineering

from nltk.stem.porter import PorterStemmer # porterstemmer is steamming algo that will reduce words to their base form
Ps =PorterStemmer()
# now we need to create a function or method that takes input string and give output
# base form the words in the string and print with spaces between base form of words

def stem(input_text):
    basewords =[]
    for i in input_text.split():
        basewords.append(Ps.stem(i))
    return " ".join(basewords)
#print(stem('fucking'))
#MoviesDF['tags'] = MoviesDF['tags'].map(stem)
MoviesDF.loc[: ,'tags']= MoviesDF['tags'].apply(stem)
#print(MoviesDF['tags'][0])
# now we need to check the similarites between the movies. by using the tags columns .now the tags was in the string formate
# it is difficult to check similarity using string , so i converted the string into a vector
#import
#from sklearn.feature_extraction.text import CountVectorizer

#cv =CountVectorizer()  #module to convert text data into a matrix of word counts,
# with a maximum of 5000 features and using the English stop words
cv = CountVectorizer(max_features=5000, stop_words='english')
#English stopwords should be removed during tokenization.
vectors = cv.fit_transform(MoviesDF['tags']).toarray()#trains the vectorizer on the input data and transforms it into a matrix of word counts
#print(vectors)#toarray() converts the sparse matrix representation into a dense NumPy array.
#print(vectors.shape)
#print(vectors[0])
fn = cv.get_feature_names_out()
#print(len(fn))
#print(fn)
# from sklearn.metrics.pairwise import cosine_similarity
similarity = cosine_similarity(vectors)
# print(similarity[0])
# print(similarity.shape) #4806,4806 #cosine similarity between two vectors is close to 1,
# it means that the vectors are very similar , 0 means orthogonal or perpendicular to each other ,-1 means dissimilar
#print(sorted(similarity[0],reverse=True))
# here we need very similar 5 movies so , sorted in descending order
#print(list(enumerate(similarity[0]))) # i need to sort this
#print(sorted(list(enumerate(similarity[0])), reverse= True)) # sorteing is done on the index based but we need on similarity based
#MoviesRecommended = sorted(list(enumerate(similarity[0])), reverse= True,key= lambda x:x[1])[1:6]
#print(MoviesSorted) # we need to extract the name of the movie by the index
#print(MoviesDF[MoviesDF['title']=='Batman Begins'].index[0])
#import re

# Function to recommend movies
'''def RecommendMovies(movie):
    global MoviesDF, similarity
    movie_indices = MoviesDF[MoviesDF['title'] == movie].index
    if len(movie_indices) == 0:
        print("Sorry, the movie '{}' is not in the dataset.".format(movie))
        # Take input for movie ID, title, and tags
        movie_id = int(input("Enter the movie ID: "))
        title = input("Enter title of the movie: ")
        tags = input("Enter the tags for the movie (separated by commas): ")

        # Add the new movie to the dataset
        new_movie = {'movie_id': [movie_id], 'title': [title], 'tags': [tags]}
        new_movie_df = pd.DataFrame(new_movie)
        MoviesDF = pd.concat([MoviesDF, new_movie_df], ignore_index=True)

        # Apply preprocessing steps to the new movie data# new data will be in last row
        MoviesDF.loc[len(MoviesDF) - 1, 'tags'] = MoviesDF.loc[len(MoviesDF) - 1, 'tags'].lower()
        MoviesDF.loc[len(MoviesDF) - 1, 'tags'] = stem(MoviesDF.loc[len(MoviesDF) - 1, 'tags'])

        # Update similarity matrix
        vectors = cv.fit_transform(MoviesDF['tags']).toarray()
        similarity = cosine_similarity(vectors)

        # Get recommendations for the new movie
        movie_index = len(MoviesDF) - 1
        distance = similarity[movie_index]
        movies_list = sorted(list(enumerate(distance)), reverse=True, key=lambda x: x[1])[1:6]
        print("Recommended movies for '{}' (Newly added):".format(title))
        for i in movies_list:
            print(MoviesDF.iloc[i[0]].title)
    else:
        movie_index = movie_indices[0]
        distance = similarity[movie_index]
        movies_list = sorted(list(enumerate(distance)), reverse=True, key=lambda x: x[1])[1:6]
        print("Recommended movies for '{}':".format(movie))
        for i in movies_list:
            print(MoviesDF.iloc[i[0]].title)
print(RecommendMovies('Salaar'))

# Example usage
#RecommendMovies('Salaar')

#RecommendMovies('Salaar')
#gg=MoviesDF[MoviesDF['title']=='Salaar'].index[0]
#print(gg)'''


# Load the Movies dataset

# Preprocess the tags

def AddMovie(movie_id, title, tags, MoviesDF):
    new_movie = {'movie_id': [movie_id], 'title': [title], 'tags': [tags]}
    new_movie_df = pd.DataFrame(new_movie)
    MoviesDF = pd.concat([MoviesDF, new_movie_df], ignore_index=True)
    return MoviesDF

def RecommendMovies(movie, MoviesDF, similarity):
    movie_indices = MoviesDF[MoviesDF['title'] == movie].index
    if len(movie_indices) == 0:
        print("Sorry, the movie '{}' is not in the dataset.")
        movie_id = int(input("Enter the movie ID: "))
        title = input("Enter title of the movie: ")
        tags = input("Enter the tags for the movie (separated by commas): ")

        MoviesDF = AddMovie(movie_id, title, tags, MoviesDF)
        MoviesDF.loc[len(MoviesDF) - 1, 'tags'] = MoviesDF.loc[len(MoviesDF) - 1, 'tags'].lower()
        MoviesDF.loc[len(MoviesDF) - 1, 'tags'] = PorterStemmer().stem(MoviesDF.loc[len(MoviesDF) - 1, 'tags'])

        vectors = cv.fit_transform(MoviesDF['tags']).toarray()
        similarity = cosine_similarity(vectors)

        movie_index = len(MoviesDF) - 1
        distance = similarity[movie_index]
        movies_list = sorted(list(enumerate(distance)), reverse=True, key=lambda x: x[1])[1:6]
        print("Recommended movies for '{}' (Newly added):".format(title))
        for i in movies_list:
            print(MoviesDF.iloc[i[0]].title)
    else:
        movie_index = movie_indices[0]
        distance = similarity[movie_index]
        movies_list = sorted(list(enumerate(distance)), reverse=True, key=lambda x: x[1])[1:6]
        print("Recommended movies for '{}':".format(movie))
        for i in movies_list:
            print(MoviesDF.iloc[i[0]].title)
    return MoviesDF, similarity

# Initialize MoviesDF
MoviesDF = pd.read_csv('MoviesDF.csv')

# Initialize similarity matrix
vectors = cv.fit_transform(MoviesDF['tags']).toarray()
similarity = cosine_similarity(vectors)

MoviesDF, similarity = RecommendMovies('Salaar', MoviesDF, similarity)
MoviesDF.to_csv('MoviesDF.csv', index=False)