"""
End to End 2-factor movie recommendation engine
ALS matrix Factorization + Sentiment Analysis
"""

#!pip install gcsfs
import pandas as pd
import gcsfs
import os
from pyspark import SparkContext
from pyspark.mllib.recommendation import ALS, MatrixFactorizationModel
import math
import re
import numpy as np
import pickle
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import sys
sc = SparkContext()
print('Spark Context successfully created...\n')


#Set working directory in GCP bucket
filepath = 'file:///home/hhv2106/projectmovierecom'


#Load pre-trained ALS model
ALS_model = MatrixFactorizationModel.load(sc,'file:///home/hhv2106/projectmovierecom/complete_matrix_model')
print('Pre-trained ALS_model successfully loaded...\n')


#Now that we have the ALS model loaded, we can load the ratings dataset
ratings_csv = os.path.join(filepath, 'ml-latest', 'ratings.csv')
ratings_raw = sc.textFile(ratings_csv)
ratings_raw_header = ratings_raw.take(1)[0]


#Clean ratings data to remove timestamp and header. Creates a tuple of (user_id, movie_id, rating) for each
ratings = ratings_raw.filter(lambda line: line!=ratings_raw_header).map(lambda line: line.split(",")).map(lambda tokens: (int(tokens[0]),int(tokens[1]),float(tokens[2]))).cache()
print('Movie ratings data successfully cleaned...\n')    
print ("There are %s recommendations in the ratings dataset" % (ratings.count()))
print('\n')


#Display sample of 5 ratings
print('Sample of 5 ratings after cleaning...\n')
ratings.take(5)


#Splitting ratings data RDD into train and test
TrainRDD, TestRDD = ratings.randomSplit([7, 3], seed=0)


#Converting TestRDD into a two-element tuple of (user_id, movie_id) suitable for predicting rating
prediction_testingRDD = TestRDD.map(lambda x: (x[0], x[1]))


#Generating predictions from TestRDD and evaluating model error. Only run if required
"""
predicted_ratings = ALS_model.predictAll(prediction_testingRDD).map(lambda r: ((r[0], r[1]), r[2]))
ratings_and_predictions = TestRDD.map(lambda r: ((int(r[0]), int(r[1])), float(r[2]))).join(predicted_ratings)
error = math.sqrt(ratings_and_predictions.map(lambda r: (r[1][0] - r[1][1])**2).mean())
    
print ("RMSE for testing the model with complete data is %s" % (error))
"""


#Loading the movie names dataset
movies_csv = os.path.join(filepath, 'ml-latest', 'movies.csv')
movies_raw = sc.textFile(movies_csv)
movies_raw_header = movies_raw.take(1)[0]


#Clean movies data to remove header. Creates a tuple of (movie_id, title, genre) for each
movies = movies_raw.filter(lambda line: line!=movies_raw_header).map(lambda line: line.split(",")).map(lambda tokens: (int(tokens[0]),tokens[1],tokens[2])).cache()


#Creates RDD of tuples containing (movie_id, title) for each
movies_titles = movies.map(lambda x: (int(x[0]),x[1]))

print('Movie titles data successfully cleaned...\n')  
print ("Number of movies in movie dataset is: %s" % (movies_titles.count()))
print('\n')


#Counting the number of ratings for a movie. Only movies with a certain number of ratings will be used.
def rating_count_and_avg(ID_and_ratings_tuple):
    nratings = len(ID_and_ratings_tuple[1])
    return ID_and_ratings_tuple[0], (nratings, float(sum(x for x in ID_and_ratings_tuple[1]))/nratings)

#Groupby object for (movie_id, list of all ratings) for each unique movie_id
movieID_ratings_RDD = (ratings.map(lambda x: (x[1], x[2])).groupByKey())

#Converts groupby object above to (movie_id, (n_ratings, avg_rating)) for each unique movie_id
movieID_average_ratings_RDD = movieID_ratings_RDD.map(rating_count_and_avg)

#Converts above into RDD of (movie_id, n_ratings)
ratings_count_RDD = movieID_average_ratings_RDD.map(lambda x: (x[0], x[1][0]))

#Adding new ratings for new user manually. Front end API will accept inputs in finished model

#Run the next two cells only if new user is being input to the model. Else skip next two.
'''
new_user_ID = 0

# The format of each line is (userID, movieID, rating)
new_user_ratings = [
     (0,287,6), # Star Wars (1977)
     (0,186,3), # Toy Story (1995)
     (0,161,9), # Casino (1995)
     (0,2,10), # Leaving Las Vegas (1995)
     (0,324,7), # Twelve Monkeys (a.k.a. 12 Monkeys) (1995)
     (0,345,6), # Flintstones, The (1994)
     (0,39,6), # Timecop (1994)
     (0,296,9), # Pulp Fiction (1994)
     (0,88,10) , # Godfather, The (1972)
     (0,5,9) # Usual Suspects, The (1995)
    ]

new_user_ratings_RDD = sc.parallelize(new_user_ratings)
print ("New user ratings are: %s" % new_user_ratings_RDD.take(10))

#Adding new user ratings to dataframe. Onlu used if adding a new user not existing one
complete_data_with_new_ratings_RDD = ratings.union(new_user_ratings_RDD)

#New user ratings have been added to existing ratings dataframe
'''


#Generating recommendations for the new user. Only run this if re-training model for new user

#Training the ALS matrix model core to identify user preferences
'''
from time import time

t0 = time()
new_ratings_matrix_model = ALS.train(complete_data_with_new_ratings_RDD, 8, seed=5, iterations=10, lambda_=0.1)
tt = time() - t0

print ("New model trained in %s seconds" % round(tt,3))


new_user_ratings_ids = list(map(lambda x: x[1], new_user_ratings))
new_user_unrated_movies_RDD = (movies.filter(lambda x: x[0] not in new_user_ratings_ids).map(lambda x: (new_user_ID, x[0])))
new_user_recommendations_RDD = new_ratings_matrix_model.predictAll(new_user_unrated_movies_RDD)
'''


#User inputs to add ratings to listed movies

new_user_ratings = []
movie_input_list = {'Star Wars (1977)':260, 'Toy Story (1995)':1, 'Casino (1995)':16, 'Leaving Las Vegas (1995)':25, 'Twelve Monkeys (a.k.a. 12 Monkeys) (1995)':32, 'Flintstones, The (1994)':335, 'Timecop (1994)':379, 'Pulp Fiction (1994)':296, 'The Godfather (1972)':858, 'The Usual Suspects (1995)':50}
rating_list = []

print('Input movies and ratings for user_ID 15.')
print('For demonstration purposes, the list of movies to choose from is given below. Please provide your rating (0-5) for each:\n')

for movie in movie_input_list:
    rating = input("Enter rating (0-5) for {}:" .format(movie))
    rating_list.append(rating)

i=0
for movie in movie_input_list:
    new_user_ratings.append((15, movie_input_list[movie], int(rating_list[i])))
    i+=1


#Adding new ratings for existing user manually. Front end API will accept inputs in finished model

new_user_ID = 15

# The format of each line is (userID, movieID, rating)
'''
new_user_ratings = [
     (15,1,8),
     (15,2,9), 
     (15,31,2), 
     (15,46,1), 
     (15,524,9), 
     (15,305,10),
     (15,70,8), 
     (15,96,4), 
     (15,88,1) , 
     (15,512,3) 
    ]
'''
new_user_ratings_RDD = sc.parallelize(new_user_ratings)
print ("New user ratings are: %s" % new_user_ratings_RDD.take(10))

#Adding new user ratings to dataframe. Onlu used if adding a new user not existing one
complete_data_with_new_ratings_RDD = ratings.union(new_user_ratings_RDD)

#Existing user new ratings have been added to existing ratings dataframe
print('\n')
print('Existing user new ratings have been added to existing ratings dataframe...\n')


#Run this cell if using new ratings input for existing user

#Get a list of movie_ids for newly rated movies
new_user_ratings_movie_ids = list(map(lambda x: x[1], new_user_ratings))

#Get an RDD of unrated movies for that user. ###can also exclude movies already rated if using existing user 
new_user_unrated_movies_RDD = (movies.filter(lambda x: x[0] not in new_user_ratings_movie_ids).map(lambda x: (new_user_ID, x[0])))

#Get RDD of recommendations for new user
new_user_recommendations_RDD = ALS_model.predictAll(new_user_unrated_movies_RDD)


# Transform new_user_recommendations_RDD into pairs of the form (Movie ID, Predicted Rating)
new_user_recommendations_rating_RDD = new_user_recommendations_RDD.map(lambda x: (x.product, x.rating))

new_user_recommendations_rating_title_and_count_RDD1 = new_user_recommendations_rating_RDD.join(movies_titles).join(ratings_count_RDD)

new_user_recommendations_rating_title_and_count_RDD = new_user_recommendations_rating_title_and_count_RDD1.map(lambda r: (r[1][0][1], r[1][0][0], r[1][1]))


#Filtering out movies with less than 25 ratings

top_movies = new_user_recommendations_rating_title_and_count_RDD.filter(lambda r: r[2]>=25).takeOrdered(250, key=lambda x: -x[1])

print ('TOP recommended movies based on ratings (with more than 25 reviews):\n%s' % '\n'.join(map(str, top_movies[:25])))


print('Now beginning sentiment analysis on user reviews for top recommended movies...\n')

#Load file for creating links from movie title to imdb_id
database = pd.read_csv('link_database.csv')

#Also create a copy of database to access uncleaned movie name at the end of the list
database_copy = database.copy()


#Some Preprocessing steps for comment database

#Function to clean titles
def clean_title(movie_title):
    movie_title = movie_title.strip().lower()
    movie_title = re.sub('\W+','', movie_title)
    #movie_title = re.sub("\d+", "", movie_title)
    return movie_title

#Function to add zeros to IMDB ID to make it compatible with output of ALS recommended movies
def add_zeros(imdbId):
    string_list = [char for char in str(imdbId)]

    def convert(s): 
        new = "" 
        for x in s: 
            new += x  
        return new 
    
    while len(string_list) < 6:
        string_list.insert(0,'0')
        
    output = convert(string_list)
    return output


#Preprocess database and database _copy
database['movie_id'] = database['movie_id'].apply(lambda x: add_zeros(x))
database['title'] = database['title'].apply(lambda x: clean_title(x))
database_copy['movie_id'] = database_copy['movie_id'].apply(lambda x: add_zeros(x))

#Preprocess listof recommended movies from ALS
list_recommended_movies = [x[0] for x in top_movies]
list_recommended_movies = list(map(lambda x: clean_title(x), list_recommended_movies))

#print("ALS recommended top movies are:\n", list_recommended_movies)

NLP_input = []

for movie in list_recommended_movies:
    if movie in list(database['title']):
        NLP_input.append(database.loc[database['title'] == movie, 'movie_id'].values[0])

print('Movie IDs input to NLP for whom user comments exist are: ', NLP_input)


#Import comment files and store them in one dataframe
df_train = pd.read_csv('train_df_full.csv')
df_test = pd.read_csv('test_df_full.csv')
df_full = pd.concat([df_train, df_test], axis=0)
df_full['movie_id'] = df_full['movie_id'].apply(lambda x: x[3:])


#Text Preprocessing Functions
REPLACE_NO_SPACE = re.compile("(\.)|(\;)|(\:)|(\!)|(\')|(\?)|(\,)|(\")|(\()|(\))|(\[)|(\])|(\d+)")
REPLACE_WITH_SPACE = re.compile("(<br\s*/><br\s*/>)|(\-)|(\/)")
NO_SPACE = ""
SPACE = " "

def preprocess_reviews(reviews):
    
    reviews = [REPLACE_NO_SPACE.sub(NO_SPACE, line.lower()) for line in reviews]
    reviews = [REPLACE_WITH_SPACE.sub(SPACE, line) for line in reviews]
    
    return reviews

def preprocess_model_reviews(review):
    
    review = REPLACE_NO_SPACE.sub(NO_SPACE, review.lower())
    review = REPLACE_WITH_SPACE.sub(SPACE, review)
    
    return review


#Build CountVectorizer and Train Logistic Regression Model
reviews_train = list(df_train['content']) 
reviews_test = list(df_test['content'])
    
reviews_train_clean = preprocess_reviews(reviews_train)
reviews_test_clean = preprocess_reviews(reviews_test)

cv = CountVectorizer(binary=True)
cv.fit(reviews_train_clean)

X = cv.transform(reviews_train_clean)
X_test = cv.transform(reviews_test_clean)

target = [1 if i < 12500 else 0 for i in range(25000)]

X_train, X_val, y_train, y_val = train_test_split(X, target, train_size = 0.80)

print('Training logistic regression model for sentiment analysis of users comments...')
train_model = LogisticRegression(C=0.05)
train_model.fit(X_train, y_train)
print ("Validation Accuracy of model is: %s" % (accuracy_score(y_val, train_model.predict(X_val))))

model = LogisticRegression(C=0.05)
model.fit(X, target)
print ("Test Accuracy of model is: %s" % accuracy_score(target, model.predict(X_test)))


#Function to calculate sentiment score for a given movie from all its comments
def get_sentiment_score(movie_id):
    review_list = list(df_full.loc[df_full['movie_id']==str(movie_id), 'content'].values)
    
    review_list_clean = []
    for review in review_list:
        try:
            review_list_clean.append(preprocess_model_reviews(review))
        except:
            print('No reviews were found, outputting zero sentiment score')
            review_list_clean.append(np.zeros(shape=(1, 90860)))
        
    review_array = cv.transform(review_list_clean)
    
    predictions = model.predict(review_array)
    
    try:
        sentiment_score = np.count_nonzero(predictions)/np.count_nonzero(predictions==0)
    except:
        print('There were no negative reviews, returning number of positive reviews!')
        sentiment_score = np.count_nonzero(predictions)
    
    print('The Sentiment Score is: ', sentiment_score)
    
    return (movie_id, sentiment_score)


#Reorder recommended movies by sentiment score
def reorder_recommended_movies(recommended_movie_id_list):
    sentiment_ranking_list = []
    
    for movie_id in recommended_movie_id_list:
        sentiment_ranking_list.append(get_sentiment_score(movie_id))
    
    sentiment_ranking_list.sort(key = lambda x: x[1], reverse=True)
    
    return sentiment_ranking_list


#Sample run
#recommended = ['100680', '453418', '177606', '074223', '042042']
#reorder_recommended_movies(recommended)


#Get list of ordered recommended movies from ALS recommended movies passed into NLP
highest_sentiment_score_list = reorder_recommended_movies(NLP_input)
print(highest_sentiment_score_list)


#Finally retrieve title names for movies with highest sentiment score
final_sentiment_ranked_list = []
for tup in highest_sentiment_score_list:
    final_sentiment_ranked_list.append(database_copy.loc[database_copy['movie_id']==str(tup[0]), 'title'].values[0])

print('\n\n')
print('The list of movies ranked by highest sentiment score is:\n')
for index, title in enumerate(final_sentiment_ranked_list):
    print(index+1, title)


###END###













