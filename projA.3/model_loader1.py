# This file is suppose to provide an interface between your implementation and the autograder. 
# In reality, the autograder should be a production system. This file provide an interface for 
# the system to call your classifier. 

# Ideally you could bundle your feature extractor and classifier in a single python function, 
# which takes a raw instance (a list of two strings) and predict a probability. 

# Here we use a simpler interface and provide the feature extractor and the classifer separately. 
# For Problem 1, you are supposed to provide
# * a feature extraction function `extract_BoW_features`, and  
# * a sklearn classifier, `classifier1`, whose `predict_proba` will be called.
# * your team name

# These two python objects will be imported by the `test_classifier_before_submission` autograder.

import pickle
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import PCA
import pandas as pd
import string

# TODO: please replace the line below with your implementations. The line below is just an 
# example. 

# Reading the data from our .ipynb file. This is for test use.
with open('data_variable.pickle', 'rb') as f:
    data = pickle.load(f)
    


# Reading our best model from the pickle file
with open('best_model.pickle', 'rb') as f:
    classifier1 = pickle.load(f)


# Reading our vectorizer from the pickle file
with open('vectorizer.pickle', 'rb') as f:
    vectorizer = pickle.load(f)

# Reading our y_train_1d from the pickle file
with open('y_train_1d.pickle', 'rb') as f:
    y_train_1d = pickle.load(f)


#define function that will remove punctuation in dataframe
def remove_punctuations(text):
    for punctuation in string.punctuation:

        #replace any punctuation with the empty string
        text = text.replace(punctuation, '')
    return text


global arr 

def extract_BoW_features(data):
    # vectorizer = CountVectorizer()

    # feat_matrix = vectorizer.transform(sr_data['text'])
    
    # arr = feat_matrix.toarray()


    # print("printing arr shape")
    # print(arr.shape[1])
    #get feature matrix from training data
    cleaned_data = clean_data(data)
    feat_matrix = vectorizer.transform(cleaned_data['text'])

    return feat_matrix



def clean_data(data):
    sr_data = pd.DataFrame(data, columns=['website_name', 'text'])

    # cleaning the data
    sr_data['text'] = sr_data['text'].apply(lambda x: x.lower())
    sr_data['text'] = sr_data['text'].apply(remove_punctuations)

    return sr_data



# TODO: please load your own trained models. Please check train_and_save_classifier.py to find 
# an example of training and saving a classiifer. 

# Define the text representation of the LogisticRegression model
# model_text = "LogisticRegression(max_iter={}, random_state=42, C={}, solver='lbfgs')".format(258, 1)

# classifier1 = LogisticRegression(max_iter=258, random_state=42, C=1, solver='lbfgs')
# arr = extract_BoW_features(data)
# classifier1.fit(arr)

# pca = PCA(n_components=100)
# classifier1 = classifier1.fit_transform(sr_data)




#use clean data function
# cleaned_data = clean_data(data)



# print('Shape of feature matrix')
# print(feat_matrix.shape)
# print('Shape of y train')
# print(y_train_1d.shape)

feat_matrix = extract_BoW_features(data)
classifier1.fit(feat_matrix, y_train_1d)


# TODO: please provide your team name -- 20 chars maximum and no spaces please.  
teamname = "AI-Beast-Mode"
