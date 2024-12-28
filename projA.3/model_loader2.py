# This file is suppose to provide an interface between your implementation and the autograder. 
# In reality, the autograder should be a production system. This file provide an interface for 
# the system to call your classifier. 

# Ideally you could bundle your feature extractor and classifier in a single python function, 
# which takes a raw instance (a list of two strings) and predict a probability. 

# Here we use a simpler interface and provide the feature extractor and the classifer separately. 
# For Problem 2, you are supposed to provide
# * a feature extraction function `extract_awesome_features`, and  
# * a sklearn classifier, `classifier2`, whose `predict_proba` will be called.
# * your team name

# These two python objects will be imported by the `test_classifier_before_submission` autograder.

import pickle
import sklearn
# from sklearn.linear_model import LogisticRegression
# from sklearn.feature_extraction.text import CountVectorizer
# from sklearn.decomposition import PCA
import pandas as pd
# from sklearn.ensemble import RandomForestClassifier




import subprocess
import sys

def install(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])

# Installing libraries
install("contractions")
install("nltk")
install("transformers")
install("torch")


import contractions
import nltk
nltk.download('wordnet')
nltk.download('all')
from nltk.corpus import wordnet
from transformers import BertTokenizer, BertModel
import torch
import numpy as np



print("MODEL LOADER 2 VERSION")
print(sklearn.__version__)

#read in data from pickle
with open('data_variable.pickle', 'rb') as f:
    data = pickle.load(f)

# Reading our best model from the pickle file
with open('best_model_2.pickle', 'rb') as f:
    classifier2 = pickle.load(f)


# Reading our vectorizer from the pickle file
with open('vectorizer.pickle', 'rb') as f:
    vectorizer2 = pickle.load(f)



# Initialize tokenizer and model
global tokenizer, model
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')



def get_bert_embeddings(text):
    # Tokenize input text and get corresponding IDs
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
    # Generate embeddings for each token
    with torch.no_grad():
        outputs = model(**inputs)
    # Extract embeddings and perform mean pooling
    embeddings = outputs.last_hidden_state.mean(dim=1)
    return embeddings.cpu().numpy()

# Extracting final features
def extract_awesome_features(data):
    # Concatenate each pair of strings and then generate embeddings
    concatenated_data = [" ".join(pair) for pair in data]
    embeddings = np.vstack([get_bert_embeddings(text) for text in concatenated_data])
    return embeddings


# TODO: please provide your team name -- 20 chars maximum and no spaces please.  
teamname = "AI-Beast-Mode"


