# Determines the sentiment on product reviews (positive or negative)

Ultimately, we use BERT encodings to create embeddings of the product reviews. 
Then, train logistic regression models on these vectors and use a grid search to find the model with the least binary classification error.
