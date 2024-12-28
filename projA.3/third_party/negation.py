'''This document contains a function we found online that is meant to 
turn negations of words into single, synonymous words. An example of this 
would be if the sentence contained the words 'not kind', this function 
could turn them into 'mean'.

We found this function from Utkarsh Lal's article on towardsdatascience.com.
The article link is here:
https://towardsdatascience.com/increasing-accuracy-of-sentiment-classification-using-negation-handling-9ed6dca91f53
'''


from nltk.corpus import wordnet

def Negation(sentence):	
    '''
    Input: Tokenized sentence (List of words)
    Output: Tokenized sentence with negation handled (List of words)
    '''
    sentence = sentence.split()
    temp = int(0)
    for i in range(len(sentence)):
        if sentence[i-1] in ['not',"n't"]:
            antonyms = []
            for syn in wordnet.synsets(sentence[i]):
                syns = wordnet.synsets(sentence[i])
                w1 = syns[0].name()
                temp = 0
                for l in syn.lemmas():
                    if l.antonyms():
                        antonyms.append(l.antonyms()[0].name())
                max_dissimilarity = 0
                for ant in antonyms:
                    syns = wordnet.synsets(ant)
                    w2 = syns[0].name()
                    syns = wordnet.synsets(sentence[i])
                    w1 = syns[0].name()
                    word1 = wordnet.synset(w1)
                    word2 = wordnet.synset(w2)
                    if isinstance(word1.wup_similarity(word2), float) or isinstance(word1.wup_similarity(word2), int):
                        temp = 1 - word1.wup_similarity(word2)
                    if temp>max_dissimilarity:
                        max_dissimilarity = temp
                        antonym_max = ant
                        sentence[i] = antonym_max
                        sentence[i-1] = ''
    while '' in sentence:
        sentence.remove('')
    
    sentence = ' '.join(sentence)
    return sentence
