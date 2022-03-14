from re import M
import numpy as np
import tensorflow as tf
from tensorflow import keras
import nltk
#nltk.download('wordnet')
#nltk.download('omw-1.4')
from nltk.stem import WordNetLemmatizer 
from typing import List

word2idx = keras.datasets.imdb.get_word_index(path="imdb_word_index.json")
# Init the Wordnet Lemmatizer
lemmatizer = WordNetLemmatizer()

def lemmatize_sentence(sentence: List[str])->List[str]:
    # Lemmatize list of words and join
    lemmatized_output = [lemmatizer.lemmatize(w) for w in sentence]
    return lemmatized_output

def word_to_index(sentence: List[str])-> List[int]:
    word2idx_output = [word2idx[word] for word in sentence if word in word2idx]
    return word2idx_output

def preprocessing(document: str, to_index: bool = True, lemm: bool = True)->List[str]:
    # Tokenize: Split the sentence into words
    token_document = [nltk.word_tokenize(sentence) for sentence in document.split(".")]
    
    if lemm: token_document = [lemmatize_sentence(sentence) for sentence in token_document]
    
    if to_index: token_document = [word_to_index(sentence) for sentence in token_document]
    
    return token_document    



def Jaccard_Similarity(doc1: str, doc2: str)->float: 
    
    # List the unique words in a document
    words_doc1 = set(doc1.lower()) 
    words_doc2 = set(doc2.lower())
    
    # Find the intersection of words list of doc1 & doc2
    intersection = words_doc1.intersection(words_doc2)

    # Find the union of words list of doc1 & doc2
    union = words_doc1.union(words_doc2)
        
    # Calculate Jaccard similarity score 
    # using length of intersection set divided by length of union set
    return float(len(intersection)) / len(union)

def cosine_distance(script: str, generate: str)-> float:
    maxlen = 200
    
    preprocess_script = preprocessing(script)
    preprocess_generate = preprocessing(generate)
    #preprocess_script = keras.preprocessing.sequence.pad_sequences(preprocess_script, maxlen=maxlen)
    #preprocess_generate = keras.preprocessing.sequence.pad_sequences(preprocess_generate, maxlen=maxlen)
    
    cosine_loss = keras.losses.CosineSimilarity(axis=1, reduction=keras.losses.Reduction.SUM)
    print(preprocess_generate)
    cosine = cosine_loss(preprocess_script, preprocess_generate).numpy()
    return cosine
    
doc_1 = "Data is the new oil of the digital economy. tested and approuved !"
doc_2 = "Data is a new oil. but not my dope !"


print(Jaccard_Similarity(doc_1,doc_2))
print(cosine_distance(doc_1,doc_2))
