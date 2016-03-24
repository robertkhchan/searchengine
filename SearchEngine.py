'''
Created on Mar 23, 2016

@author: robert
'''

import os
import numpy as np
from scipy import spatial
from nltk.stem.porter import PorterStemmer
from nltk.tokenize import RegexpTokenizer
from TFIDFCalculator import TFIDFCalculator
from stopwords import StopWords

class SearchEngine(object):
    '''Search Engine class responsible for maintaining TF-IDF values of trained corpus
       and knowing how to return most relevant documents based on a query string 
    '''
    
    def __init__(self):        
        self.doc_words = None
        self.all_words = None
        self.all_docs = None
        self.tfidf_values = None
        self.stemmer = PorterStemmer()
        self.tokenizer = RegexpTokenizer(r'\w+')
        self.stop_words = StopWords()
        self.num_results = 10
    
    
    def setNumResults(self, num_results):
        self.num_results = num_results


    def initFromCache(self, cache_file_path):
        '''Initialize Search Engine from cached data file.
        
        Arguments:
            cache_file_path (str): full path to cached data file
            
        '''
        data = np.load(cache_file_path)
        self.doc_words = data['doc_words'].tolist()
        self.all_words = data['all_words'].tolist()
        self.all_docs = data['all_docs'].tolist()
        self.tfidf_values = data['tfidf_values']
        
        print('Search engine initialized from cached data file')
    
    
    def initFromDataset(self, data_dir_path, cache_file_path):
        '''Initialize Search Engine by calculating TF-IDF scores for input dataset,
           then save resulting artifacts as a single cached data file.
           
        Arguments:
            data_dir_path (str): full path to directory containing dataset
            cached_file_path (str): full path to resulting cached data file
            
        '''

        # Read in all words from all documents
        self.doc_words = {}
        self.all_words = set()
        self.all_docs = []
        for doc_name in os.listdir(data_dir_path):
            print('Processing word counts for file: '+doc_name)
            doc_words = self.getWordsFromFile(os.path.join(data_dir_path, doc_name))
            self.all_docs.append(doc_name)
            self.all_words |= set(doc_words)
            self.doc_words[self.all_docs.index(doc_name)] = doc_words

        self.all_words = list(self.all_words)
        
        # Calculate TF-IDF for each word for each document        
        self.tfidf_values = np.zeros((len(self.all_docs),len(self.all_words)))
        calculator = TFIDFCalculator()
        all_doc_words = self.doc_words.values()
        for doc_name in self.all_docs:
            print('Calculating TFIDF for file: '+doc_name)
            doc_index = self.all_docs.index(doc_name)
            doc_words = self.doc_words[doc_index]
            for word in set(doc_words):
                word_index = self.all_words.index(word)
                self.tfidf_values[doc_index][word_index] = calculator.getTFIDF(word, doc_words, all_doc_words)  
                    
        # Save artifacts as single cached data file
        np.savez(cache_file_path,
                tfidf_values = self.tfidf_values, 
                all_docs = self.all_docs, 
                all_words = self.all_words,
                doc_words = self.doc_words) 
        
        print('Search engine initialized from dataset.')
    
        
    def getWordsFromFile(self, file_path):
        '''Get list of words from input file that have been stemmed and are not stop words. 
        
        Return:
            words (list): list of non stop words
            
        '''  
        words = []
        with open(file_path,'r') as file:
            for line in file:
                for word in self.tokenizer.tokenize(line):
                    word = self.stemmer.stem(word.lower())
                    if (word not in self.stop_words):
                        words.append(word)
        
        return words
         
        
    def findResults(self, query):
        '''Find results from trained corpus that are most relevant to query 
           based on cosine similarity
           
        Arguments:
            query (str): user input query
            
        Return:
            results (list): most relevant results
            
        '''
        results = []
        
        # Parse query string for words that exist in trained corpus
        queried_words = []
        for word in self.tokenizer.tokenize(query):
            word = self.stemmer.stem(word.lower())
            if (word not in self.stop_words) and (word in self.all_words):
                queried_words.append(word)
                
        if (len(queried_words) > 0):
            # Calculate similarity measure between query string and all documents
            queried_vector = [1 if word in queried_words else 0 for word in self.all_words]
                
            doc_sim = np.zeros(len(self.all_docs))
            for i in range(len(self.all_docs)-1):
                doc_vector = self.tfidf_values[i,:]
                doc_sim[i] = 1 - spatial.distance.cosine(queried_vector, doc_vector)
                
            # Sort documents from lease to most relevant
            sortedDocIndex = np.argsort(doc_sim).tolist()
            
            # Return the last num_results from the sorted list
            results = [self.all_docs[i] for i in sortedDocIndex[::-1][:self.num_results]]
        
        return results
    
    