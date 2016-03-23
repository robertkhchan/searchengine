'''
Created on Mar 21, 2016

@author: robert
'''
import os
import numpy as np
from scipy import spatial
from nltk.stem.porter import PorterStemmer
from nltk.tokenize import RegexpTokenizer
from TFIDFCalculator import TFIDFCalculator
from stopwords import StopWords

class Driver(object):
    
    stop_words = StopWords()
    
    def __init__(self, data_dir_path, cache_file_path, num_entries):
        self.data_dir_path = data_dir_path
        self.cache_file_path = cache_file_path
        self.num_entries = int(num_entries)
        
        self.stemmer = PorterStemmer()
        self.tokenizer = RegexpTokenizer(r'\w+')
        
        self.doc_words = {}
        self.all_words = None
        self.all_docs = None
        self.tfidf_values = None
        
        if (os.path.isfile(self.cache_file_path)):
            data = np.load(self.cache_file_path)
            self.tfidf_values = data['tfidf_values']
            self.all_docs = data['all_docs'].tolist()
            self.all_words = data['all_words'].tolist()
            self.doc_words = data['doc_words'].tolist()
            print('restored from data file')
            
        else:     
            self.all_words = set()
            self.all_docs = []
            
            for doc_name in os.listdir(self.data_dir_path):
                print('Processing word counts for file: '+doc_name)
                doc_words = self.getWords(self.data_dir_path, doc_name)
                self.doc_words[doc_name] = doc_words
                self.all_words |= set(doc_words)
                self.all_docs.append(doc_name)
            
            self.all_words = list(self.all_words)
            self.tfidf_values = np.zeros((len(self.all_docs),len(self.all_words)))
                    
            calculator = TFIDFCalculator()
            all_doc_words = self.doc_words.values()
            for doc_name in self.all_docs:
                print('Calculating TFIDF for file: '+doc_name)
                doc_index = self.all_docs.index(doc_name)
                doc_words = self.doc_words[doc_name]
                for word in set(doc_words):
                    word_index = self.all_words.index(word)
                    self.tfidf_values[doc_index][word_index] = calculator.getTFIDF(word, doc_words, all_doc_words)  
                        
            np.savez(self.cache_file_path,
                    tfidf_values = self.tfidf_values, 
                    all_docs = self.all_docs, 
                    all_words = self.all_words,
                    doc_words = self.doc_words) 
            
            print('created new data file.')
            
    
    def getWords(self, doc_dir, doc_name):        
        words = []
        with open(doc_dir+doc_name,'r') as file:
            for line in file:
                for word in self.tokenizer.tokenize(line):
                    word = self.stemmer.stem(word.lower())
                    if (word not in Driver.stop_words):
                        words.append(word)
        
        return words
         
        
    def findResults(self, query):
        
        queried_words = []
        for word in self.tokenizer.tokenize(query):
            word = self.stemmer.stem(word.lower())
            if (word not in Driver.stop_words) and (word in self.all_words):
                queried_words.append(word)
                
        if (len(queried_words) > 0):
            queried_vector = [0 for x in range(len(self.all_words))]
        
            for w in queried_words:
                queried_vector[self.all_words.index(w)] = 1
                
            doc_dist = [0 for x in range(len(self.all_docs))]
            for i in range(len(self.all_docs)-1):
                doc_vector = self.tfidf_values[i,:]
                doc_dist[i] = 1 - spatial.distance.cosine(queried_vector, doc_vector)
                
            sortedDocIndex = np.argsort(doc_dist).tolist()
            
            return [self.all_docs[i] for i in sortedDocIndex[::-1][:self.num_entries]]
            
        else:
            return []
        

if __name__ == '__main__':
    
    data_dir_path = input("Please enter directory path to the dataset:")
    cache_file_path = input("Please enter file path to where you want to store/load cached data:")
    num_entries = input("Please enter the number of entries you like to retrieve: ")
    sm = Driver(data_dir_path + "/", cache_file_path, num_entries);
    
    while(True):
        query = input("Please enter your query:")
        if (query):
            results = sm.findResults(query)
            if (len(results) > 0):
                print(results)
            else:
                print("No results match your query.")
        else:
            break