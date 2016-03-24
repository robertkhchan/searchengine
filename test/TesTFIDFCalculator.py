'''
Created on Mar 22, 2016

@author: robert
'''
from unittest import TestCase
from TFIDFCalculator import TFIDFCalculator
from math import log


class TestTFIDFCalculator(TestCase):
    
    def testGetTF(self):
        allTermsInDocument = ["hello","world"]
        term = "hello"
        
        termInDocCount = 1.0
        allTermsInDocCount = 2.0
        expectedTF = termInDocCount / allTermsInDocCount 
        
        mytfidf = TFIDFCalculator()
        actualTF = mytfidf.getTF(term, allTermsInDocument)
        
        self.assertEqual(expectedTF, actualTF)
        

    def testGetIDF(self):
        allDocuments = [["hello","world"],["my","name","Robert"],["hello","who","you"]]
        term = "hello"

        numDocumentsWithThisTerm = 2.0
        numDocuments = 3.0
        expectedIDF = 1.0 + log(numDocuments / numDocumentsWithThisTerm)
        
        mytfidf = TFIDFCalculator()
        actualIDF = mytfidf.getIDF(term, allDocuments)
        
        self.assertEquals(expectedIDF, actualIDF)