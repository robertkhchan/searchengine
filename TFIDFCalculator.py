from math import log

class TFIDFCalculator:
    
    def __init__(self):
        self.term_counts = {}  
    
    
    def getTFIDF(self, term, allTermsInDocument, allDocuments):
        tf = self.getTF(term, allTermsInDocument)
        idf = self.getIDF(term, allDocuments)
        return tf * idf
    
    
    def getTF(self, term, allTermsInDocument):
        return allTermsInDocument.count(term) / float(len(allTermsInDocument))
        
        
    def getIDF(self, term, allDocuments):
        # Try to get numDocumentsWithThisTerm from cache
        numDocumentsWithThisTerm = self.term_counts.get(term, 0)
        
        # Compute numDocumentsWithThisTerm if it does not exist in cache
        if (numDocumentsWithThisTerm == 0):
            for doc in allDocuments:
                if (term in doc):
                    numDocumentsWithThisTerm += 1
            self.term_counts[term] = numDocumentsWithThisTerm 
     
        # Avoid division by zero
        if numDocumentsWithThisTerm > 0:
            idf = 1.0 + log(float(len(allDocuments)) / numDocumentsWithThisTerm)
        else:
            idf = 1.0
            
        return idf