'''
Created on Mar 21, 2016

@author: robert
'''
from SearchEngine import SearchEngine

if __name__ == '__main__':
    
    engine = SearchEngine()
    
    while (True):
        num_results = input("Please enter the number of entries you like to retrieve for each query: [10] ")
        if (num_results):
            if (num_results.isdigit()):
                engine.setNumResults(int(num_results))
                break;
            else:
                print("Invalid input")
        else:
            break;
    
    while (True):
        isLoadFromCache = input("Do you want to start search engine from cached data file? [Y/n] ")
        if (not isLoadFromCache or isLoadFromCache.lower() == "y"):
            try:
                cache_file_path = input("Please enter full path to the cached data file:")
                engine.initFromCache(cache_file_path)
                break
            except Exception as e:
                print(e)
            
        else:
            try:
                data_dir_path = input("Please enter full directory path to the dataset:")
                cache_file_path = input("Where do you want to save the new cached data file:")
                engine.initFromDataset(data_dir_path, cache_file_path)
                break
            except Exception as e:
                print(e)
                
            
    while (True):
        query = input("Please enter your query:")
        if (query):
            results = engine.findResults(query)
            if (len(results) > 0):
                print(results)
            else:
                print("No results match your query.")
        else:
            break