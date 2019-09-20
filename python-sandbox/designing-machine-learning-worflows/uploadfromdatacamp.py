# -*- coding: utf-8 -*-
"""
Created on Wed Sep 18 17:00:46 2019

@author: F279814

upload/download from datacamp
"""

#%% all imports
#to be run in datacamp
import numpy as np
import pandas as pd
import inspect
import subprocess
import json
import yaml


#%% uploadFromDatacamp
#to be run in datacamp

proxy=""
#to uncomment when working from Michelin
#proxy="10.225.92.1:80"

def retrieve_name(var):
    callers_local_vars = inspect.currentframe().f_back.f_back.f_locals.items()
    return [var_name for var_name, var_val in callers_local_vars if var_val is var]

def uploadFromDatacamp(*argv):
    dict_urls = {}
    curl_proxy_option='-q'
    if proxy!='':
        curl_proxy_option='-x'+proxy
    array_int = np.asarray([ [1,2,3], [4,5,6], [7,8,9] ])
    for arg in argv:  
        if (type(arg) == type(str())):
            #get the list of str
            dict_str=dict_urls.get(type(arg),{})
            if (len(retrieve_name(arg)) > 0 ):
                #print('string ', retrieve_name(arg)[0], arg)
                dict_str[retrieve_name(arg)[0]]=arg
                dict_urls[type(arg)]=dict_str
            #else:
            #on ne fait rien avec les strings passes en direct
                #print('direct string: ',arg)
#        if (type(arg) == (type(pd.DataFrame()) or type(pd.Series()))):
        if (type(arg) == (type(pd.Series())) or type(arg) == (type(pd.DataFrame()))):
            nom_du_df = retrieve_name(arg)[0]
            nom_du_csv=F"{nom_du_df}.csv"
            arg.to_csv(nom_du_csv)
            curl_command=" ".join(str(x) for x in ['curl', curl_proxy_option, '-F', "file=@"+nom_du_csv, 'https://file.io'])
            sortie_curl = subprocess.getoutput(curl_command)
            dict_str=dict_urls.get(type(arg),{})
            dict_str[nom_du_csv]=urlFromFileIO(sortie_curl)
            dict_urls[type(arg)]=dict_str
        if (type(arg) == type(list())):
            if (len(retrieve_name(arg)) > 0 ):
            #on ne fait rien avec les listes passees en direct
                nom_de_la_liste = retrieve_name(arg)[0]
                nom_du_txt=F"{nom_de_la_liste}.txt"
                with open(nom_du_txt, 'w') as f:
                    f.write(json.dumps(arg))        
                curl_command=" ".join(str(x) for x in ['curl', curl_proxy_option, '-F', "file=@"+nom_du_txt, 'https://file.io'])
                sortie_curl = subprocess.getoutput(curl_command)
                dict_str=dict_urls.get(type(arg),{})
                dict_str[nom_du_txt]=urlFromFileIO(sortie_curl)
                dict_urls[type(arg)]=dict_str
        
        if (type(arg) == type(array_int)):
            nom_du_nd = retrieve_name(arg)[0]
            nom_du_csv=F"{nom_du_nd}.csv"            
            np.savetxt(nom_du_csv, arg, delimiter=",")
            curl_command=" ".join(str(x) for x in ['curl', curl_proxy_option, '-F', "file=@"+nom_du_csv, 'https://file.io'])
            sortie_curl = subprocess.getoutput(curl_command)
            dict_str=dict_urls.get(type(arg),{})
            dict_str[nom_du_csv]=urlFromFileIO(sortie_curl)
            dict_urls[type(arg)]=dict_str
    return dict_urls

def urlFromFileIO(outputCurl):
    #extract text between {}
    outputCurl=outputCurl[outputCurl.find("{"):outputCurl.find("}")+1]
    print(outputCurl)
    d = json.loads(outputCurl)
    return(d['link'])
    
#%% saveFromFileIO
# prend en entree un dict : type, filename, url
#       et un prefix optionnel
# et telecharge tout avec les bon prefix+filename    

proxy="10.225.92.1:80"

def saveFromFileIO(dict_urls, prefix=''):
    #we accept both string and dict
    curl_proxy_option='-q'
    if proxy!='':
        curl_proxy_option='-x'+proxy
    if (type(dict_urls)==type(str())):
        dict_urls = dict_urls.replace("'", '"')
        print(dict_urls)
        dict_urls=yaml.load(dict_urls)
    print(dict_urls)
    for python_type, filename_url in dict_urls.items():
        for filename, url in filename_url.items():
            #print(prefix+filename, url)
            sortie_curl = subprocess.getoutput(['curl', curl_proxy_option, url, '--output',prefix+filename])
            print(sortie_curl)

#%% loadlistfromtxt
def loadListFromTxt(filename):
    liste=[]
    with open(filename, 'r') as f:
        liste = json.loads(f.read())
    return liste

#%% loadndarrayfromcsv
def loadNDArrayFromCsv(filename):
    myArray = np.genfromtxt(filename, delimiter=',')
    return myArray

#%% test uploadFromDatacamp with new dict structure
def test():
    TEST_uploadFromDatacamp_test = 'je suis un test'
    TEST_uploadFromDatacamp_test2 = 'test2'
    TEST_uploadFromDatacamp_liste=[1,2,3]
    TEST_uploadFromDatacamp_liste2=['test','re','3256',32]
    TEST_uploadFromDatacamp_s = pd.Series([1, 2, 5, 7])
    TEST_uploadFromDatacamp_X_test_test=pd.read_csv('X_test.csv',index_col=0)
    TEST_uploadFromDatacamp_X_train_test=pd.read_csv('X_train.csv',index_col=0)
    
    #should create 7 files TEST_uploadFromDatacamp_*.{csv,txt}
    uploadFromDatacamp(TEST_uploadFromDatacamp_s, TEST_uploadFromDatacamp_liste, TEST_uploadFromDatacamp_liste2, \
                       TEST_uploadFromDatacamp_X_test_test, TEST_uploadFromDatacamp_X_train_test,TEST_uploadFromDatacamp_test, TEST_uploadFromDatacamp_test2 )
    
    TEST_uploadFromDatacamp_a = np.asarray([ [1,2,3], [4,5,6], [7,8,9] ])
    TEST_uploadFromDatacamp_b = np.asarray([ [1.,2.], [4.,5.] ])
    
    #should create 2 files TEST_uploadFromDatacamp_*.{csv,txt}
    uploadFromDatacamp(TEST_uploadFromDatacamp_a,TEST_uploadFromDatacamp_b)
    
#%% get function from datacamp (example)
#import inspect
#print(inspect.getsource(my_metric))
   
    