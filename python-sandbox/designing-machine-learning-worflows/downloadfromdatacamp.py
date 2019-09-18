# -*- coding: utf-8 -*-
"""
Created on Wed Sep 18 09:34:46 2019

@author: F279814

download from datacamp
"""

#%% uploadFromDatacamp
import numpy as np
import pandas as pd
import inspect
import subprocess

proxy=""
proxy="10.225.92.1:80"

def retrieve_name(var):
    callers_local_vars = inspect.currentframe().f_back.f_back.f_locals.items()
    return [var_name for var_name, var_val in callers_local_vars if var_val is var]

def uploadFromDatacamp(*argv):
    dict_urls = {}
    curl_proxy_option='-q'
    if proxy!='':
        curl_proxy_option='-x'+proxy
    for arg in argv:  
        if (type(arg) == type(str())):
            if (len(retrieve_name(arg)) > 0 ):
                print('string ', retrieve_name(arg)[0], arg)
            else:
                print('direct string: ',arg)
        if (type(arg) == type(pd.DataFrame())):
            nom_du_df = retrieve_name(arg)[0]
            print('dataframe', arg.head())
            arg.to_csv(F"{nom_du_df}.csv")
            nom_du_csv=F"{nom_du_df}.csv"
            print("\nUsing curl to push "+nom_du_csv, flush=True)
            sortie_curl = subprocess.getoutput(['curl', curl_proxy_option, '-F', F"file=@{nom_du_df}.csv", 'https://file.io'])
            dict_urls[nom_du_csv] = urlFromFileIO(sortie_curl)
    return dict_urls

import json
def urlFromFileIO(outputCurl):
    d = json.loads(outputCurl)
    return(d['link'])
    
#%% saveFromFileIO
proxy="10.225.92.1:80"

def saveFromFileIO(dict_urls):
    #we accept boith string and dict
    curl_proxy_option=''
    if proxy!='':
        curl_proxy_option='-x'+proxy

    if (type(dict_urls)==type(str())):
        dict_urls = dict_urls.replace("'", '"')
        dict_urls=json.loads(dict_urls)
    for k, v in dict_urls.items():
        print(k, v)
#curl http://some.url --output some.file
        sortie_curl = subprocess.getoutput(['curl', curl_proxy_option, v, '--output',k])
        print(sortie_curl)


#%% test
import numpy as np
import pandas as pd

test = 'je suis un test'
#uploadFromDatacamp(test, 'test2')
df=pd.read_csv('X.csv')
df2=pd.read_csv('X.csv')

urls = uploadFromDatacamp(df)
#saveFromFileIO(urls)
