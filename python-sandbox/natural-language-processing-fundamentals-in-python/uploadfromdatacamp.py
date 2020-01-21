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
import pickle

#%% uploadToFileIO(*argv):
    # liste des objets à envoyer
    #print un nested dictionnaire avec {type: {filename: url_sur_fileIO}}
    #à donner en entree de saveFromFileIO
def uploadToFileIO(*argv, proxy=''):
    dict_urls = {}
    for arg in argv:
        filename = uploadToFileIO_get_filename(arg)
        uploadToFileIO_saveas_filename(arg, filename)
        dict_str=dict_urls.get(type(arg),{})
        dict_str[filename]=uploadToFileIO_pushto_fileio(filename, proxy)
        dict_urls[type(arg)]=dict_str
    return dict_urls


#retieve the name of the variable given in a list
def retrieve_name(var):
    callers_local_vars = inspect.currentframe().f_back.f_back.f_back.f_locals.items()
    return [var_name for var_name, var_val in callers_local_vars if var_val is var]

#return url from curl output (closed to json format)
def urlFromFileIO(outputCurl):
    #extract text between {}
    outputCurl=outputCurl[outputCurl.find("{"):outputCurl.find("}")+1]
    print(outputCurl)
    d = json.loads(outputCurl)
    return(d['link'])

#return appropriate filename for a given variable
def uploadToFileIO_get_filename(variable):
    filename=retrieve_name(variable)[0]
    if ( type(variable) == type(pd.Series()) or type(variable) == type(pd.DataFrame()) or type(variable) == type(np.asarray([ [1,2,3], [4,5,6], [7,8,9] ])) ):
        filename=filename+".csv"
    if (type(variable) == type(str()) or type(variable) == type(list())):
        filename=filename+".txt"
    return filename;

#save variable as a file named filename
#no return
def uploadToFileIO_saveas_filename(variable,filename):
    if ( type(variable) == type(pd.Series()) or type(variable) == type(pd.DataFrame()) ):
        variable.to_csv(filename, sep=',')
    if (type(variable) == type(np.asarray([ [1,2,3], [4,5,6], [7,8,9] ]))):
        np.savetxt(filename, variable, fmt='%5s',delimiter=",")
        #variable.tofile(filename,format='%5s',sep=",")
    if (type(variable) == type(str()) or type(variable) == type(list())):
        with open(filename, 'w') as f:
            f.write(json.dumps(variable))



#upload filename to file.io and return url of this file on file.io
#as an optionnal parameter take a proxy            
def uploadToFileIO_pushto_fileio(filename,proxy=''):
    curl_proxy_option='-q'
    if proxy!='':
        curl_proxy_option='-x'+proxy
    curl_command=" ".join(str(x) for x in ['curl', curl_proxy_option, '-F', "file=@"+filename, 'https://file.io'])
    sortie_curl = subprocess.getoutput(curl_command)
    return urlFromFileIO(sortie_curl)
    
#%% saveFromFileIO
# prend en entree un dict : type, filename, url
#       et un prefix optionnel
# et telecharge tout avec les bon prefix+filename    
def saveFromFileIO(dict_urls, prefix='', proxy=''):
    #we accept both string and dict
    curl_proxy_option='-q'
    if proxy!='':
        curl_proxy_option='-x'+proxy
    if (type(dict_urls)==type(str())):
        dict_urls = dict_urls.replace("'", '"')
        print(dict_urls)
        dict_urls=yaml.load(dict_urls, Loader=yaml.FullLoader)
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
def loadNDArrayFromCsv(filename, dtype='float32'):
    myArray = np.genfromtxt(filename, delimiter=',', dtype=dtype)
    #myArray = np.fromfile(filename, sep=',', dtype=dtype)
    return myArray

	
def print_func(fonction):
  lines = inspect.getsource(fonction)
  print(lines)
