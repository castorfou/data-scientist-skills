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
from keras.models import Sequential, load_model
from simhash import Simhash
import os

     
#%% saveFromFileIO
# prend en entree un dict : type, filename, url
#       et un prefix optionnel
# et telecharge tout avec les bon prefix+filename    
# avec un mecanisme de lock (supprimer le fichier lock pour forcer le retelechargement)
def saveFromFileIO(dict_urls, prefix='', proxy=''):
    hash =  Simhash(dict_urls).value
    filename_lock = prefix+str(hash)+".lock"
    #si le fichier existe
    if os.path.exists(filename_lock):
        os.utime(filename_lock, None)
        print("Téléchargements déjà effectués - SKIP")
    else:
        print("Téléchargements à lancer")
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
        open(filename_lock, 'a').close()
    
            
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


def loadModelFromH5(filename):
    return load_model(filename)
	
def print_func(fonction):
  lines = inspect.getsource(fonction)
  print(lines)
