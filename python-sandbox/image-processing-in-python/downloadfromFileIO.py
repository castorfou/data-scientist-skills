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
import re

     
#%% saveFromFileIO
# prend en entree un dict : type, filename, url
#       et un prefix optionnel
# et telecharge tout avec les bon prefix+filename    
# avec un mecanisme de lock (supprimer le fichier lock pour forcer le retelechargement)
def saveFromFileIO(dict_urls, prefix='', proxy='', prefixToc=''):
    if (len(prefixToc)>0):
        prefix=getPrefixfromTOC(prefixToc)
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
    return prefix
    

#%% getPrefixfromTOC
def getPrefixfromTOC(prefixToc):
    notebook_fullname = notebook_path()
    notebook_name = os.path.basename(notebook_fullname)
    notebook = os.path.splitext(notebook_name)[0]
    prefix = 'data_from_datacamp/'+notebook
    prefix=prefix+'-Exercise'+prefixToc+'_'
    return prefix
    
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


#%% getImage
def getImage(filename, dtype='uint8'):
    myArray = np.genfromtxt(filename, delimiter=',', dtype=dtype)
    entre_crochet = re.search(r"\[([A-Za-z0-9_]+)\]", filename)
    shape_text=entre_crochet.group(1)
    shape = tuple(int(i) for i in shape_text.split('_'))
    return np.reshape(myArray, shape)

    
    
    
def loadModelFromH5(filename):
    return load_model(filename)
	
def print_func(fonction):
  lines = inspect.getsource(fonction)
  print(lines)

  
from notebook import notebookapp
import urllib
import json
import os
import ipykernel

def notebook_path():
    """Returns the absolute path of the Notebook or None if it cannot be determined
    NOTE: works only when the security is token-based or there is also no password
    """
    connection_file = os.path.basename(ipykernel.get_connection_file())
    kernel_id = connection_file.split('-', 1)[1].split('.')[0]

    for srv in notebookapp.list_running_servers():
        try:
            if srv['token']=='' and not srv['password']:  # No token and no password, ahem...
                req = urllib.request.urlopen(srv['url']+'api/sessions')
            else:
                req = urllib.request.urlopen(srv['url']+'api/sessions?token='+srv['token'])
            sessions = json.load(req)
            for sess in sessions:
                if sess['kernel']['id'] == kernel_id:
                    return os.path.join(srv['notebook_dir'],sess['notebook']['path'])
        except:
            pass  # There may be stale entries in the runtime directory 
    return None
    
''' 
get_ipython().run_line_magic('reload_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')

from downloadfromFileIO import getPrefixfromTOC

getPrefixfromTOC('1.1')
'''