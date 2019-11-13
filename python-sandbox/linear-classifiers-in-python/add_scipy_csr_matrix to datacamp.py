# -*- coding: utf-8 -*-
"""
Created on Tue Nov 12 16:44:17 2019

@author: F279814
"""

#%% ecrire une matrice

import scipy.sparse
import numpy as np

sparse_matrix = scipy.sparse.csr_matrix(np.array([[0, 0, 3], [4, 0, 0]]))

sparse_matrix.todense()


scipy.sparse.save_npz('data/sparse_matrix.npz', sparse_matrix) 


#%% lire une matrice
sparse_matrix_load = scipy.sparse.load_npz('data/sparse_matrix.npz')

sparse_matrix_load.todense()

#%% utiliser uploadfromdatacamp

from uploadfromdatacamp import *

uploadToFileIO(sparse_matrix_load,sparse_matrix , proxy="10.225.92.1:80")

