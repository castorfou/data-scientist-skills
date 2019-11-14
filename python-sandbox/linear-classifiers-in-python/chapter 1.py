#!/usr/bin/env python
# coding: utf-8

# # Scikit-learn refresher

# ## KNN classification
# In this exercise you'll explore a subset of the Large Movie Review Dataset. The variables X_train, X_test, y_train, and y_test are already loaded into the environment. The X variables contain features based on the words in the movie reviews, and the y variables contain labels for whether the review sentiment is positive (+1) or negative (-1).
# 
# This course touches on a lot of concepts you may have forgotten, so if you ever need a quick refresher, download the Scikit-Learn Cheat Sheet and keep it handy!

# ### init data from datacamp

# In[4]:


#uploadToFileIO(X_train, y_train, X_test,y_test)
tobedownloaded="{scipy.sparse.csr.csr_matrix: {'X_test.npz': 'https://file.io/4Xh8ts',  'X_train.npz': 'https://file.io/lq2Bw8'}, numpy.ndarray: {'y_test.csv': 'https://file.io/qXdUpC',  'y_train.csv': 'https://file.io/EDeV60'}}"
prefix='data_from_datacamp/ZZZ_Chap1.11_'


# In[6]:


from uploadfromdatacamp import saveFromFileIO
#saveFromFileIO(tobedownloaded, prefix=prefix, proxy="10.225.92.1:80")


# In[11]:


from scipy.sparse import load_npz
from uploadfromdatacamp import loadNDArrayFromCsv
X_train = load_npz(prefix+'X_train.npz')
X_test = load_npz(prefix+'X_test.npz')
y_test = loadNDArrayFromCsv(prefix+'y_test.csv')
y_train = loadNDArrayFromCsv(prefix+'y_train.csv')


# ### code from datacamp

# In[13]:


from sklearn.neighbors import KNeighborsClassifier

# Create and fit the model
knn = KNeighborsClassifier()
knn.fit(X_train, y_train)

# Predict on the test features, print the results
pred = knn.predict(X_test[0])
print("Prediction for test example 0:", pred)


# ![image.png](attachment:image.png)

# ## Comparing models
# Compare k nearest neighbors classifiers with k=1 and k=5 on the handwritten digits data set, which is already loaded into the variables X_train, y_train, X_test, and y_test. You can set k with the n_neighbors parameter when creating the KNeighborsClassifier object, which is also already imported into the environment.
# 
# Which model has a higher test accuracy?
# ![image.png](attachment:image.png)

# ### init data from datacamp

# In[15]:


#uploadToFileIO(X_train, y_train, X_test,y_test)
tobedownloaded="{numpy.ndarray: {'X_test.csv': 'https://file.io/wOkEk0',  'X_train.csv': 'https://file.io/4jDsMz',  'y_test.csv': 'https://file.io/vKmnDM',  'y_train.csv': 'https://file.io/7DNhQj'}}"
prefix='data_from_datacamp/ZZZ_Chap1.12_'


# In[16]:


from uploadfromdatacamp import saveFromFileIO
#saveFromFileIO(tobedownloaded, prefix=prefix, proxy="10.225.92.1:80")


# In[18]:


from uploadfromdatacamp import loadNDArrayFromCsv
X_train = loadNDArrayFromCsv(prefix+'X_train.csv')
X_test = loadNDArrayFromCsv(prefix+'X_test.csv')
y_test = loadNDArrayFromCsv(prefix+'y_test.csv')
y_train = loadNDArrayFromCsv(prefix+'y_train.csv')


# ### code from datacamp

# In[21]:


from sklearn.neighbors import KNeighborsClassifier

# Create and fit the model, n_neighbors=1
knn = KNeighborsClassifier(n_neighbors=1)
knn.fit(X_train, y_train)
print(knn.score(X_test, y_test))

# Create and fit the model, n_neighbors=5
knn5 = KNeighborsClassifier(n_neighbors=5)
knn5.fit(X_train, y_train)
print(knn5.score(X_test, y_test))


# ![image.png](attachment:image.png)

# # Applying logistic regression and SVM
# ![image.png](attachment:image.png)

# ## Running LogisticRegression and SVC
# In this exercise, you'll apply logistic regression and a support vector machine to classify images of handwritten digits.

# ### init data from datacamp

# In[30]:


from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split


# ### code from datacamp

# In[31]:


from sklearn import datasets
digits = datasets.load_digits()
X_train, X_test, y_train, y_test = train_test_split(digits.data, digits.target)

# Apply logistic regression and print scores
lr = LogisticRegression()
lr.fit(X_train, y_train)
print(lr.score(X_train, y_train))
print(lr.score(X_test, y_test))

# Apply SVM and print scores
svm = SVC()
svm.fit(X_train, y_train)
print(svm.score(X_train, y_train))
print(svm.score(X_test, y_test))


# ![image.png](attachment:image.png)

# ## Sentiment analysis for movie reviews
# In this exercise you'll explore the probabilities outputted by logistic regression on a subset of the Large Movie Review Dataset.
# 
# The variables X and y are already loaded into the environment. X contains features based on the number of times words appear in the movie reviews, and y contains labels for whether the review sentiment is positive (+1) or negative (-1).

# ### init data from datacamp

# In[1]:


#uploadToFileIO(X, y)
tobedownloaded="{scipy.sparse.csr.csr_matrix: {'X.npz': 'https://file.io/mqLA6C'}, numpy.ndarray: {'y.csv': 'https://file.io/xp8UZB'}}"
prefix='data_from_datacamp/ZZZ_Chap1.22_'


# In[2]:


from uploadfromdatacamp import saveFromFileIO
#saveFromFileIO(tobedownloaded, prefix=prefix, proxy="10.225.92.1:80")


# In[3]:


from scipy.sparse import load_npz
from uploadfromdatacamp import loadNDArrayFromCsv
X = load_npz(prefix+'X.npz')
y = loadNDArrayFromCsv(prefix+'y.csv')


# import inspect<br>
# print(inspect.getsource(get_features))

# In[18]:


#raw_text=vectorizer.inverse_transform(X)
#raw_df=pd.DataFrame(raw_text)
#uploadToFileIO(raw_df)
tobedownloaded="{pandas.core.frame.DataFrame: {'raw_df.csv': 'https://file.io/7vH9ZM'}}"
prefix='data_from_datacamp/ZZZ_Chap22_'


# In[19]:


from uploadfromdatacamp import saveFromFileIO
#saveFromFileIO(tobedownloaded, prefix=prefix, proxy="10.225.92.1:80")


# In[57]:


import pandas as pd
raw_df=pd.read_csv(prefix+'raw_df.csv',index_col=0)
raw_df
raw_text=raw_df.values.tolist()
#on supprime les nan
raw_text_sansNan=raw_df.applymap(lambda x: [x] if pd.notnull(x) else []).sum(1).tolist()
#on change les list de str en ndArray
raw_ndarray=[np.asarray(i) for i in raw_text_sansNan]
type(raw_ndarray[0])


# In[75]:


len(raw_ndarray[10]),type(raw_ndarray[10])
np.asanyarray(raw_ndarray).ravel()


# In[11]:


from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer
import sklearn
print(sklearn.__version__)
vectorizer=CountVectorizer()
def get_features(review):
    return vectorizer.transform([review])


# In[77]:


#X = vectorizer.fit_transform(corpus)
vectorizer.fit_transform(np.asanyarray(raw_ndarray))


# ### code from datacamp

# ![image.png](attachment:image.png)

# In[70]:


# Instantiate logistic regression and train
lr = LogisticRegression(solver='liblinear')
lr.fit(X, y)

# Predict sentiment for a glowing review
review1 = "LOVED IT! This movie was amazing. Top 10 this year."
review1_features = get_features(review1)
print("Review:", review1)
print("Probability of positive review:", lr.predict_proba(review1_features)[0,1])

# Predict sentiment for a poor review
review2 = "Total junk! I'll never watch a film by that director again, no matter how good the reviews."
review2_features = get_features(review2)
print("Review:", review2)
print("Probability of positive review:", lr.predict_proba(review2_features)[0,1])


# In[73]:


review1 = "LOVED IT! This movie was amazing. Top 10 this year."
vectorizer.transform([review1])


# ![image.png](attachment:image.png)

# # Linear classifiers

# ## Visualizing decision boundaries
# In this exercise, you'll visualize the decision boundaries of various classifier types.
# 
# A subset of scikit-learn's built-in wine dataset is already loaded into X, along with binary labels in y.

# ### init data from datacamp

# In[78]:


#uploadToFileIO(X,y)
tobedownloaded="{numpy.ndarray: {'X.csv': 'https://file.io/8R2MXu',  'y.csv': 'https://file.io/0LiZhh'}}"
prefix='data_from_datacamp/ZZZ_Chap1.31_'


# In[79]:


from uploadfromdatacamp import saveFromFileIO
#saveFromFileIO(tobedownloaded, prefix=prefix, proxy="10.225.92.1:80")


# In[84]:


from uploadfromdatacamp import loadNDArrayFromCsv
X = loadNDArrayFromCsv(prefix+'X.csv')
y = loadNDArrayFromCsv(prefix+'y.csv',dtype='bool')


# In[85]:


y[:10]


# In[97]:


#import inspect
#print(inspect.getsource(plot_4_classifiers))
import matplotlib.pyplot as plt
def plot_4_classifiers(X, y, clfs):

    # Set-up 2x2 grid for plotting.
    fig, sub = plt.subplots(2, 2)
    plt.subplots_adjust(wspace=0.2, hspace=0.2)

    for clf, ax, title in zip(clfs, sub.flatten(), ("(1)", "(2)", "(3)", "(4)")):
        # clf.fit(X, y)
        plot_classifier(X, y, clf, ax, ticks=True)
        ax.set_title(title)
    plt.show()
    
def plot_classifier(X, y, clf, ax=None, ticks=False, proba=False, lims=None): # assumes classifier "clf" is already fit
    X0, X1 = X[:, 0], X[:, 1]
    xx, yy = make_meshgrid(X0, X1, lims=lims)

    if ax is None:
        plt.figure()
        ax = plt.gca()
        show = True
    else:
        show = False

    # can abstract some of this into a higher-level function for learners to call
    cs = plot_contours(ax, clf, xx, yy, cmap=plt.cm.coolwarm, alpha=0.8, proba=proba)
    if proba:
        cbar = plt.colorbar(cs)
        cbar.ax.set_ylabel('probability of red $\Delta$ class', fontsize=20, rotation=270, labelpad=30)
        cbar.ax.tick_params(labelsize=14)
    #ax.scatter(X0, X1, c=y, cmap=plt.cm.coolwarm, s=30, edgecolors='k', linewidth=1)
    labels = np.unique(y)
    if len(labels) == 2:
        ax.scatter(X0[y==labels[0]], X1[y==labels[0]], cmap=plt.cm.coolwarm, s=60, c='b', marker='o', edgecolors='k')
        ax.scatter(X0[y==labels[1]], X1[y==labels[1]], cmap=plt.cm.coolwarm, s=60, c='r', marker='^', edgecolors='k')
    else:
        ax.scatter(X0, X1, c=y, cmap=plt.cm.coolwarm, s=50, edgecolors='k', linewidth=1)

    ax.set_xlim(xx.min(), xx.max())
    ax.set_ylim(yy.min(), yy.max())
#     ax.set_xlabel(data.feature_names[0])
#     ax.set_ylabel(data.feature_names[1])
    if ticks:
        ax.set_xticks(())
        ax.set_yticks(())
#     ax.set_title(title)
    if show:
        plt.show()
    else:
        return ax
def make_meshgrid(x, y, h=.02, lims=None):
    """Create a mesh of points to plot in

    Parameters
    ----------
    x: data to base x-axis meshgrid on
    y: data to base y-axis meshgrid on
    h: stepsize for meshgrid, optional

    Returns
    -------
    xx, yy : ndarray
    """

    if lims is None:
        x_min, x_max = x.min() - 1, x.max() + 1
        y_min, y_max = y.min() - 1, y.max() + 1
    else:
        x_min, x_max, y_min, y_max = lims
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    return xx, yy
def plot_contours(ax, clf, xx, yy, proba=False, **params):
    """Plot the decision boundaries for a classifier.

    Parameters
    ----------
    ax: matplotlib axes object
    clf: a classifier
    xx: meshgrid ndarray
    yy: meshgrid ndarray
    params: dictionary of params to pass to contourf, optional
    """
    if proba:
        Z = clf.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:,-1]
        Z = Z.reshape(xx.shape)
        out = ax.imshow(Z,extent=(np.min(xx), np.max(xx), np.min(yy), np.max(yy)), origin='lower', vmin=0, vmax=1, **params)
        ax.contour(xx, yy, Z, levels=[0.5])
    else:
        Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)
        out = ax.contourf(xx, yy, Z, **params)
    return out


# ### code from datacamp

# In[98]:


from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.neighbors import KNeighborsClassifier

# Define the classifiers
classifiers = [LogisticRegression(), LinearSVC(), SVC(), KNeighborsClassifier()]

# Fit the classifiers
for c in classifiers:
    c.fit(X,y)

# Plot the classifiers
plot_4_classifiers(X, y, classifiers)
plt.show()


# ![image.png](attachment:image.png)

# In[ ]:




