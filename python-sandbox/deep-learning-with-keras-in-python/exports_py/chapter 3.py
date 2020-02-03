#!/usr/bin/env python
# coding: utf-8

# # Learning curves

# ## Learning the digits
# You're going to build a model on the digits dataset, a sample dataset that comes pre-loaded with scikit learn. The digits dataset consist of 8x8 pixel handwritten digits from 0 to 9:
# 
# 
# You want to distinguish between each of the 10 possible digits given an image, so we are dealing with multi-class classification.
# The dataset has already been partitioned into X_train, y_train, X_test, and y_test using 30% of the data as testing data. The labels are one-hot encoded vectors, so you don't need to use Keras to_categorical() function.
# 
# Let's build this new model!

# ### init

# In[1]:


#upload and download

from uploadfromdatacamp import saveFromFileIO
""" à executer sur datacamp: (apres copie du code uploadfromdatacamp.py)
uploadToFileIO(X_train, y_train, X_test,  y_test)
"""

tobedownloaded="""
{numpy.ndarray: {'X_test.csv': 'https://file.io/nTYzyX',
  'X_train.csv': 'https://file.io/9aZm3g',
  'y_test.csv': 'https://file.io/UMtsxi',
  'y_train.csv': 'https://file.io/yIMzde'}}
"""
prefix='data_from_datacamp/Chap3-Exercise1.1_'
#saveFromFileIO(tobedownloaded, prefix=prefix, proxy="10.225.92.1:80")

#initialisation

from uploadfromdatacamp import loadNDArrayFromCsv
X_test = loadNDArrayFromCsv(prefix+'X_test.csv')
X_train = loadNDArrayFromCsv(prefix+'X_train.csv')
y_test = loadNDArrayFromCsv(prefix+'y_test.csv')
y_train = loadNDArrayFromCsv(prefix+'y_train.csv')


# ### code

# In[2]:


# Import the sequential model and dense layer
from keras.models import Sequential
from keras.layers import Dense


# In[4]:


# Instantiate a Sequential model
model = Sequential()

# Input and hidden layer with input_shape, 16 neurons, and relu 
model.add(Dense(16, input_shape = (64,), activation = 'relu'))

# Output layer with 10 neurons (one per digit) and softmax
model.add(Dense(10, activation='softmax'))

# Compile your model
model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])

# Test if your model works and can process input data
print(model.predict(X_train))


# ## Is the model overfitting?
# Let's train the model you just built and plot its learning curve to check out if it's overfitting! You can make use of loaded function plot_loss() to plot training loss against validation loss, you can get both from the history callback.
# 
# If you want to inspect the plot_loss() function code, paste this in the console: print(inspect.getsource(plot_loss))

# ### init

# In[6]:


""" à executer sur datacamp: (apres copie du code uploadfromdatacamp.py)
import inspect
print_func(plot_loss)
"""
import matplotlib.pyplot as plt
def plot_loss(loss,val_loss):
  plt.figure()
  plt.plot(loss)
  plt.plot(val_loss)
  plt.title('Model loss')
  plt.ylabel('Loss')
  plt.xlabel('Epoch')
  plt.legend(['Train', 'Test'], loc='upper right')
  plt.show()


# ### code

# In[7]:


# Train your model for 60 epochs, using X_test and y_test as validation data
history = model.fit(X_train, y_train, epochs=60, validation_data=(X_test, y_test), verbose=0)

# Extract from the history object loss and val_loss to plot the learning curve
plot_loss(history.history['loss'], history.history['val_loss'])


# ## Do we need more data?
# It's time to check whether the digits dataset model you built benefits from more training examples!
# 
# In order to keep code to a minimum, various things are already initialized and ready to use:
# 
# The model you just built.
# X_train,y_train,X_test, and y_test.
# The initial_weights of your model, saved after using model.get_weights().
# A defined list of training sizes: training_sizes.
# A defined EarlyStopping callback monitoring loss: early_stop.
# Two empty lists to store the evaluation results: train_accs and test_accs.
# Train your model on the different training sizes and evaluate the results on X_test. End by plotting the results with plot_results().
# 
# The full code for this exercise can be found on the slides!

# ### init

# In[13]:


train_sizes = [ 125,  502,  879, 1255]
initial_weights = model.get_weights()
# Import the early stopping callback
from keras.callbacks import EarlyStopping

# Define a callback to monitor val_acc
early_stop = EarlyStopping(monitor='loss', patience=5)
train_accs=[]
test_accs=[]

from sklearn.model_selection import train_test_split

""" à executer sur datacamp: (apres copie du code uploadfromdatacamp.py)
import inspect
inspect.print_func(plot_results)
"""
import matplotlib.pyplot as plt
def plot_results(train_accs,test_accs):
  plt.plot(train_sizes, train_accs, 'o-', label="Training Accuracy")
  plt.plot(train_sizes, test_accs, 'o-', label="Test Accuracy")
  plt.title('Accuracy vs Number of training samples')
  plt.xlabel('Training samples')
  plt.ylabel('Accuracy')
  plt.legend(loc="best")
  plt.show()


# ### code

# In[14]:


for size in train_sizes:
  	# Get a fraction of training data (we only care about the training data)
    X_train_frac, X_test_frac, y_train_frac, y_test_frac = train_test_split(
      X_train, y_train, train_size = size)
    # Set the model weights and fit the model on the training data
    model.set_weights(initial_weights)
    model.fit(X_train_frac, y_train_frac, epochs = 50, callbacks = [early_stop])

    # Evaluate and store the train fraction and the complete test set results
    train_accs.append(model.evaluate(X_train_frac, y_train_frac)[1])
    test_accs.append(model.evaluate(X_test, y_test)[1])

# Plot train vs test accuracies
plot_results(train_accs, test_accs)


# # Activation functions
# 

# ## Comparing activation functions
# Comparing activation functions involves a bit of coding, but nothing you can't do!
# 
# You will try out different activation functions on the multi-label model you built for your irrigation machine in chapter 2. The function get_model() returns a copy of this model and applies the activation function, passed on as a parameter, to its hidden layer.
# 
# You will build a loop that goes through several activation functions, generates a new model for each and trains it. Storing the history callback in a dictionary will allow you to compare and visualize which activation function performed best in the next exercise!

# ### init

# In[19]:


""" à executer sur datacamp: (apres copie du code uploadfromdatacamp.py)
import inspect
print_func(get_model)
"""
def get_model(act_function):
  if act_function not in ['relu', 'leaky_relu', 'sigmoid', 'tanh']:
    raise ValueError('Make sure your activation functions are named correctly!')
  print("Finishing with",act_function,"...")
  return ModelWrapper(act_function)

#upload and download

from uploadfromdatacamp import saveFromFileIO
""" à executer sur datacamp: (apres copie du code uploadfromdatacamp.py)
uploadToFileIO(X_train, y_train, X_test, y_test)
"""

tobedownloaded="""
{numpy.ndarray: {'X_test.csv': 'https://file.io/tdIosC',
  'X_train.csv': 'https://file.io/MBtOlo',
  'y_test.csv': 'https://file.io/JtPlid',
  'y_train.csv': 'https://file.io/XXrEWI'}}
"""
prefix='data_from_datacamp/Chap3-Exercise2.1_'
#saveFromFileIO(tobedownloaded, prefix=prefix, proxy="10.225.92.1:80")

#initialisation

from uploadfromdatacamp import loadNDArrayFromCsv
X_train = loadNDArrayFromCsv(prefix+'X_train.csv')
y_train = loadNDArrayFromCsv(prefix+'y_train.csv')
X_test = loadNDArrayFromCsv(prefix+'X_test.csv')
y_test = loadNDArrayFromCsv(prefix+'y_test.csv')


# ### code

# In[20]:


# Set a seed
np.random.seed(27)

# Activation functions to try
activations = ['relu', 'leaky_relu', 'sigmoid', 'tanh']

# Loop over the activation functions
activation_results = {}

for act in activations:
  # Get a new model with the current activation
  model = get_model(act)
  # Fit the model
  history = model.fit(X_train, y_train, epochs=20, verbose=0, validation_data=(X_test, y_test))
  activation_results[act] = history


# ## Comparing activation functions II
# The code used in the previous exercise has been executed to obtain theactivation_results with the difference that 100 epochs instead of 20 are used. That way you'll have more epochs to further compare how the training evolves per activation function.
# 
# For every history callback of each activation function in activation_results:
# 
# The history.history['val_loss'] has been extracted.
# The history.history['val_acc'] has been extracted.
# Both are saved in two dictionaries: val_loss_per_function and val_acc_per_function.
# Pandas is also loaded for you to use as pd. Let's plot some quick comparison validation loss and accuracy charts with pandas!

# ### code

# In[21]:


# Create a dataframe from val_loss_per_function
val_loss= pd.DataFrame(val_loss_per_function)

# Call plot on the dataframe
val_loss.plot()
plt.show()

# Create a dataframe from val_acc_per_function
val_acc = pd.DataFrame(val_acc_per_function)

# Call plot on the dataframe
val_acc.plot()
plt.show()


# # Batch size and batch normalization
# 

# ## Changing batch sizes
# You've seen models are usually trained in batches of a fixed size. The smaller a batch size, the more weight updates per epoch, but at a cost of a more unstable gradient descent. Specially if the batch size is too small and it's not representative of the entire training set.
# 
# Let's see how different batch sizes affect the accuracy of a binary classification model that separates red from blue dots.
# 
# You'll use a batch size of one, updating the weights once per sample in your training set for each epoch. Then you will use the entire dataset, updating the weights only once per epoch.

# ### init

# In[26]:


""" à executer sur datacamp: (apres copie du code uploadfromdatacamp.py)
import inspect
print_func(get_model)
"""
def get_model():
  model = Sequential()
  model.add(Dense(4,input_shape=(2,),activation='relu'))
  model.add(Dense(1,activation="sigmoid"))
  model.compile('sgd', 'binary_crossentropy', metrics=['accuracy'])
  return model

#upload and download

from uploadfromdatacamp import saveFromFileIO
""" à executer sur datacamp: (apres copie du code uploadfromdatacamp.py)
uploadToFileIO(X_train, y_train, y_test, X_test)
"""

tobedownloaded="""
{numpy.ndarray: {'X_test.csv': 'https://file.io/skgd1g',
  'X_train.csv': 'https://file.io/pkVnje',
  'y_test.csv': 'https://file.io/TrwSD0',
  'y_train.csv': 'https://file.io/82MOmj'}}
"""
prefix='data_from_datacamp/Chap3-Exercise3.1_'
#saveFromFileIO(tobedownloaded, prefix=prefix, proxy="10.225.92.1:80")

#initialisation

from uploadfromdatacamp import loadNDArrayFromCsv
X_train = loadNDArrayFromCsv(prefix+'X_train.csv')
y_train = loadNDArrayFromCsv(prefix+'y_train.csv')
X_test = loadNDArrayFromCsv(prefix+'X_test.csv')
y_test = loadNDArrayFromCsv(prefix+'y_test.csv')


# ### code

# In[27]:


# Get a fresh new model with get_model
model = get_model()

# Train your model for 5 epochs with a batch size of 1
model.fit(X_train, y_train, epochs=5, batch_size=1)
print("\n The accuracy when using a batch of size 1 is: ",
      model.evaluate(X_test, y_test)[1])


# In[29]:


model = get_model()

# Fit your model for 5 epochs with a batch of size the training set
model.fit(X_train, y_train, epochs=5, batch_size=X_train.shape[0])
print("\n The accuracy when using the whole training set as a batch was: ",
      model.evaluate(X_test, y_test)[1])


# ## Batch normalizing a familiar model
# Remember the digits dataset you trained in the first exercise of this chapter?
# 
# 
# A multi-class classification problem that you solved using softmax and 10 neurons in your output layer.
# You will now build a new deeper model consisting of 3 hidden layers of 50 neurons each, using batch normalization in between layers. The kernel_initializer parameter is used to initialize weights in a similar way.

# ### code

# In[31]:


# Import batch normalization from keras layers
from keras.layers import BatchNormalization

# Build your deep network
batchnorm_model = Sequential()
batchnorm_model.add(Dense(50, input_shape=(64,), activation='relu', kernel_initializer='normal'))
batchnorm_model.add(BatchNormalization())
batchnorm_model.add(Dense(50, activation='relu', kernel_initializer='normal'))
batchnorm_model.add(BatchNormalization())
batchnorm_model.add(Dense(50, activation='relu', kernel_initializer='normal'))
batchnorm_model.add(BatchNormalization())
batchnorm_model.add(Dense(10, activation='softmax', kernel_initializer='normal'))

# Compile your model with sgd
batchnorm_model.compile(optimizer='sgd', loss='categorical_crossentropy', metrics=['accuracy'])


# ## Batch normalization effects
# Batch normalization tends to increase the learning speed of our models and make their learning curves more stable. Let's see how two identical models with and without batch normalization compare.
# 
# The model you just built batchnorm_model is loaded for you to use. An exact copy of it without batch normalization: standard_model, is available as well. You can check their summary() in the console. X_train, y_train, X_test, and y_test are also loaded so that you can train both models.
# 
# You will compare the accuracy learning curves for both models plotting them with compare_histories_acc().
# 
# You can check the function pasting print(inspect.getsource(compare_histories_acc)) in the console.

# ### init

# In[33]:


#print(inspect.getsource(compare_histories_acc))
import matplotlib.pyplot as plt
def compare_histories_acc(h1,h2):
  plt.plot(h1.history['acc'])
  plt.plot(h1.history['val_acc'])
  plt.plot(h2.history['acc'])
  plt.plot(h2.history['val_acc'])
  plt.title("Batch Normalization Effects")
  plt.xlabel('Epoch')
  plt.ylabel('Accuracy')
  plt.legend(['Train', 'Test', 'Train with Batch Normalization', 'Test with Batch Normalization'], loc='best')
  plt.show()

#upload and download

from uploadfromdatacamp import saveFromFileIO
""" à executer sur datacamp: (apres copie du code uploadfromdatacamp.py)
uploadToFileIO(X_train, y_train, y_test, X_test)
"""

tobedownloaded="""
{numpy.ndarray: {'X_test.csv': 'https://file.io/7IsHZ0',
  'X_train.csv': 'https://file.io/Ji9O8i',
  'y_test.csv': 'https://file.io/Yq3e1a',
  'y_train.csv': 'https://file.io/SoTP67'}}
"""
prefix='data_from_datacamp/Chap3-Exercise3.3_'
#saveFromFileIO(tobedownloaded, prefix=prefix, proxy="10.225.92.1:80")

#initialisation

from uploadfromdatacamp import loadNDArrayFromCsv
X_train = loadNDArrayFromCsv(prefix+'X_train.csv')
y_train = loadNDArrayFromCsv(prefix+'y_train.csv')
X_test = loadNDArrayFromCsv(prefix+'X_test.csv')
y_test = loadNDArrayFromCsv(prefix+'y_test.csv')


# In[36]:


# Build your deep network
standard_model = Sequential()
standard_model.add(Dense(50, input_shape=(64,), activation='relu', kernel_initializer='normal'))
standard_model.add(Dense(50, activation='relu', kernel_initializer='normal'))
standard_model.add(Dense(50, activation='relu', kernel_initializer='normal'))
standard_model.add(Dense(10, activation='softmax', kernel_initializer='normal'))

# Compile your model with sgd
standard_model.compile(optimizer='sgd', loss='categorical_crossentropy', metrics=['accuracy'])


# ### code

# In[37]:


# Train your standard model, storing its history
history1 = standard_model.fit(X_train, y_train, validation_data=(X_test,y_test), epochs=10, verbose=0)

# Train the batch normalized model you recently built, store its history
history2 = batchnorm_model.fit(X_train, y_train, validation_data=(X_test,y_test), epochs=10, verbose=0)

# Call compare_acc_histories passing in both model histories
compare_histories_acc(history1, history2)


# # Hyperparameter tuning
# 

# ## Preparing a model for tuning
# Let's tune the hyperparameters of a binary classification model that does well classifying the breast cancer dataset.
# 
# You've seen that the first step to turn a model into a sklearn estimator is to build a function that creates it. This function is important since you can play with the parameters it receives to achieve the different models you'd like to try out.
# 
# Build a simple create_model() function that receives a learning rate and an activation function as parameters. The Adam optimizer has been imported as an object from keras.optimizers so that you can change its learning rate parameter.

# ### code

# In[38]:


# Creates a model given an activation and learning rate
def create_model(learning_rate=0.01, activation='relu'):
  
    # Create an Adam optimizer with the given learning rate
    opt = Adam(lr=learning_rate)

    # Create your binary classification model  
    model = Sequential()
    model.add(Dense(128, input_shape=(30,), activation=activation))
    model.add(Dense(256, activation=activation))
    model.add(Dense(1, activation='sigmoid'))

    # Compile your model with your optimizer, loss, and metrics
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model


# ## Tuning the model parameters
# It's time to try out different parameters on your model and see how well it performs!
# 
# The create_model() function you built in the previous exercise is loaded for you to use.
# 
# Since fitting the RandomizedSearchCV would take too long, the results you'd get are printed in the show_results() function. You could try random_search.fit(X,y) in the console yourself to check it does work after you have built everything else, but you will probably timeout your exercise (so copy your code first if you try it!).
# 
# You don't need to use the optional epochs and batch_size parameters when building your KerasClassifier since you are passing them as params to the random search and this works as well.

# ### init

# In[50]:


#print(inspect.getsource(show_results))
def show_results():
  print("Best: 0.975395 using {learning_rate: 0.001, epochs: 50, batch_size: 128, activation: relu} \n 0.956063 (0.013236) with: {learning_rate: 0.1, epochs: 200, batch_size: 32, activation: tanh} \n 0.970123 (0.019838) with: {learning_rate: 0.1, epochs: 50, batch_size: 256, activation: tanh} \n 0.971880 (0.006524) with: {learning_rate: 0.01, epochs: 100, batch_size: 128, activation: tanh} \n 0.724077 (0.072993) with: {learning_rate: 0.1, epochs: 50, batch_size: 32, activation: relu} \n 0.588752 (0.281793) with: {learning_rate: 0.1, epochs: 100, batch_size: 256, activation: relu} \n 0.966608 (0.004892) with: {learning_rate: 0.001, epochs: 100, batch_size: 128, activation: tanh} \n 0.952548 (0.019734) with: {learning_rate: 0.1, epochs: 50, batch_size: 256, activation: relu} \n 0.971880 (0.006524) with: {learning_rate: 0.001, epochs: 200, batch_size: 128, activation: relu}\n 0.968366 (0.004239) with: {learning_rate: 0.01, epochs: 100, batch_size: 32, activation: relu}\n 0.910369 (0.055824) with: {learning_rate: 0.1, epochs: 100, batch_size: 128, activation: relu}")

from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import KFold

#upload and download

from uploadfromdatacamp import saveFromFileIO
""" à executer sur datacamp: (apres copie du code uploadfromdatacamp.py)
uploadToFileIO(X, y)
"""

tobedownloaded="""
{numpy.ndarray: {'X.csv': 'https://file.io/b9GjD2',
  'y.csv': 'https://file.io/BxNyNL'}}
"""
prefix='data_from_datacamp/Chap3-Exercise4.2_'
#saveFromFileIO(tobedownloaded, prefix=prefix, proxy="10.225.92.1:80")

#initialisation

from uploadfromdatacamp import loadNDArrayFromCsv
X = loadNDArrayFromCsv(prefix+'X.csv')
y = loadNDArrayFromCsv(prefix+'y.csv')


from keras.optimizers import Adam


# ### code

# In[51]:


# Import KerasClassifier from keras wrappers
from keras.wrappers.scikit_learn import KerasClassifier

# Create a KerasClassifier
model = KerasClassifier(build_fn = create_model)

# Define the parameters to try out
params = {'activation': ['relu', 'tanh'], 'batch_size': [32, 128, 256], 
          'epochs': [50, 100, 200], 'learning_rate': [0.1, 0.01, 0.001]}

# Create a randomize search cv object passing in the parameters to try
random_search = RandomizedSearchCV(model, param_distributions = params, cv = KFold(3))

# Running random_search.fit(X,y) would start the search,but it takes too long! 
show_results()


# In[52]:


random_search.fit(X,y)


# In[53]:


random_search.best_estimator_


# In[54]:


random_search.best_params_


# In[55]:


random_search.best_score_


# ## Training with cross-validation
# Time to train your model with the best parameters found: 0.001 for the learning rate, 50 epochs,a 128 batch_size and relu activations.
# 
# The create_model() function has been redefined so that it now creates a model with those parameters. X and y are loaded for you to use as features and labels.
# 
# In this exercise you do pass the best epochs and batchsize values found for your model to the KerasClassifier object so that they are used when performing crossvalidation.
# 
# End this chapter by training an awesome tuned model on the breast cancer dataset!

# ### init

# In[57]:


#print(inspect.getsource(create_model))
def create_model():
  opt = Adam(lr=0.001)
  model = Sequential()
  model.add(Dense(128,input_shape=(30,),activation='relu'))
  model.add(Dense(256,activation='tanh'))
  model.add(Dense(1,activation='sigmoid'))
  model.compile(optimizer=opt,loss='binary_crossentropy',metrics=['accuracy'])
  return model

#upload and download

from uploadfromdatacamp import saveFromFileIO
""" à executer sur datacamp: (apres copie du code uploadfromdatacamp.py)
uploadToFileIO(X, y)
"""

tobedownloaded="""
{numpy.ndarray: {'X.csv': 'https://file.io/pWaHsi',
  'y.csv': 'https://file.io/0U2TNf'}}
"""
prefix='data_from_datacamp/Chap3-Exercise4.3_'
#saveFromFileIO(tobedownloaded, prefix=prefix, proxy="10.225.92.1:80")

#initialisation

from uploadfromdatacamp import loadNDArrayFromCsv
X = loadNDArrayFromCsv(prefix+'X.csv')
y = loadNDArrayFromCsv(prefix+'y.csv')

# Import cross_val_score
from sklearn.model_selection import cross_val_score


# ### code

# In[60]:


# Import KerasClassifier from keras wrappers
from keras.wrappers.scikit_learn import KerasClassifier

# Create a KerasClassifier
model = KerasClassifier(build_fn = create_model, epochs = 50, 
             batch_size = 128, verbose = 0)

# Calculate the accuracy score for each fold
kfolds = cross_val_score(model, X, y, cv = 3)

# Print the mean accuracy
print('The mean accuracy was:', kfolds.mean())

# Print the accuracy standard deviation
print('With a standard deviation of:', kfolds.std())


# In[ ]:




