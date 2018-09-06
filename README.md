ANN with Keras
================

Building an **artificial deep neural network** using **keras** and Python ... We will be using the Forest Cover Type dataset at <https://archive.ics.uci.edu/ml/machine-learning-databases/covtype/> to predict forest cover type ...

``` r
# run the script from command line

$ python forest_type_keras.py

### READING DATA ###
# data shape
(581012, 55)

# Class column values and length
[5 2 1 7 3 6 4] , 7  Classes

# Encoding and scaling data

### BUILDING THE MODEL ###
# (1, 54) input => 500 nodess => 0.1 dropout => 500 nodes => 0.1 dropout => 8 outputs (from 0:7)
# Compiling the model
# Fitting the model on training data
Epoch 1/10
389278/389278 [==============================] - 63s 163us/step - loss: 0.6065 - acc: 0.7328
Epoch 2/10
389278/389278 [==============================] - 62s 160us/step - loss: 0.4779 - acc: 0.7918
# .........
Epoch 10/10
389278/389278 [==============================] - 63s 161us/step - loss: 0.3104 - acc: 0.8708

# Evaluating the model on test data
loss     : 0.2739101243541741
accuracy : 0.888136689371112

# Predict new values
actual data   :  [1, 5, 1, 3, 2, 7, 2, 6, 2, 7]
predicted data:  [1, 5, 1, 6, 2, 7, 2, 6, 2, 7]

```
