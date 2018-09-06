import pandas as pd
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense, Dropout
print("### READING DATA ###")
data = pd.read_csv('covtype.data.gz', compression='gzip', header=None, sep=',')
# to delete rows with any unknown column value
# unknown_indx = data.apply(
#     lambda x: x.astype(str).str.contains(r'\?.*').any(),
#     axis = 1)
# sum(unknown_indx)
# data = data[~unknown_indx]
print("# data shape")
print(data.shape)
print("")
print("# Class column values and length")
print(pd.unique(data.iloc[:, 54]), ",", len(pd.unique(data.iloc[:, 54])), " Classes")
print("")
print("# Encoding and scaling data")
x = data.iloc[:, 0:54]
y = to_categorical(data.iloc[:, 54]) # to_categorical(data.iloc[:, 54] - 1) to have classes from 0:6 not 1:7
x = x.apply(lambda x: x/x.max(), axis = 0)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33)
print("")
print("### BUILDING THE MODEL ###")
model = Sequential()
print("# (1, 54) input => 500 nodess => 0.1 dropout => 500 nodes => 0.1 dropout => 8 outputs (from 0:7)")
model.add(Dense(500, activation = 'relu', input_shape = (54,))) # 54 is the number of variables in each obs.
model.add(Dropout(0.1))
model.add(Dense(500, activation = 'relu'))
model.add(Dropout(0.1))
model.add(Dense(8, activation = 'softmax')) # outputs starting from 0 so y is from 0:7 and 7 if y is from 0:6
print("# Compiling the model")
model.compile(loss = 'categorical_crossentropy',
              optimizer = 'adam',
              metrics = ['accuracy'])
print("# Fitting the model on training data")
model.fit(x_train, y_train,
          batch_size = 32, epochs = 10, verbose = 1)
print("# Evaluating the model on test data")
loss, accuracy = model.evaluate(x_test, y_test, verbose = 0)
print("loss    ", ":", loss)
print("accuracy", ":", accuracy)
print("")
print("# Predict new values")
predictions = model.predict(x_test)
print("actual data   : ", [y_test[i].argmax(axis = 0) for i in range(0, 10)])
print("predicted data: ", [predictions[i].argmax(axis = 0) for i in range(0, 10)])
