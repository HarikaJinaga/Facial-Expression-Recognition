import cv2
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense , Activation , Dropout ,Flatten
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.metrics import categorical_accuracy
from keras.models import model_from_json
from keras.optimizers import *
from keras.layers.normalization import BatchNormalization

data = pd.read_pickle("final_data.pkl")

print(data.shape)

data = data[data['X'] != None]

print(data.shape)

data_nosketch = data[data['sketch'] == 0]
data_nosketch
data_nosketch.shape

def return_dataset(data, class_name):
    for index, row in data.iterrows():
        
        vec = data['X'][index]
        
      
        if index == 0:
            X = vec
            y = np.reshape(data[class_name][index], (1, 1))

        else:
            X = np.concatenate((X, vec), axis=1)
            y = np.concatenate((y, np.reshape(data[class_name][index], (1, 1))), axis=0)
            
    return X, y
X = data['X']
y = data['expression']


def encode(data):
    print('Shape of data (BEFORE encode): %s' % str(data.shape))
    encoded = to_categorical(data)
    print('Shape of data (AFTER  encode): %s\n' % str(encoded.shape))
    return encoded
  
encoded_y = encode(y)
print(encoded_y)



#print(y)

X_train, X_test, y_train, y_test = train_test_split(X, encoded_y, test_size = 0.1, random_state = 42)
print(y_train.shape)
print(X_train.shape)

index = 0

X_list = list(X_train)
y_list = list(y_train)

X_list_n = []
X_test_n = []

for example in X_list:
  index = index + 1
  try:
    if (example.shape != (48, 48, 3)):
      print(index)
  except:
    print(X_list[index - 1])
    print(index - 1)
    X_list.pop(index - 1)
    y_list.pop(index - 1)
    
for example in X_list:
  gray = cv2.cvtColor(example, cv2.COLOR_BGR2GRAY)
  X_list_n.append(gray)

  
  
for example in list(X_test):
  gray = cv2.cvtColor(example, cv2.COLOR_BGR2GRAY)
  X_test_n.append(gray)
    
    
print(len(X_list))
print(len(y_list))

X_t = np.stack(X_list_n, axis=0)
y_t = np.stack(y_list, axis=0)

X_tes = np.stack(list(X_test_n), axis=0)
y_tes = np.stack((y_test), axis=0)

X_t = np.reshape(X_t, (X_t.shape[0], X_t.shape[1], X_t.shape[2], 1))
X_tes = np.reshape(X_tes, (X_tes.shape[0], X_tes.shape[1], X_tes.shape[2], 1))

print(y_tes.shape)

from skimage import data

coins = data.coins()

print(type(coins), coins.dtype, coins.shape)
plt.imshow(coins, cmap='gray', interpolation='nearest');

print(y_tes.shape)
print(X_tes.shape)

print("-----------")

print(type(X_tes[0]))
print(X_tes[0].dtype)
print(X_tes[0].reshape((X_tes[0].shape[0], X_tes[0].shape[1])).shape)

plt.imshow(X_t[32].reshape((X_tes[0].shape[0], X_tes[0].shape[1])), cmap='gray')
plt.show()

print(y_t[32])

model = Sequential()

# 1 - Convolution
model.add(Conv2D(64,(3,3), padding='same', input_shape=(48, 48,1)))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

# 2nd Convolution layer
model.add(Conv2D(128,(5,5), padding='same'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

# 3rd Convolution layer
model.add(Conv2D(512,(3,3), padding='same'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))


# Flattening
model.add(Flatten())

# Fully connected layer 1st layer
model.add(Dense(256))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dropout(0.25))


# Fully connected layer 2nd layer
model.add(Dense(512))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dropout(0.25))

model.add(Dense(7, activation='sigmoid'))

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=[categorical_accuracy])
#train the model
history = model.fit(X_t, y_t,
              batch_size=64,
              epochs=30,
              verbose=1,
              validation_split=0.1111)
from sklearn.metrics import classification_report, confusion_matrix
 
pred_list = []; actual_list = []

predictions = model.predict(X_tes)

for i in predictions:
    pred_list.append(np.argmax(i))

for i in y_tes:
    actual_list.append(np.argmax(i))

confusion_matrix(actual_list, pred_list)

predictions.shape

np.sum(np.sum(confusion_matrix(actual_list, pred_list), axis=1), axis=0)

print(history.history.keys())
# summarize history for accuracy
plt.plot(history.history['categorical_accuracy'])
plt.plot(history.history['val_categorical_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
