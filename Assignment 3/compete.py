
import keras
from keras import optimizers
from keras.utils import to_categorical
import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import Conv2D, Activation, Flatten
from keras.layers import MaxPooling2D
from keras.layers import Dense
from sys import argv



# main code
df = pd.read_csv(argv[1],header = None)
df = df.values
temp = []
for i in df:
  q = np.fromstring(i[0],dtype = 'int', sep = ' ')
  temp.append(q)
df = np.asarray(temp)
test = pd.read_csv(argv[2],header = None)
test = test.values
temp = []
for i in test:
  q = np.fromstring(i[0],dtype = 'int', sep = ' ')
  temp.append(q)
test = np.asarray(temp)
col = df.shape[1]
X = df[:,0:col-1]
Y = df[:,col-1]
col = test.shape[1]
test = test[:,0:col-1]
# pre processing

t1 = np.copy(X)
t2 = np.copy(test)

X = np.copy(t1)
test = np.copy(t2)

temp = []
for i in X:
  r = i[:1024]
  r = r.reshape(32,32)
  g = i[1024:2048]
  g = g.reshape(32,32)
  b = i[2048:3072]
  b = b.reshape(32,32)
  q = [r,g,b]
  temp.append(q)
X = np.asarray(temp)
temp = []
for i in test:
  r = i[:1024]
  r = r.reshape(32,32)
  g = i[1024:2048]
  g = g.reshape(32,32)
  b = i[2048:3072]
  b = b.reshape(32,32)
  q = [r,g,b]
  temp.append(q)
test = np.asarray(temp)
# one hot encoding
y = to_categorical(Y)

import matplotlib.pyplot as plt
X = X.transpose([0,2,3,1])
test = test.transpose([0,2,3,1])

X = (X.astype('float32'))/255
test = (test.astype('float32'))/255

from keras.preprocessing.image import ImageDataGenerator
datagen = ImageDataGenerator(width_shift_range=0.1, height_shift_range=0.1, horizontal_flip=True)
it_train = datagen.flow(X, y, batch_size=1000)

from keras.layers import Input
from keras.layers import Dropout, concatenate
input_img = Input(shape = (32, 32, 3))
from keras.layers import Conv2D, MaxPooling2D


from keras.layers import BatchNormalization
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Flatten, Dense
model = Sequential()
model.add(Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same', input_shape=(32, 32, 3)))
model.add(Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
model.add(BatchNormalization())
model.add(MaxPooling2D((2, 2)))
model.add(Dropout(0.2))
model.add(Conv2D(256, (5, 5), activation='relu', kernel_initializer='he_uniform', padding='same'))
model.add(Conv2D(256, (5, 5), activation='relu', kernel_initializer='he_uniform', padding='same'))
model.add(BatchNormalization())
model.add(MaxPooling2D((2, 2)))
model.add(Dropout(0.3))
model.add(Conv2D(512, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
model.add(Conv2D(512, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
model.add(BatchNormalization())
model.add(MaxPooling2D((2, 2)))
model.add(Dropout(0.4))
model.add(Flatten())
model.add(Dense(1024, activation='relu', kernel_initializer='he_uniform'))
model.add(Dropout(0.5))
model.add(Dense(512, activation='relu', kernel_initializer='he_uniform'))
model.add(Dense(100))
model.add(BatchNormalization())
model.add(Activation('softmax'))
opt = keras.optimizers.Adam(lr=0.005, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
model.compile(loss='categorical_crossentropy', optimizer = opt, metrics=['accuracy'])
model.fit_generator(it_train,steps_per_epoch = 50,epochs = 200)

y_ans = model.predict(test)
a = []
for i in y_ans:
    a.append(np.argmax(i))
np.savetxt(argv[3],a,newline='\n')
