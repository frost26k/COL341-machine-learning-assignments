from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten, Activation, LeakyReLU
from keras.layers import Dense, Dropout, Concatenate
from keras.layers.normalization import BatchNormalization
from keras import optimizers
from keras.preprocessing.image import ImageDataGenerator
from keras import regularizers
from keras.layers import Input
import numpy as np
import keras.initializers
import pandas as pd
import sys


def parta(tr,te,op):
        train = pd.read_csv(tr,header=None, delimiter=' ')
        X = train.values.copy()
        Xtrain = X[:,0:X.shape[1]-1].copy()
        Ytrain = X[:,X.shape[1]-1].copy()

        Ytrainmat = np.zeros((50000,10))

        for i in range(Ytrain.shape[0]):
            Ytrainmat[i,Ytrain[i]]=1

        images = np.zeros((Xtrain.shape[0],32,32,3))
        for i in range(Xtrain.shape[0]):
            images[i,:,:,:] = np.transpose(Xtrain[i,:].reshape(3,32,32),(1,2,0))
        images = images.astype(np.uint8)




        test = pd.read_csv(te,header=None, delimiter=' ')
        XF = test.values.copy()
        Xtest = XF[:,0:XF.shape[1]-1].copy()
        imagestest = np.zeros((Xtest.shape[0],32,32,3))
        for i in range(Xtest.shape[0]):
            imagestest[i,:,:,:] = np.transpose(Xtest[i,:].reshape(3,32,32),(1,2,0))
        imagestest = imagestest.astype(np.uint8)


        mean = np.mean(images,axis=(0,1,2,3))
        std = np.std(images,axis=(0,1,2,3))
        images = (images-mean)/(std+1e-7)
        imagestest = (imagestest-mean)/(std+1e-7)



        classifier = Sequential()
        weight_decay = 1e-4
        classifier.add(Conv2D(1024, (3, 3), strides=1, padding='same', input_shape = (32, 32, 3), activation = 'relu'))
        classifier.add(BatchNormalization())
        classifier.add(MaxPooling2D(pool_size = (2, 2)))
        classifier.add(Dropout(0.3))

        classifier.add(Conv2D(512, (3, 3), strides=1, padding='same', activation = 'relu', kernel_initializer='he_uniform'))
        classifier.add(Conv2D(512, (3, 3), strides=1, padding='same', activation = 'relu', kernel_initializer='he_uniform'))
        classifier.add(BatchNormalization())
        classifier.add(MaxPooling2D(pool_size = (2, 2)))
        classifier.add(Dropout(0.2))

        classifier.add(Conv2D(256, (3, 3), strides=1, padding='same', activation = 'relu', kernel_initializer='he_uniform'))
        classifier.add(BatchNormalization())
        classifier.add(MaxPooling2D(pool_size = (2, 2)))
        classifier.add(Dropout(0.3))

        classifier.add(Conv2D(256, (3, 3), strides=1, padding='same', activation = 'relu', kernel_initializer='he_uniform'))
        classifier.add(Conv2D(256, (3, 3), strides=1, padding='same', activation = 'relu', kernel_initializer='he_uniform'))
        classifier.add(BatchNormalization())
        classifier.add(MaxPooling2D(pool_size = (2, 2)))
        classifier.add(Dropout(0.3))

        classifier.add(Conv2D(1024, (3, 3), strides=1, padding='same', activation = 'relu', kernel_initializer='he_uniform'))
        classifier.add(Conv2D(128, (3, 3), strides=1, padding='same', activation = 'relu', kernel_initializer='he_uniform'))
        classifier.add(BatchNormalization())
        classifier.add(MaxPooling2D(pool_size = (2, 2)))

        classifier.add(Flatten())
        classifier.add(Dense(units = 1024, activation = 'relu'))
        classifier.add(Dense(units = 512, activation = 'relu'))
        classifier.add(Dense(units = 128, activation = 'relu'))
        classifier.add(BatchNormalization())

        classifier.add(Dense(units = 10, activation = 'softmax'))
        opt = optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
        #opt = optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
        classifier.compile(optimizer = opt, loss = 'categorical_crossentropy', metrics = ['accuracy'])



        classifier.fit(images, Ytrainmat, validation_split= 0.2, epochs=15)
        predictions = classifier.predict(imagestest)
        predicted = np.argmax(predictions, axis=1)




        for o in predicted:
            print(o,file=open(op,"a"))


if __name__ == '__main__':
    parta(*sys.argv[1:])
