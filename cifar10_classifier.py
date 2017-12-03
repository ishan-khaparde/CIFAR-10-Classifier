import keras

from keras.models import Sequential
from keras.utils import np_utils
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Dense, Activation, Flatten, Dropout, BatchNormalization
from keras.layers import Conv2D, MaxPooling2D
from keras.datasets import cifar10
from keras.models import load_model
from keras import regularizers, optimizers
from pathlib import Path
from PIL import Image

import numpy as np

(x_train, y_train), (x_test, y_test) = cifar10.load_data()
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
label_vector = ['airplane','automobile','bird','cat','deer','dog','frog','horse','ship','truck']

mean = np.mean(x_train,axis=(0,1,2,3))
std = np.std(x_train,axis=(0,1,2,3))

x_train = (x_train-mean)/(std+1e-7)
x_test = (x_test-mean)/(std+1e-7)

num_classes = 10
y_train = np_utils.to_categorical(y_train,num_classes)
y_test = np_utils.to_categorical(y_test,num_classes)
baseMapNum = 32
weight_decay = 1e-4

#Building the CNN
model = Sequential()

model.add(Conv2D(baseMapNum, (3,3), padding='same', kernel_regularizer=regularizers.l2(weight_decay), input_shape=x_train.shape[1:]))
model.add(Activation('relu'))
model.add(BatchNormalization())
model.add(Conv2D(baseMapNum, (3,3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
model.add(Activation('relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.2))

model.add(Conv2D(2*baseMapNum, (3,3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
model.add(Activation('relu'))
model.add(BatchNormalization())
model.add(Conv2D(2*baseMapNum, (3,3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
model.add(Activation('relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.3))

model.add(Conv2D(4*baseMapNum, (3,3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
model.add(Activation('relu'))
model.add(BatchNormalization())
model.add(Conv2D(4*baseMapNum, (3,3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
model.add(Activation('relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.4))

model.add(Flatten())
model.add(Dense(num_classes, activation='softmax'))
print(x_train.shape)
model.summary()

#data augmentation
datagen = ImageDataGenerator(
    featurewise_center=False,
    samplewise_center=False,
    featurewise_std_normalization=False,
    samplewise_std_normalization=False,
    zca_whitening=False,
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True,
    vertical_flip=False
    )

datagen.fit(x_train)

model_file_path = "cifar10_model.ep100.h5"
model_file = Path(model_file_path)

if model_file.is_file():
    
    loaded_model = load_model(model_file_path)
    scores = loaded_model.evaluate(x_test,y_test,batch_size = 128,verbose = 0)
    print('\nTest result: %.3f loss: %.3f' % (scores[1]*100,scores[0]))
    img = Image.open("C:/Users/ishan/Study/CS 688/Project/dog_test.jpg")
    img = img.resize((32,32),resample = 0)

    img_array = np.reshape(img, (1,32,32,3))
    
    img_array = img_array.astype('float32')
    
   # for i in range(0,10):
    predictions = model.predict(img_array,batch_size=128,verbose = 0)
    label = model.predict_classes(img_array,batch_size=128,verbose = 0)

    print(label_vector[label[0]])
    print(predictions)

else: 
    #training
    batch_size = 64
    epochs = 10
    opt_rms = keras.optimizers.rmsprop(lr=0.001,decay=1e-6)
    
    model.compile(loss='categorical_crossentropy',
        optimizer=opt_rms,
        metrics=['accuracy'])
    model.fit_generator(datagen.flow(x_train, y_train, batch_size=batch_size),steps_per_epoch=x_train.shape[0] // batch_size,epochs=epochs,verbose=1,validation_data=(x_test,y_test))
    model.save('cifar10_model.ep10.h5')
    
    #validation
    scores = model.evaluate(x_test, y_test, batch_size=128, verbose=0)
    print('\nTest result: %.3f loss: %.3f' % (scores[1]*100,scores[0]))
    labels = model.predict_classes(x_test,batch_size=32)
    print(labels)
