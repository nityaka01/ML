# ML
DL/AI Projects
import tensorflow as tf 
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential 
from tensorflow.keras.layers import Dense, Conv2D, MaxPool2D, BatchNormalization, Dropout, Flatten 
from tensorflow.keras.utils import to_categorical 
import numpy as np 
import matplotlib.pyplot as plt 

(X_train,y_train),(X_test,y_test) = mnist.load_data()
#getting dimensions of array
print(X_train.shape)
print(y_train.shape) #rank one array
print(X_test.shape)
print(y_test.shape) #rank one array

def view_training_data(idx):
    plt.imshow(X_train[idx])
    print(y_train[idx])
idx = 10 
view_training_data(idx)

#Normalize and reshape 
X_train = X_train.reshape(-1,28,28,1) #reshape images into 28x28
X_train = X_train.astype('float32') #necessary before dividing by 255
X_train/=255 #Normalize
y_train = to_categorical(y_train) #hot encoding sparse vs hot encoding categorical cross entropy

X_test = X_test.reshape(-1,28,28,1)
X_test = X_test.astype('float32')
X_test/=255
y_test = to_categorical(y_test)

#developing model
#adding 2 convolutional layers, 2 pooling layers, 2 dense layers 
model = Sequential()
model.add(Conv2D(16,(3,3), activation = 'relu'))
model.add(MaxPool2D(2,2))
model.add(Conv2D(16,(3,3), activation = 'relu'))
model.add(MaxPool2D(2,2))

model.add(Flatten()) #Allows for input into dense/fully connected network
model.add(Dense(64,activation = 'relu'))
model.add(Dense(10, activation = 'softmax'))

#learning and training model
model.compile(optimizer = 'sgd', loss = 'categorical_crossentropy', metrics = ['accuracy']) #helps decide optimizer as well as loss function
history = model.fit(X_train, y_train, epochs = 5, batch_size = 32, validation_split = 0.1) #this helps determine the number of epochs we aim to drop after each batch size of 32 images. e.g: train on 32 images, drop epochs

def predict(X_test,idx):
    test_image = X_test[idx] #determines image 
    plt.imshow(test_image.reshape(28,28)) #shows image beforehand for us to predict
    preds = np.argmax(model.predict(test_image.reshape(-1,28,28,1))) #reshapes image and predicts 
    print(preds) #prints prediction
