#importing libraries
from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers import Dense
import matplotlib.pyplot as plt

#initializing the sequential
classifier =Sequential()

#step 1&2: Apply convolution and Maxpooling
classifier.add(Convolution2D(32,5,5,input_shape=(64,64,3),activation='relu'))
classifier.add(MaxPooling2D(pool_size=(1,1)))

classifier.add(Convolution2D(32,5,5,input_shape=(50,50,3),activation='relu'))
classifier.add(MaxPooling2D(pool_size=(1,1)))

classifier.add(Convolution2D(32,5,5,input_shape=(46,46,3),activation='relu'))
classifier.add(MaxPooling2D(pool_size=(1,1)))

classifier.add(Convolution2D(32,3,3,input_shape=(44,44,3),activation='relu'))
classifier.add(MaxPooling2D(pool_size=(1,1)))

classifier.add(Convolution2D(32,3,3,input_shape=(42,42,3),activation='relu'))
classifier.add(MaxPooling2D(pool_size=(1,1)))

classifier.add(Convolution2D(32,3,3,input_shape=(40,40,3),activation='relu'))
classifier.add(MaxPooling2D(pool_size=(1,1)))

#Dropout is a regularisation technique which is used to prevent overfitting. it is a technique in which randomly selected neurons are neglected.
classifier.add(Dropout(0.25))

#step 3:Apply Flattening
classifier.add(Flatten())

#step 4:Apply full connection
classifier.add(Dense(output_dim=128,activation='relu'))
classifier.add(Dropout(0.5))
classifier.add(Dense(output_dim=3,activation='softmax'))

#compiling the cnn
classifier.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])

#fitting the images into cnn model
from keras.preprocessing.image import ImageDataGenerator
train_datagen=ImageDataGenerator(rescale=1./255)   #shear_range=0.2,zoom_range=0.2,horizontal_flip=True

test_datagen=ImageDataGenerator(rescale=1./255)

train_set=train_datagen.flow_from_directory('C:/Users/asus/OneDrive/Desktop/Mini_project/Mini_project_2/dataset5_testing/Train',shuffle=True,target_size=(64,64),class_mode='categorical',batch_size=16) #classes=['open','closed']

test_set=test_datagen.flow_from_directory('C:/Users/asus/OneDrive/Desktop/Mini_project/Mini_project_2/dataset5_testing/Test',shuffle=True,target_size=(64,64),class_mode='categorical',batch_size=16)
BS=16
SPE= len(train_set.classes)//BS
VS = len(test_set.classes)//BS
print(SPE , VS)

h=classifier.fit_generator(train_set,samples_per_epoch=SPE,nb_epoch=54,validation_data=test_set,nb_val_samples=VS)

#to plot training and validation accuracy values
plt.plot(h.history['accuracy'])
plt.plot(h.history['val_accuracy'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()

#to plot training and validation loss values
plt.plot(h.history['loss'])
plt.plot(h.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()

classifier.save("Target2.h5")