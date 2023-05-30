from keras.layers import Input, Lambda, Dense, Flatten
from keras.models import Model
from keras.applications.vgg16 import VGG16
from keras.applications.vgg16 import preprocess_input
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator #dataaugmentation
from keras.models import Sequential
import numpy as np
from glob import glob
import matplotlib.pyplot as plt
from zipfile import ZipFile
import tensorflow as tf

IMAGE_SIZE = [224,224]

train_path = r'F:\University\AI2\traitementImage\ProjetFRTest\images\train/'
valid_path = r'F:\University\AI2\traitementImage\ProjetFRTest\images\validation/'

#add preprocessing layer to the front of VGG excluding top layer cuz it has 1000 of classes, 3 RGB
vgg = VGG16(input_shape=IMAGE_SIZE+[3], weights='imagenet', include_top=False) #using des poids preentraine imagenet

#dont train existing weights
for layer in vgg.layers:
    layer.trainable=False

#useful for getting number of classes in our folder (we add the last layer according to number of classes)
folders=glob(r'drive/MyDrive/ProjetFaceRecognition/DataSetRH/train/*')

#making our own layers
x = Flatten()(vgg.output) #convertir les sorties to one dimension
prediction = Dense(len(folders), activation='softmax')(x) #creation de couche entierement connecte (x est input de dense)

#creating the model
model= Model(inputs=vgg.input, outputs= prediction) #the whole neural network

model.summary()

#training
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy']) #fonction de perte, 

train_datagen = ImageDataGenerator(rescale=1./255, shear_range=0.2, zoom_range=0.2, horizontal_flip=True) #data variabilite augmentation

test_datagen = ImageDataGenerator(rescale=1./255) #redimensionnement des valeurs de validation

#creating the training set
#charger les images et appliquer les transformations

training_set = train_datagen.flow_from_directory(r'drive/MyDrive/ProjetFaceRecognition/DataSetRH/train/', target_size=(224,224), batch_size=20, class_mode='categorical')

test_set = test_datagen.flow_from_directory(r'drive/MyDrive/ProjetFaceRecognition/DataSetRH/validation/', target_size=(224,224), batch_size=20, class_mode='categorical')


#start the training
r = model.fit_generator(
    training_set, validation_data=test_set, epochs=20, steps_per_epoch=len(training_set), validation_steps=len(test_set)
) 

plt.plot(r.history['loss'], label='train loss')
plt.plot(r.history['val_loss'], label='validation loss')
plt.legend()
plt.show()

plt.plot(r.history['accuracy'], label='train acc')
plt.plot(r.history['val_accuracy'], label='validation acc')
plt.legend()
plt.show()

