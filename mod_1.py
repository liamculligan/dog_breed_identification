# -*- coding: utf-8 -*-
"""
Created on Sun Mar 18 16:35:36 2018

@author: Liam
"""

#Listing 5.4. Copying images to training, validation, and test directories
import pandas as pd
import os

#Set up folder structure
#Where uncompressed
original_dataset_dir = r'\Users\Liam\Documents\R\files\kaggle\dog_breed_identification'

#Directory where you’ll store your smaller dataset
base_dir = os.path.join(original_dataset_dir, 'subset')
if not os.path.exists(base_dir):
    os.mkdir(base_dir)

#Directories for the training, validation, and test splits
train_dir = os.path.join(base_dir, 'train')
if not os.path.exists(train_dir):
    os.mkdir(train_dir)

validation_dir = os.path.join(base_dir, 'validation')
if not os.path.exists(validation_dir):
    os.mkdir(validation_dir)
    
test_dir = os.path.join(base_dir, 'test')
if not os.path.exists(test_dir):
    os.mkdir(test_dir)

#Read in image labels
labels = pd.read_csv('labels.csv')

#Only use top 16 breeds for this example
top_breeds = labels['breed'].value_counts()[0:16]

#Extract top 16 breeds
unique_breeds = list(top_breeds.index)

#Place each breed's image in its own folder
for breed in unique_breeds:
    
    print(breed)
    
    if not os.path.exists(os.path.join(train_dir, breed)):
        os.makedirs(os.path.join(train_dir, breed))
    
    breed_labels = labels.loc[labels['breed'] == breed]
    
    for i, breed_label in breed_labels.iterrows():
        
        print(breed_label['id'])
        
        if os.path.exists('train/' + breed_label['id'] + '.jpg'):
            os.rename('train/' + breed_label['id'] + '.jpg', os.path.join(train_dir, breed) + '/' + breed_label['id'] + '.jpg')

num_classes = len(os.listdir(train_dir))
print(num_classes)

all_folders = os.listdir(train_dir)

import shutil
import numpy as np
    
for folder in all_folders:
    
    if not os.path.exists(os.path.join(validation_dir, folder)):
        os.makedirs(os.path.join(validation_dir, folder))
        
    if not os.path.exists(os.path.join(test_dir, folder)):
        os.makedirs(os.path.join(test_dir, folder))
    
    source = os.path.join(train_dir, folder)
    dest_1 = os.path.join(validation_dir, folder)
    dest_2 = os.path.join(test_dir, folder)
    files = os.listdir(source)

    for f in files:
        random_num = np.random.rand(1)
        if random_num < 0.2:
            shutil.move(source + '/'+ f, dest_1 + '/'+ f)
        if (random_num >= 0.2) and (random_num < 0.4):
            shutil.move(source + '/'+ f, dest_2 + '/'+ f)

print('Number of training images:', sum([len(files) for r, d, files in os.walk(train_dir)]))                                              
print('Number of validation images:', sum([len(files) for r, d, files in os.walk(validation_dir)]))
print('Number of test images:', sum([len(files) for r, d, files in os.walk(test_dir)]))

#Using ImageDataGenerator to read images from directories
from keras.preprocessing.image import ImageDataGenerator
train_datagen = ImageDataGenerator(
    rescale = 1./255,
    rotation_range = 40,
    width_shift_range = 0.2,
    height_shift_range = 0.2,
    shear_range = 0.2,
    zoom_range = 0.2,
    horizontal_flip = True)

val_test_datagen = ImageDataGenerator(rescale = 1./255)               

batch_size = 20

train_generator = train_datagen.flow_from_directory(
        train_dir,                                              
        target_size = (150, 150),                                 
        batch_size = batch_size,
        class_mode = 'categorical')                                    

validation_generator = val_test_datagen.flow_from_directory(
        validation_dir,
        target_size = (150, 150),
        batch_size = batch_size,
        class_mode = 'categorical')

test_generator = val_test_datagen.flow_from_directory(
        test_dir,
        target_size = (150, 150),
        batch_size = batch_size,
        class_mode = 'categorical')

"""
#Setting up a data augmentation configuration via ImageDataGenerator
datagen = ImageDataGenerator(
      rotation_range=40,
      width_shift_range=0.2,
      height_shift_range=0.2,
      shear_range=0.2,
      zoom_range=0.2,
      horizontal_flip=True,
      fill_mode='nearest')

#Displaying some randomly augmented training images
from keras.preprocessing import image                           

fnames = [os.path.join(train_cats_dir, fname) for
     fname in os.listdir(train_cats_dir)]

img_path = fnames[3]                                            

img = image.load_img(img_path, target_size=(150, 150))          

x = image.img_to_array(img)                                     
x = x.reshape((1,) + x.shape)                                   

i = 0                                                           
for batch in datagen.flow(x, batch_size=1):                     
    plt.figure(i)                                               
    imgplot = plt.imshow(image.array_to_img(batch[0]))          
    i += 1                                                      
    if i % 4 == 0:                                              
        break                                                   

plt.show()
"""

#Model
from keras import layers
from keras import models
from keras import optimizers
from keras import callbacks

def get_callbacks(file_name, patience = 5):
   es = callbacks.EarlyStopping('val_loss', patience = patience, mode = "min")
   msave = callbacks.ModelCheckpoint(file_name, save_best_only = True)
   return [es, msave]

model_callbacks = get_callbacks('simple_cnn.h5', patience = 5)

def get_model():
    model = models.Sequential()
    
    model.add(layers.Conv2D(32, (3, 3), activation = 'relu', input_shape = (150, 150, 3)))
    model.add(layers.MaxPooling2D((2, 2)))
    
    model.add(layers.Conv2D(64, (3, 3), activation = 'relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    
    model.add(layers.Conv2D(128, (3, 3), activation = 'relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    
    model.add(layers.Conv2D(128, (3, 3), activation = 'relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    
    model.add(layers.Flatten())
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(512, activation = 'relu'))
    model.add(layers.Dense(num_classes, activation = 'softmax'))
    
    model.compile(loss = 'categorical_crossentropy',
                  optimizer = optimizers.RMSprop(lr=1e-4),
                  metrics = ['acc'])

    return model

#Model
model = get_model()

model.summary()

history = model.fit_generator(
      train_generator,
      epochs = 100,
      validation_data = validation_generator,
      callbacks = model_callbacks)

#Plot performance
import matplotlib.pyplot as plt

acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(1, len(acc) + 1)

plt.plot(epochs, acc, 'bo', label = 'Training acc')
plt.plot(epochs, val_acc, 'b', label = 'Validation acc')
plt.title('Training and validation accuracy')
plt.legend()

plt.figure()

plt.plot(epochs, loss, 'bo', label = 'Training loss')
plt.plot(epochs, val_loss, 'b', label = 'Validation loss')
plt.title('Training and validation loss')
plt.legend()

plt.show()

#Predict on test set
test_preds = model.predict_generator(test_generator)
test_preds_df = pd.DataFrame(test_preds)

#Extract column names
col_names = train_generator.class_indices
#Switch keys and values
col_names = {y:x for x,y in col_names.items()}

test_preds_df = test_preds_df.rename(columns = col_names)

test_preds_df_class = test_preds_df.idxmax(axis = 'columns')

"""
from sklearn.metrics import accuracy_score

actual_target = test_generator.classes

from sklearn.preprocessing import OneHotEncoder
enc = OneHotEncoder(sparse=False)

actual_target_ohe = actual_target.reshape(len(actual_target), 1)
actual_target_ohe = enc.fit_transform(actual_target_ohe)

accuracy_score(actual_target, test_preds, normalize=True, sample_weight=None)
"""