# -*- coding: utf-8 -*-
"""
Created on Tue Mar 27 09:19:34 2018

@author: Liam
"""

#Listing 5.4. Copying images to training, validation, and test directories
import pandas as pd
import os

original_dataset_dir = r'\Users\Liam\Documents\R\files\kaggle\dog_breed_identification'
base_dir = os.path.join(original_dataset_dir, 'subset')
train_dir = os.path.join(base_dir, 'train')
validation_dir = os.path.join(base_dir, 'validation')
test_dir = os.path.join(base_dir, 'test')

"""
#Set up folder structure
#Where uncompressed

#Directory where you’ll store your smaller dataset
if not os.path.exists(base_dir):
    os.mkdir(base_dir)

#Directories for the training, validation, and test splits
if not os.path.exists(train_dir):
    os.mkdir(train_dir)

if not os.path.exists(validation_dir):
    os.mkdir(validation_dir)
    
if not os.path.exists(test_dir):
    os.mkdir(test_dir)

#Read in image labels
labels = pd.read_csv('labels.csv')

#Only use top 16 breeds for this example
top_breeds = labels['breed'].value_counts()[0:16]

#Extract top 16 breeds
unique_breeds = list(top_breeds.index)
"""

num_classes = len(os.listdir(train_dir))
print(num_classes)

#Instantiating the VGG16 convolutional base
from keras.applications import MobileNet

conv_base = MobileNet(weights='imagenet',
                  include_top=False,
                  input_shape=(160, 160, 3))

#Method 2 - Feature extraction with data augmentation
#Much slower and more expensive

#Adding a densely connected classifier on top of the convolutional base
from keras import models
from keras import layers
from keras import callbacks

model = models.Sequential()
model.add(conv_base)
model.add(layers.Flatten())
model.add(layers.Dense(256, activation='relu'))
model.add(layers.Dense(16, activation='softmax'))

model.summary()

#Before you compile and train the model, it’s very important to freeze the convolutional base. 
#Freezing a layer or set of layers means preventing their weights from being updated during training
print('This is the number of trainable weights before freezing the conv base:', len(model.trainable_weights))

conv_base.trainable = False

print('This is the number of trainable weights after freezing the conv base:', len(model.trainable_weights))

#Training the model end to end with a frozen convolutional base
from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers

train_datagen = ImageDataGenerator(
      rescale = 1./255,
      rotation_range = 40,
      width_shift_range = 0.2,
      height_shift_range = 0.2,
      shear_range = 0.2,
      zoom_range = 0.2,
      horizontal_flip = True,
      fill_mode = 'nearest')

test_datagen = ImageDataGenerator(rescale=1./255)                

train_generator = train_datagen.flow_from_directory(
        train_dir,                                               
        target_size=(160, 160),                                  
        batch_size=20,
        class_mode='categorical')                                     

validation_generator = test_datagen.flow_from_directory(
        validation_dir,
        target_size=(160, 160),
        batch_size=20,
        class_mode='categorical')

test_generator = test_datagen.flow_from_directory(
        test_dir,
        target_size=(160, 160),
        batch_size=20,
        class_mode='categorical',
        shuffle = False)

model.compile(loss = 'categorical_crossentropy',
              optimizer = optimizers.RMSprop(lr=2e-5),
              metrics = ['acc'])

def get_callbacks(file_name, patience = 5):
   es = callbacks.EarlyStopping('val_loss', patience = patience, mode = "min")
   msave = callbacks.ModelCheckpoint(file_name, save_best_only = True)
   return [es, msave]

model_callbacks = get_callbacks('transfer_learning_mobile_net.h5', patience = 5)

history = model.fit_generator(
      train_generator,
      epochs = 100,
      validation_data = validation_generator,
      callbacks = model_callbacks)

test_loss, test_acc = model.evaluate_generator(test_generator)
print('test acc:', test_acc)
print('test loss:', test_loss)

#Method 3 - Fine tuning
#Builds on Method 2

#Freezing all layers up to a specific one
conv_base.trainable = True

conv_base.summary()

set_trainable = False

for layer in conv_base.layers:
    if layer.name == 'conv_pw_13':
        set_trainable = True
    if set_trainable:
        layer.trainable = True
    else:
        layer.trainable = False

#Fine-tuning the model - low learning rate to limit magnitude of modifications
model.compile(loss='binary_crossentropy',
              optimizer=optimizers.RMSprop(lr=1e-5),
              metrics=['acc'])

history = model.fit_generator(
      train_generator,
      epochs=100,
      validation_data=validation_generator)

test_loss, test_acc = model.evaluate_generator(test_generator)
print('test acc:', test_acc)
print('test loss:', test_loss)

#Predict on test set
test_preds = model.predict_generator(test_generator)
test_preds_df = pd.DataFrame(test_preds)

#Extract column names
col_names = train_generator.class_indices
#Switch keys and values
col_names = {y:x for x,y in col_names.items()}

test_preds_df = test_preds_df.rename(columns = col_names)

test_preds_df.index = test_generator.filenames

#test_preds_df_class = test_preds_df.idxmax(axis = 'columns')
#print(test_preds_df_class[0:100])

print(test_generator.filenames[0:10])
