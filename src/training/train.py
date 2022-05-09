import tensorflow as tf
print(tf.__version__)
 
from tensorflow.keras.applications import VGG16
 
from tensorflow.keras.models import Model,load_model
from tensorflow.keras.layers import Input,GlobalAveragePooling2D,Dropout,Dense
 
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing.image import load_img,img_to_array,array_to_img
from tensorflow.keras.optimizers import SGD
from tensorflow.keras import optimizers
from tensorflow.keras.utils import plot_model
import json

import glob
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.metrics import confusion_matrix,classification_report
import seaborn as sn
import random
import matplotlib.cm as cm

def string_to_tuple(txt):
    txt=txt.replace("(", "")
    txt=txt.replace(")", "")
    txt=txt.replace(" ", "")
    txt= txt.split(",")
    t=[]
    for i in txt:
	    t.append(int(i))
    
    t=tuple(t)
    return t



def train( train_path= 'small_dataset/train', validation_path= 'small_dataset/val', batch_size= 10,
           image_size= (224,224), learning_rate= 1e-3, momentum=0.9, loss='categorical_crossentropy', metrics = ['accuracy']
           ,nbr_epochs= 1 ) :
 
  
  train_datagen = ImageDataGenerator(rescale=1./255)
  train_generator = train_datagen.flow_from_directory(
                  train_path,
                  target_size=image_size,
                  batch_size=batch_size
                  )

  validation_datagen = ImageDataGenerator(rescale=1./255) # attention !!!!!!
  validation_generator = validation_datagen.flow_from_directory(
                  validation_path,
                  target_size=image_size,
                  batch_size=batch_size,
                  shuffle=False,
                  )

  class_number = train_generator.num_classes

  # Downloading the Pretrained Model
  base_model = VGG16(include_top =False,input_shape = (224,224,3))


  #Adding New Layers
  x = base_model.output
  x = GlobalAveragePooling2D()(x)
  x = Dropout(0.3)(x)
  x = Dense(256, activation='relu')(x)
  predictions = Dense(class_number, activation='softmax')(x)
  model = Model(base_model.input, predictions)


  #Compiling the Model
  model.compile(optimizer=optimizers.SGD(learning_rate, momentum), loss= loss, metrics= metrics )



  # Train the model
  history=model.fit(train_generator,
          epochs=nbr_epochs,
          validation_data=validation_generator
  )

if __name__ == '__main__':

  with open("parameters.json") as f:
      pars = json.load(f)
      train_args = pars["training"]
  
  print(train_args)
  
  train_path= train_args["train_path"] 
  validation_path = train_args["validation_path"] 
  batch_size = train_args["batch_size"] 
  image_size= string_to_tuple (train_args["image_size"] ) 
  learning_rate = train_args["learning_rate"] 
  momentum =train_args["momentum"] 
  loss =train_args["loss"] 
  metrics = train_args["metrics"] 
  nbr_epochs =train_args["nbr_epochs"] 


    
  train(train_path , validation_path , batch_size , image_size , learning_rate , momentum  , loss  , metrics ,nbr_epochs)









 
