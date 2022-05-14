
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
#import seaborn as sn
#import random
#import matplotlib.cm as cm

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


def evaluate(model_path="./my_model.h5",validation_path= 'small_dataset/val', batch_size= 10,
           image_size= (224,224) ):
    """
        models folder
        target_dir: model path        

    """
    # load models
    model =  load_model(model_path)
    #validation gene
    validation_datagen = ImageDataGenerator(rescale=1./255) # attention !!!!!!
    validation_generator = validation_datagen.flow_from_directory(
                  validation_path,
                  target_size=image_size,
                  batch_size=batch_size,
                  shuffle=False,
                  )

    Classes_names =list(validation_generator.class_indices.keys())


    validation_generator.reset()
    # predict the validation dataset
    results = model.predict(validation_generator)
    results = np.array(results)
    predicted_labels_idx = np.argmax(results,axis=1)
    predicted_labels_names = np.array([Classes_names[label] for label in predicted_labels_idx])
  
    # get the real label of validation dataset
    real_labels_idx = validation_generator.labels
    real_labels_names = np.array([Classes_names[label] for label in real_labels_idx])
 
    print("------------------------classification report--------------------------")
    print(classification_report(real_labels_names,predicted_labels_names,labels=Classes_names))


    #build confusion matrix df
    conf_matrix = confusion_matrix(real_labels_names,predicted_labels_names,labels=Classes_names,normalize='pred')
    conf_matrix_df = pd.DataFrame(conf_matrix, index = Classes_names,columns = Classes_names)


    #plot confusion matrix 

    plt.figure(figsize = (20,20))
    heatmap = sn.heatmap(conf_matrix_df, annot=False,fmt='.2f',cmap="OrRd",vmin=0, vmax=1)
    heatmap.set_xlabel('Predicted disease',fontsize = 10,fontweight="bold")
    heatmap.set_ylabel('Real disease',fontsize = 10,fontweight="bold")
    heatmap.figure.savefig("Confusion matrix.png")


if __name__ == '__main__':
    
    with open("parameters.json") as f:
      pars = json.load(f)
      eval_args = pars["evaluation"]
    print("your paramater loaded form paramaters.json: ")
    print(eval_args)
    
    validation_path= eval_args["validation_path" ]#'small_dataset/val'
    batch_size= eval_args["batch_size" ] #10
    image_size=  string_to_tuple (eval_args["image_size"])# (224,224)
    model_path= eval_args["model_path"] #model_path"my_model.h5"
    # load models
    model = load_model(model_path)
 
    evaluate(model_path ,validation_path , batch_size , image_size )



