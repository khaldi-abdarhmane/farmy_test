from tensorflow.keras.optimizers import SGD
from tensorflow.keras import optimizers
import yaml
import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import pandas as pd
from tensorflow.keras.models import Model,load_model
import pandas as pd
import matplotlib.pyplot as plt
import string
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix,classification_report

class Params(object):
      def __init__(self,
                   
                   rescale=255,
                   image_size=(224,224),
                   batch_size=10,
                   
                   nbr_epoch=3,
                   loss='categorical_crossentropy',
                   metrics = ['accuracy'],
                   learning_rate= 1e-3,
                   momentum=0.9,
                   include_top=False,
                   input_shape_str=""
                                  ):
                self.rescale=rescale
                self.image_size=tuple(map(int, image_size[1:-1].split(',')))
                self.batch_size=batch_size
                
                self.nbr_epoch=nbr_epoch
                self.loss=loss
                self.metrics=metrics
                self.learning_rate=learning_rate
                self.momentum=momentum
                
                self.input_shape = tuple(map(int, input_shape_str[1:-1].split(',')))
                self.include_top=include_top
                self.optimizer=optimizers.SGD(learning_rate=learning_rate, momentum=momentum)
                
                
def params_fct():
    #params = yaml.safe_load(open("./../../params.yaml"))["train"]
    params = yaml.safe_load(open("params.yaml"))["train"]
    include_top_p=params.get('include_top')
    input_shape_p_str=params.get('input_shape')
    loss_p=params.get('loss')
    metrics_p=params.get('metrics')
    learning_rate_p=params.get('learning_rate')
    momentum_p=params.get('momentum')
    nbr_epoch_p=params.get('nbr_epoch')
    rescale_p=params.get('rescale')
    image_size_p=params.get('image_size')
    batch_size_p=params.get('batch_size')
    print(params)# for test
      
    params=Params(                 rescale=rescale_p,
                                   image_size=image_size_p,
                                   batch_size=batch_size_p,
                   
                                   nbr_epoch=nbr_epoch_p,
                                   loss=loss_p,
                                   metrics = metrics_p,
                                   learning_rate= learning_rate_p,
                                   momentum=momentum_p,
                   
                                   include_top=include_top_p,
                                   input_shape_str=input_shape_p_str
                   )
    return params


class generator(object):
      def __init__(self,rescale=255,train_path="",validation_path="",image_size=(224,224),batch_size=10):
          train_datagen = ImageDataGenerator(1./rescale) 
          self.train_generator = train_datagen.flow_from_directory( train_path,
                                                              target_size=image_size,
                                                              batch_size=batch_size )
          validation_datagen = ImageDataGenerator(rescale) # attention !!!!!! 
          self.validation_generator = validation_datagen.flow_from_directory( validation_path,
                                                                         target_size=image_size,
                                                                         batch_size=batch_size,
                                                                         shuffle=False ) 
          self.class_number = self.train_generator.num_classes     
          

def modelLoad(target_dir):
     return load_model(target_dir)
def historyLoad(target_dir):
    df = pd.read_csv(target_dir)
    return df

def classification_report_fct(validation_generator,model,classification_report_path,conf_matrix_path):
    Classes_names =list(validation_generator.class_indices.keys())
    validation_generator.reset()
    results = model.predict(validation_generator)
    predicted_labels_idx = np.argmax(results,axis=1)
    predicted_labels_names = np.array([Classes_names[label] for label in predicted_labels_idx])
    real_labels_idx = validation_generator.labels
    real_labels_names = np.array([Classes_names[label] for label in real_labels_idx])
    clsf_report = pd.DataFrame(classification_report(real_labels_names,predicted_labels_names,labels=Classes_names,output_dict=True)).transpose()
    clsf_report.to_csv(classification_report_path, index= True)
    conf_matrix = confusion_matrix(real_labels_names,predicted_labels_names,labels=Classes_names,normalize='pred')
    conf_matrix_df = pd.DataFrame(conf_matrix, index = Classes_names,columns = Classes_names)
    conf_matrix_df.to_csv(conf_matrix_path)
    
    
def observing_accuracy(df,savepath):
    print(df['val_loss'])
    plt.plot(df['accuracy'])
    plt.plot(df['val_accuracy'])
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'validation'], loc='upper left')
    # fig = plt.get_figure()
    plt.savefig(savepath)