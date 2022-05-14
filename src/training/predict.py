from tensorflow.keras.models import Model,load_model
from tensorflow.keras.preprocessing.image import load_img,img_to_array,array_to_img
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import json

def prepare_image_for_prediction(img_path, size = (224,224)):
    # `img` is a PIL image of size 
    img = load_img(img_path, target_size=size)
    # `array` is a float32 Numpy array of shape (224, 224, 3)
    array = img_to_array(img)
    # We add a dimension to transform our array into a "batch"
    # of size (1, 224, 224, 3)
    array = np.expand_dims(array, axis=0)
    array = array/255
    return array

def predict_class(image_path, model, Classes_names):
  image_array = prepare_image_for_prediction(image_path)
  result = model.predict(image_array)
  index_max = result.argmax(axis=1)[0]
  return Classes_names[index_max]

def get_list_class(batch_size= 10, image_size= (224,224) ):
    train_path="small_dataset/train"
    train_datagen = ImageDataGenerator(rescale=1./255)
    train_generator = train_datagen.flow_from_directory(
                    train_path,
                    target_size=image_size,
                    batch_size=batch_size
                    )
    Classes_names =list(train_generator.class_indices.keys())
    return Classes_names

"""
def load_model(target_dir="models/my_model.h5"):
    model = load_model(target_dir)
    return model
"""
if __name__ == "__main__":
    
    with open("parameters.json") as f:
      pars = json.load(f)
      pr_args = pars["predict"]
    print("your paramater loaded form paramaters.json: ")
    print(pr_args)

    target_dir= pr_args["target_dir"]  
    image_path= pr_args ["image_path"] 

    Classes_names_list=get_list_class()# predict class give index and we use the list name classe to get the label    
    model= load_model(target_dir)
    predicted_class=predict_class(image_path, model,Classes_names_list)
    print("image path ={} \n predicted class of \n {}  ".format( image_path, predicted_class ) )

