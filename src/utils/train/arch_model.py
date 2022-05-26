from tensorflow.keras.layers import Input,GlobalAveragePooling2D,Dropout,Dense
from tensorflow.keras.models import Model
class Arch_model(object):
      def __init__(self,base_model,class_number:int):  
          x = base_model.output
          x = GlobalAveragePooling2D()(x)
          x = Dropout(0.3)(x)
          x = Dense(256, activation='relu')(x)
          self.predictions = Dense(class_number, activation='softmax')(x)
          self.model = Model(base_model.input, self.predictions)



