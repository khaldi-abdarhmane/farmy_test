import tensorflow as tf
from params_fct import params_fct
print(tf.__version__)
from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import Model,load_model
from tensorflow.keras.layers import Input,GlobalAveragePooling2D,Dropout,Dense
import sys
import yaml
from params import Params
from arch_model import Arch_model
from generator import generator
import os


params=params_fct()

if len(sys.argv) != 5:
    sys.stderr.write("Arguments error. Usage:\n")
    sys.stderr.write("\tpython train.py features model\n")
    sys.exit(1)
    
train_path = sys.argv[1]
validation_path = sys.argv[2]
output_model = sys.argv[3]
output_history= sys.argv[4]


generatorobjet=generator(rescale=params.rescale,
                         image_size=params.image_size,
                         batch_size=params.batch_size,
                         train_path=train_path,
                         validation_path= validation_path
                        )



base_model_1 = VGG16(include_top =params.include_top,
                     input_shape =params.input_shape)

arch_model=Arch_model(base_model= base_model_1,class_number=generatorobjet.class_number)

model_=arch_model.model
model_.compile(optimizer=params.optimizer, loss=params.loss, metrics= params.metrics )


history=model_.fit(generatorobjet.train_generator,
                   epochs= params.nbr_epoch,
                   validation_data=generatorobjet.validation_generator)

# target_dir = '/home/khaldi-user/Desktop/piplean/data/model'
model_.save(os.path.join(output_model))

import pandas as pd

history_df = pd.DataFrame(history.history)
history_df.to_csv(os.path.join(output_history),index=False)

