from distutils.command.upload import upload
import tensorflow as tf
from utils.params_fct import params_fct
print(tf.__version__)
from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import Model,load_model
from tensorflow.keras.layers import Input,GlobalAveragePooling2D,Dropout,Dense
import sys
import yaml
from utils.params import Params

from utils.train.arch_model import Arch_model

from utils.generator import generator
import os
import pandas as pd
import mlflow
import mlflow.keras




params =params_fct()
params_dict= yaml.safe_load(open("params.yaml"))["train"]
print("[params]:",params)
print("[params_dict]:",params_dict)
upload_artifact=True

if len(sys.argv) != 5:
    sys.stderr.write("Arguments error. Usage:\n")
    sys.stderr.write("\tpython train.py features model\n")
    sys.exit(1)
    
train_path = sys.argv[1]
validation_path = sys.argv[2]
output_model = sys.argv[3]
output_history= sys.argv[4]
artifact_path="./../../results" # stock temporary the artifact of the experiments


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

mlflow.set_tracking_uri("http://ec2-52-91-102-87.compute-1.amazonaws.com:8080/")
mlflow.set_experiment("24may")

print("[ output model ]\n",output_model)
with mlflow.start_run() as run:
    experiment_artifact_path=os.path.join(artifact_path ,"training") ####

    history=model_.fit(generatorobjet.train_generator,
                   epochs= params.nbr_epoch,
                   validation_data=generatorobjet.validation_generator)
    model_.save(os.path.join(output_model))

    mlflow.log_params(params_dict)
    mlflow.keras.save_model(model_,os.path.join(experiment_artifact_path,"model_artifacts"))
    
    # if upload_artifact:# upload artifact to mlflow
    

    print("----mlflow.get_artifact_uri() : ",mlflow.get_artifact_uri())
    history_df = pd.DataFrame(history.history)
    history_df.to_csv(os.path.join(experiment_artifact_path,"history.csv" ),index=False)
    
    mlflow.log_artifacts(artifact_path)
    import os
    print("-- current folder ---",os.getcwd())






