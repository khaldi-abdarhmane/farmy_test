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
import mlflow.tensorflow
import shutil

from pathlib import Path
# create results/training folder to stock training artifact
Path("./../../results/training").mkdir(parents=True,exist_ok=True)


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
mlflow_server_url= params_dict["mlflow_server_url"]# "http://ec2-54-163-48-30.compute-1.amazonaws.com:8080/" # update this ip with the mlflow ; ec2 public address is dynamics
experiment_name=    params_dict["experiment_name"] #"mlflow_dvc_pipeline"

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

mlflow.set_tracking_uri( mlflow_server_url)
mlflow.set_experiment(experiment_name)

print("[ output model ]\n",output_model)
with mlflow.start_run() as run:
    experiment_artifact_path=os.path.join(artifact_path ,"training") ####
    model_artifact_path=os.path.join(experiment_artifact_path,"model_artifacts")
    print("----mlflow.get_artifact_uri() : ",mlflow.get_artifact_uri())

    
    history=model_.fit(generatorobjet.train_generator,
                   epochs= params.nbr_epoch,
                   validation_data=generatorobjet.validation_generator)
    history_df = pd.DataFrame(history.history)
    history_path= os.path.join(experiment_artifact_path,"history.csv" )
    history_df.to_csv(history_path ,index=False)
                 
    model_.save(os.path.join(output_model))
    """
            # temporary
    from utils.Loadingmodel_data import modelLoad,historyLoad 
    model_=modelLoad("./../../results/training/model2.h5")
    #----
    """

    if Path(model_artifact_path).exists():
        shutil.rmtree(model_artifact_path) # remove  model_artifact_path folder 

    mlflow.log_params(params_dict)
    mlflow.log_artifact(history_path)
    mlflow.keras.log_model(model_,"keras")





