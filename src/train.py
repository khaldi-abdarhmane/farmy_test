from distutils.command.upload import upload
import tensorflow as tf
from utils.utils import params_fct
print(tf.__version__)
from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import Model,load_model
from tensorflow.keras.layers import Input,GlobalAveragePooling2D,Dropout,Dense
import sys
import yaml
from utils.utils import Params
from utils.train.arch_model import Arch_model
from utils.utils import generator
import os
import pandas as pd
import mlflow
import mlflow.keras
import mlflow.tensorflow
import shutil
import json
from pathlib import Path

# create results/training folder to stock training artifact
Path("./../../results/training").mkdir(parents=True,exist_ok=True)
Path("./../../results/training/history").mkdir(parents=True,exist_ok=True)



params =params_fct()
params_dict= yaml.safe_load(open("params.yaml"))["train"]
print("[params]:",params)
print("[params_dict]:",params_dict)
upload_artifact=True

if len(sys.argv) != 6:
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
    model_artifact_path=os.path.join(experiment_artifact_path,"keras")

    print("----mlflow.get_artifact_uri() : ",mlflow.get_artifact_uri())
 

    #model_.save(os.path.join(output_model))# temporrary
    """
    ### temp init
    model_ =mlflow.keras.load_model(model_artifact_path)
    ####
    """

    history=model_.fit(generatorobjet.train_generator,
                   epochs= params.nbr_epoch,
                   validation_data=generatorobjet.validation_generator)
    history_df = pd.DataFrame(history.history)
    history_path= os.path.join(experiment_artifact_path,"history/history.csv" )
    history_df.to_csv(history_path ,index=False)# save history file
    
    #history_path= os.path.join(experiment_artifact_path,"history/history.csv" ) # remove

    if Path(model_artifact_path).exists():  # remove  model_artifact_path folder  ,else he give exception
        shutil.rmtree(model_artifact_path)
    mlflow.keras.save_model(model_, model_artifact_path)

    mlflow.log_params(params_dict)
    mlflow.log_artifact(history_path)
    mlflow.keras.log_model(model_,"keras")
 

    #mlflow.log_artifact("./../../results/training/keras")



run_id= run.info.run_id
experiment_id =run.info.experiment_id
experiment_name= mlflow.get_experiment( experiment_id).name  
run_info= {
    "experiment_id" : experiment_id,
    "experiment_name": experiment_name,
    "run_id": run_id,
    "mlflow_server_url": mlflow.get_tracking_uri(),
}

with open( os.path.join( artifact_path , "run_info.json"), "w" ) as run_info_file:
    run_info_file.write(json.dumps(run_info) )


"""
            # temporary
    from utils.Loadingmodel_data import modelLoad,historyLoad 
    model_=modelLoad("./../../results/training/model2.h5")
    #----
"""




