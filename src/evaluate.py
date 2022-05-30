from utils.utils import modelLoad,historyLoad 
import sys
from utils.utils import observing_accuracy
from utils.utils import classification_report_fct
from utils.utils import generator
from pathlib import Path
from utils.utils import params_fct
from utils.evaluate.confusion_matrix_plt import confusion_matrix_plt
import mlflow

if len(sys.argv) != 9:
    sys.stderr.write("Arguments error. Usage:\n")
    sys.stderr.write("\tpython train.py features model\n")
    sys.exit(1)

# create results/evaluate folder to stock evaluate artifact
Path("./../../results/evaluate/evaluate_plt/").mkdir(parents=True,exist_ok=True)
Path("./../../results/evaluate/evaluate_csv/").mkdir(parents=True,exist_ok=True)

    
model_path = sys.argv[1]
history_df_path = sys.argv[2]
validation_path = sys.argv[3]
output1 = sys.argv[4]

train_path = sys.argv[5]

output2 = sys.argv[6]
output3 = sys.argv[7]
observing_accuracy_path = sys.argv[8]


df=historyLoad(history_df_path)
model=modelLoad(model_path)

params=params_fct()

generatorobjet=generator(rescale=params.rescale,
                         image_size=params.image_size,
                         batch_size=params.batch_size,
                         train_path=train_path,
                         validation_path= validation_path
                        )

 
observing_accuracy(df=df,savepath= observing_accuracy_path)
classification_report_fct(validation_generator= generatorobjet.validation_generator,model=model,classification_report_path= output1,conf_matrix_path=output2)
confusion_matrix_plt(conf_matrix_path=output2,output=output3)


"""
evaluation_artifact_path= "./../../results/evaluate/"
load_last_run= "ca016b69505043218e0f0c498d65da6d" #mlflow.last_active_run()
print("----load_last_run_ id =",load_last_run)

with mlflow.start_run( run_id= load_last_run) as last_run: 
    mlflow.log_artifacts(evaluation_artifact_path)

"""



