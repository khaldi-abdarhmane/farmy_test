from Loadingmodel_data import modelLoad,historyLoad 
import sys
from observing_accuracy import observing_accuracy
from classification_report import classification_report_fct

from generator import generator

from params_fct import params_fct
from confusion_matrix_plt import confusion_matrix_plt
if len(sys.argv) != 9:
    sys.stderr.write("Arguments error. Usage:\n")
    sys.stderr.write("\tpython train.py features model\n")
    sys.exit(1)
    
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


