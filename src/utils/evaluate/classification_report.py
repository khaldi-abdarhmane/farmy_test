import string
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix,classification_report
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

    
