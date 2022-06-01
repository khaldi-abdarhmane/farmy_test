
import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.metrics import confusion_matrix,classification_report
import shutil

import random
import matplotlib.cm as cm
from pathlib import Path

train_path = sys.argv[1]
validation_path = sys.argv[2]
plot_path = sys.argv[3]



<<<<<<< HEAD
Path("./../../results//exploratory/exploratory_img/plots/").mkdir(parents=True,exist_ok=True)
print("------output ", output)
"""
result_path=  "./../../results/"

if Path(result_path).exists():
  shutil.rmtree(result_path)
"""   
=======
 
Path(plot_path).mkdir(parents=True,exist_ok=True)
>>>>>>> b26fe8b5bb509ec20bfcd0b6905583624904bf86


lists = os.listdir(train_path)
diseases = []
crops = []
file_lst = []
for folder in lists:
    files = os.listdir(os.path.join(train_path,folder))
    files = [folder+'/'+file  for file in files]
    file_lst.extend(files)
    if(folder != 'background'): 
      diseases.extend([folder for i in range(len(files))])
      crops.extend([folder.split(sep='___')[0] for i in range(len(files))])
train_df = pd.DataFrame(list(zip(file_lst,crops,diseases)),columns =["Paths","Crops","Diseases"])

fig, axes = plt.subplots(nrows=1,ncols=2,figsize=(20,8))
with plt.style.context('ggplot'):
  train_df['Crops'].value_counts().plot(kind='pie', title='Validation data',ax = axes[0],subplots=True)
  train_df['Diseases'].value_counts().plot(kind='bar', color='C1',title='Validation data',ax = axes[1],subplots=True)
 
#plt.savefig(plot_path.__str__()+"/train.png")  
plt.savefig(os.path.join(plot_path,"train.png")) 

lists = os.listdir(validation_path)
diseases = []
crops = []
file_lst = []
for folder in lists:
    files = os.listdir(os.path.join(validation_path,folder))
    files = [folder+'/'+file  for file in files]
    file_lst.extend(files)
    if(folder != 'background'): 
      diseases.extend([folder for i in range(len(files))])
      crops.extend([folder.split(sep='___')[0] for i in range(len(files))])
validation_df = pd.DataFrame(list(zip(file_lst,crops,diseases)),columns =["Paths","Crops","Diseases"])

fig, axes = plt.subplots(nrows=1,ncols=2,figsize=(20,8))
with plt.style.context('ggplot'):
  validation_df['Crops'].value_counts().plot(kind='pie', title='Validation data',ax = axes[0],subplots=True)
  validation_df['Diseases'].value_counts().plot(kind='bar', color='C1',title='Validation data',ax = axes[1],subplots=True)
   
plt.savefig(os.path.join(plot_path,"valid.png"))  
#plt.savefig(plot_path.__str__()+"/valid.png")  