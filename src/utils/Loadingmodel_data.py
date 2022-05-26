#Loading model & data
import pandas as pd
from tensorflow.keras.models import Model,load_model
def modelLoad(target_dir):
     return load_model(target_dir)
def historyLoad(target_dir):
    df = pd.read_csv(target_dir)
    return df
    