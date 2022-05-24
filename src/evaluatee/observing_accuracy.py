#Observing Accuracy
import pandas as pd
import matplotlib.pyplot as plt
def observing_accuracy(df,savepath):
    print(df['val_loss'])
    plt.plot(df['accuracy'])
    plt.plot(df['val_accuracy'])
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'validation'], loc='upper left')
    # fig = plt.get_figure()
    plt.savefig(savepath)
   