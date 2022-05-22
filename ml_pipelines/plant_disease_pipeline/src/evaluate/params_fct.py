import yaml
from params import Params
import subprocess
import os
def params_fct():
  #Another example.
    list_files = subprocess.run(["pwd"])
    #params = yaml.safe_load(open("./../../params.yaml"))["train"]
    params = yaml.safe_load(open("./params.yaml"))["train"]

    include_top_p=params.get('include_top')
    input_shape_p_str=params.get('input_shape')
    loss_p=params.get('loss')
    metrics_p=params.get('metrics')
    learning_rate_p=params.get('learning_rate')
    momentum_p=params.get('momentum')
    nbr_epoch_p=params.get('nbr_epoch')
    rescale_p=params.get('rescale')
    image_size_p=params.get('image_size')
    batch_size_p=params.get('batch_size')
      
    params=Params(                 rescale=rescale_p,
                                   image_size=image_size_p,
                                   batch_size=batch_size_p,
                   
                                   nbr_epoch=nbr_epoch_p,
                                   loss=loss_p,
                                   metrics = metrics_p,
                                   learning_rate= learning_rate_p,
                                   momentum=momentum_p,
                   
                                   include_top=include_top_p,
                                   input_shape_str=input_shape_p_str
                   )
    return params
    
    
