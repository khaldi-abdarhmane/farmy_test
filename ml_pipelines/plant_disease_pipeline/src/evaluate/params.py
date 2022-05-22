from tensorflow.keras.optimizers import SGD
from tensorflow.keras import optimizers
class Params(object):
      def __init__(self,
                   
                   rescale=255,
                   image_size=(224,224),
                   batch_size=10,
                   
                   nbr_epoch=3,
                   loss='categorical_crossentropy',
                   metrics = ['accuracy'],
                   learning_rate= 1e-3,
                   momentum=0.9,
                   include_top=False,
                   input_shape_str=""
                                  ):
                self.rescale=rescale
                self.image_size=tuple(map(int, image_size[1:-1].split(',')))
                self.batch_size=batch_size
                
                self.nbr_epoch=nbr_epoch
                self.loss=loss
                self.metrics=metrics
                self.learning_rate=learning_rate
                self.momentum=momentum
                
                self.input_shape = tuple(map(int, input_shape_str[1:-1].split(',')))
                self.include_top=include_top
                self.optimizer=optimizers.SGD(learning_rate=learning_rate, momentum=momentum)
              