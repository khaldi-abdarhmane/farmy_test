from tensorflow.keras.preprocessing.image import ImageDataGenerator
class generator(object):
      def __init__(self,rescale=255,train_path="",validation_path="",image_size=(224,224),batch_size=10):
          train_datagen = ImageDataGenerator(1./rescale) 
          self.train_generator = train_datagen.flow_from_directory( train_path,
                                                              target_size=image_size,
                                                              batch_size=batch_size )
          validation_datagen = ImageDataGenerator(rescale) # attention !!!!!! 
          self.validation_generator = validation_datagen.flow_from_directory( validation_path,
                                                                         target_size=image_size,
                                                                         batch_size=batch_size,
                                                                         shuffle=False ) 
          self.class_number = self.train_generator.num_classes 