

# Building the CNN

# Importing the Keras libraries and packages
from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense

# Initialising the CNN
classifier = Sequential()

#Convolution
classifier.add(Convolution2D(64, 3, 3, input_shape = (128,128,3), activation = 'relu'))

# Pooling
classifier.add(MaxPooling2D(pool_size = (2, 2)))

# Adding a second convolutional layer
classifier.add(Convolution2D(64, 3, 3, activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (2, 2)))

#  Flattening
classifier.add(Flatten())

# Full connection
classifier.add(Dense(output_dim = 128, activation = 'relu'))
classifier.add(Dense(output_dim = 128, activation = 'relu'))
# for output with more than 2 classes use activation 'softmax' and output_dim=no of classes
classifier.add(Dense(output_dim = 5, activation = 'softmax'))

# Compiling the CNN
# for output with more than 2 classes use loss='categorical_crossentropy'
classifier.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])

#  Fitting the CNN to the images

from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)

test_datagen = ImageDataGenerator(rescale = 1./255)

training_set = train_datagen.flow_from_directory('dataset/training_set',
                                                 target_size = (128,128),
                                                 batch_size = 32,
                                                 class_mode = 'categorical')

test_set = test_datagen.flow_from_directory('dataset/test_set',
                                            target_size = (128,128),
                                            batch_size = 32,
                                            class_mode = 'categorical')

classifier.fit_generator(training_set,
                         samples_per_epoch = 723,
                         nb_epoch = 45,
                         validation_data = test_set,
                         nb_val_samples = 144)


#prediction on single input image


import numpy as np
from keras.preprocessing import image

test_image=image.load_img('dataset/singleimg/Cardiomegaly (6).jpg',target_size=(128,128))
test_image= image.img_to_array(test_image)
test_image=np.expand_dims(test_image,axis=0)
result=classifier.predict(test_image)


training_set.class_indices
if result[0][0] ==1 :
    
     prediction='Atelectasis'
if result[0][1] ==1 :
    prediction='Cardiomegaly'
if result[0][2] ==1 :
    prediction='Effusion'
if result[0][3] ==1 :
    prediction= 'Fibrosis'
if result[0][4] ==1 :
    prediction='Infiltration'

