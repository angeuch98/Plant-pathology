# -*- coding: utf-8 -*-
"""## TRAINING
"""

import numpy as np # import numpy library
import tensorflow as tf # import Tensorflow library
from PIL import Image # Import Image from PIL - a convenient package for dealing with images!
import requests
import matplotlib.pyplot as plt # and this is for visualization
import seaborn as sns # visualization library, similar to matplotlib (based on matplotlib, actually)
from tensorflow.data import Dataset # an object to store our train, val, and test data
from tensorflow.keras import Sequential, Model # Keras API for adding layers to deep neural network
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, GlobalAveragePooling2D, Dropout # CNN layers
from mpl_toolkits.axes_grid1 import ImageGrid
import PIL
from keras.layers import BatchNormalization
from tensorflow.keras import backend as K
from tensorflow.keras import layers
from tensorflow.keras import regularizers
from tensorboard.plugins.hparams import api as hp
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.layers.experimental import preprocessing
import csv
import pandas as pd
from keras import layers
from keras import models
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import confusion_matrix
from keras import optimizers
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator
import os, shutil
from google.colab import drive
drive.mount('/content/drive')
tf.random.set_seed(123)

"""### **A function that draws plots containing the history of network training. The plots show loss and accuracy in each epoch**

"""

# History plotting function
def plot_history(history):
  # History for accuracy
  plt.figure(figsize=(14,5))
  plt.subplot(1,2,1)
  plt.plot(history.history['acc'])
  plt.plot(history.history['val_acc'])
  plt.title('model accuracy')
  plt.ylabel('accuracy')
  plt.xlabel('epoch')
  plt.legend(['train', 'valid'], loc='lower right')
  # History for loss
  plt.subplot(1,2,2)
  plt.plot(history.history['loss'])
  plt.plot(history.history['val_loss'])
  plt.title('model loss')
  plt.ylabel('loss')
  plt.xlabel('epoch')
  plt.legend(['train', 'valid'], loc='upper right')
  plt.show()

"""### **A function that draws a confusion matrix containing the number of test images classified into a given class**

"""

# draving confusion matrix
def confusion_map(matrix):
    plt.figure(figsize=(8,6))
    ax=plt.subplot()
    sns.heatmap(matrix,annot=True,fmt='g',ax=ax)
    ax.set_xlabel('Predicted labels'); ax.set_ylabel('True labels');
    ax.set_title('Confusion matrix');
    ax.xaxis.set_ticklabels(['healthy','multiple_diseases', 'rust', 'scab']);
    ax.yaxis.set_ticklabels(['healthy','multiple_diseases', 'rust', 'scab']);

"""# Network training


#### training and test images have been loaded into the colab memory in the form of a zip, the following functions are used to unpack folders with photos
"""

!mkdir train_classes
!unzip train_classes.zip -d train_classes

!mkdir test_classes
!unzip test_classes.zip -d test_classes

"""## Creating a training, test and validation set using ImageDataGenerator and augmentation:

Data augumentation consists in introducing  slightly modified copies of existing data into the trainig material, which usually translates positively into the results of machine learning, and prevents to overfiting.
"""

train_datagen = ImageDataGenerator(rescale=1./255,
                                    validation_split=0.2,
                                    horizontal_flip=True,
                                    vertical_flip=True,
                                    rotation_range=40,
                                    zoom_range=0.3)

test_datagen = ImageDataGenerator(rescale=1./255.)

train_generator = train_datagen.flow_from_directory(
  #"C:/Users/48507/Downloads/plant-pathology-2020-fgvc7/train_classes",
  '/content/train_classes/train_classes',
  #train_path,
  shuffle=True,
  target_size=(256, 256),
  batch_size=16,
  subset='training')

validation_generator = train_datagen.flow_from_directory(
 #"C:/Users/48507/Downloads/plant-pathology-2020-fgvc7/train_classes",
 '/content/train_classes/train_classes',
  #train_path,
  shuffle=True,
  target_size=(256, 256),
  batch_size=16,
  subset='validation')

# Generators for the test data
test_generator = test_datagen.flow_from_directory(
    #"C:/Users/48507/Downloads/plant-pathology-2020-fgvc7/test_classes",
    "/content/test_classes/test_classes",
    #test_path,
    shuffle=False,
    target_size=(256, 256),
    batch_size=1)

"""### Showing exemplary images belonging to train set"""

x,y = train_generator.next()
print("Shape of the batch: ",x.shape)
print("Number of classes: ",len(y[0]))

_,axs = plt.subplots(4,4, figsize=(10,10))
axs = axs.flatten()
for i,ax in zip(range(16), axs):
    ax.imshow(x[i])
plt.suptitle("Train set examples")
plt.show()

"""### Showing exemplary images belonging to test set"""

x1,y1 = validation_generator.next()
print("Shape of the batch: ",x1.shape)
print("Number of classes: ",len(y1[0]))

_,axs1 = plt.subplots(4,4, figsize=(10,10))
axs1 = axs1.flatten()
for j,ax1 in zip(range(16), axs1):
    ax1.imshow(x1[j])
plt.suptitle("Validation set examples")
plt.show()

x,y=train_generator.next()
x1,y1=validation_generator.next()
y_int = np.argmax(y,axis=-1)
y1_int = np.argmax(y1,axis=-1)
class_mapping = train_generator.class_indices
class_mapping1 = validation_generator.class_indices

def show_grid(image_list,nrows,ncols,class_map,label_list=None,show_labels=False,savename=None,figsize=(10,10),showaxis='off',myset=None):
    if type(image_list) is not list:
        if(image_list.shape[-1]==1):
            image_list = [image_list[i,:,:,0] for i in range(image_list.shape[0])]
        elif(image_list.shape[-1]==3):
            image_list = [image_list[i,:,:,:] for i in range(image_list.shape[0])]
    fig = plt.figure(None, figsize,frameon=False)
    plt.suptitle(myset)
    grid = ImageGrid(fig, 111,  # similar to subplot(111)
                     nrows_ncols=(nrows, ncols),  # creates 2x2 grid of axes
                     axes_pad=0.3,  # pad between axes in inch.
                     share_all=True,
                     )
    for i in range(nrows*ncols):
        ax = grid[i]
        ax.imshow(image_list[i],cmap='Greys_r')  # The AxesGrid object work as a list of axes.
        ax.axis('off')
        if show_labels:
            #ax.set_title(class_mapping[y_int[i]])
            ax.set_title([k for k, v in class_map.items() if v == label_list[i]])
    if savename != None:
        plt.savefig(savename,bbox_inches='tight')

typeset="Train data"
typeset1="Validation data"

"""### Showing exemplary images belonging to train set (with labels)"""

show_grid(x,4,4,class_map=class_mapping,label_list=y_int,show_labels=True,figsize=(20,15),savename='/content/Images/image_grid_train.png',myset=typeset)

"""### Showing exemplary images belonging to validation set (with labels)"""

show_grid(x1,4,4,class_map=class_mapping1,label_list=y1_int,show_labels=True,figsize=(20,15),savename='/content/Images/image_grid_valid.png',myset=typeset1)

"""*kursywa*#### Above have presented several sample images from the training and validation sets. It can be concluded that performed augumentation did not negatively affect the apperance of the most important elements i.e. the leaves, which means that the augumentation parameters were selected correctly."""

xt,yt = test_generator.next()
print("Shape of the batch: ",xt.shape)

# class indexes in test set
print("Indices of the class (test): ",test_generator.class_indices)
names=test_generator.class_indices.keys()
# class indexes in train set;
print("Indices of the class (train): ",train_generator.class_indices)

"""## Early stopping:

#### Early stopping is very usefull tool because it stops training when classification quality is not improving. This is a type of regularization used to avoid overfitting.
"""

early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor="val_loss",
    patience=10 # The number of epochs without improvement after which the training will be stopped.
    )

"""## Creating an instance of a simple convolutional network:"""

NO_CLASSES=4

simpleCNN = models.Sequential()
simpleCNN.add(layers.Conv2D(32, (3, 3), activation='relu',
 input_shape=(256,256, 3)))
#simpleCNN.add(BatchNormalization())
simpleCNN.add(layers.MaxPooling2D((2, 2)))
simpleCNN.add(layers.Conv2D(32, (3, 3), activation='relu')) #32
simpleCNN.add(layers.MaxPooling2D((2, 2)))
simpleCNN.add(layers.Conv2D(64, (3, 3), activation='relu'))
simpleCNN.add(layers.MaxPooling2D((2, 2)))
simpleCNN.add(layers.Conv2D(64, (3, 3), activation='relu'))
simpleCNN.add(layers.MaxPooling2D((2, 2)))
simpleCNN.add(layers.Flatten())
simpleCNN.add(layers.Dropout(0.5))
simpleCNN.add(layers.Dense(64, activation='relu')) #64
simpleCNN.add(layers.Dense(NO_CLASSES, activation='softmax'))

model_checkpoint_callback0 = tf.keras.callbacks.ModelCheckpoint(
    filepath='SimpleCNN.h5',
    save_weights_only=True,
    monitor='val_loss',
    save_best_only=True
    )

simpleCNN.summary()

simpleCNN.compile(loss='categorical_crossentropy',
 optimizer=Adam(0.001),
 metrics=['acc'])

historyCNN = simpleCNN.fit(train_generator,
                    validation_data=validation_generator,
                    epochs=100,
                    callbacks=[early_stopping, model_checkpoint_callback0])

plot_history(historyCNN)

simpleCNN.evaluate(test_generator, return_dict=True)

y_test=test_generator.labels

y_predict=simpleCNN.predict(test_generator)
y_predict=np.argmax(y_predict,axis=1)

conf_mat = confusion_matrix(y_test, y_predict)
confusion_map(conf_mat)

"""### It can be noticed that due to the fact that the 'multiple disease' class is the least numerous, only 3 images out of 14 were correctly classified. For this reason, we decided to weight the classes during network training. The least numerous class was given the greatest weight to make the model focus more on this class during training."""

class_weight = {0: 1.0,
                1: 1.5,
                2: 1.0,
                3: 1.0}
NO_CLASSES=4

"""#### Different weight values were tried, but for larger values (>1.5) the overall quality of the classification was worse because the model focused too much on training the smallest class, which affected the quality of the classification of the other classes. For values <1.5, however, there was no improvement in the quality of classification for the 'multiple disease' class. So a value of 1.5 was finally chosen because it did not negatively affect the overall classification quality and improved the classification quality for the multiple disease class.

## Simple CNN with class weight
"""

simpleCNN2 = models.Sequential()
simpleCNN2.add(layers.Conv2D(32, (3, 3), activation='relu',
 input_shape=(256,256, 3)))
simpleCNN2.add(layers.MaxPooling2D((2, 2)))
simpleCNN2.add(layers.Conv2D(32, (3, 3), activation='relu')) #32
simpleCNN2.add(layers.MaxPooling2D((2, 2)))
simpleCNN2.add(layers.Conv2D(64, (3, 3), activation='relu'))
simpleCNN2.add(layers.MaxPooling2D((2, 2)))
simpleCNN2.add(layers.Conv2D(64, (3, 3), activation='relu'))
simpleCNN2.add(layers.MaxPooling2D((2, 2)))
simpleCNN2.add(layers.Flatten())
simpleCNN2.add(layers.Dropout(0.5))
simpleCNN2.add(layers.Dense(64, activation='relu')) #64
simpleCNN2.add(layers.Dense(NO_CLASSES, activation='softmax'))

model_checkpoint_callback0 = tf.keras.callbacks.ModelCheckpoint(
    filepath='/content/drive/MyDrive/models/SimpleCNN2.h5',
    save_weights_only=True,
    monitor='val_loss',
    save_best_only=True
    )
simpleCNN2.compile(loss='categorical_crossentropy',
 optimizer=Adam(0.001),
 metrics=['acc'])

historyCNN_class = simpleCNN2.fit(train_generator,
                    validation_data=validation_generator,
                    epochs=100,
                    class_weight=class_weight,
                    callbacks=[early_stopping, model_checkpoint_callback0])

plot_history(historyCNN_class)

simpleCNN2.evaluate(test_generator, return_dict=True)

y_predict_class=simpleCNN2.predict(test_generator)
y_predict_class=np.argmax(y_predict_class,axis=1)
conf_mat_class = confusion_matrix(y_test, y_predict_class)
confusion_map(conf_mat_class)

"""### It can be seen that after assigning weights, the model coped better with the classification of objects belonging to the 'multiple disease' class.

## Transfer learning - ResNet50V2
"""

# learning rate sheduler
lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate = 0.001, # initial learning rate, 0.001 (default in most optimizers)
    decay_steps = 100, # number of steps after which learning rate will be decayed
    decay_rate = 0.99, # rate of the decay
    staircase=False # if True decay the learning rate at discrete intervals
)

# TRANSFERED RESNET
tf.keras.backend.clear_session()

base_model = tf.keras.applications.ResNet50V2(
    include_top=False, # wether to include last layers in the network
    weights='imagenet', # wether to use pre-trainned weights
    input_shape=(256,256, 3),
    classes=4
)
# fine tuning:
# Dissable training in the layers of the base_model
base_model.trainable = False

base_inputs = base_model.layers[0].input
base_outputs = base_model.layers[128].output

model_inputs = base_inputs
x = GlobalAveragePooling2D()(base_outputs)
x = Dropout(0.3)(x)
x = Dense(512, activation= 'selu')(x)
x = Dropout(0.2)(x)
x = Dense(256, activation= 'selu')(x)
x = Dropout(0.5)(x)
model_outputs = Dense(4, activation= 'softmax')(x) # Dense layer with filters equal to number of classes and 'softmax' activation function


model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath='modelResNet.h5',
    save_weights_only=True,
    monitor='val_loss',
    save_best_only=True
    )

ResNet = Model(inputs = model_inputs, outputs=model_outputs) # pass model_inputs to the inputs and model_outputs to the outputs
ResNet.summary()

ResNet.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
              loss='categorical_crossentropy',
              metrics=['acc'])


historyResNet = ResNet.fit(train_generator,
                     validation_data=validation_generator,
                     class_weight=class_weight,
                     epochs=100,
                     callbacks=[early_stopping, model_checkpoint_callback])

plot_history(historyResNet)

ResNet.evaluate(test_generator, return_dict=True)

y_predict2=ResNet.predict(test_generator)
y_predict2=np.argmax(y_predict2,axis=1)

conf_mat2 = confusion_matrix(y_test, y_predict2)
confusion_map(conf_mat2)

"""#### Based on the obtained confusion matrix, it can be seen that ResNET, despite the set weights for classes, did not manage to classify images belonging to the 'multiple disease' class. In the plots showing the history of trainig we can see that the graphs "jump". W e thought the likely cause was too small a batch size but after increasing the batch size similar results were obtained so we stayed with batch size 32. Another possible cause could be unrepresentative validation set because bigger 'jumps' are for this set, but we thing that 20% of train set is enough for validation. It is also possible that the learning rate should be reduced but the lower learning rate, the longer the network training , which in our case is a limitation due to the use of colab. However for a simple convolutional network we used the same learning rate and got much better results of the classification, and some literature sources give us the information that learning rate = 0.001 is optimal value.

## Transfer learning - Xception
"""

#transfered Xception
tf.keras.backend.clear_session()

base_model = tf.keras.applications.Xception(
    include_top=False, # wether to include last layers in the network
    weights='imagenet', # wether to use pre-trainned weights
    input_shape=(256,256, 3),
    classes=4
)
# fine tuning:
# Dissable training in the layers of the base_model
base_model.trainable = False

base_inputs = base_model.layers[0].input
base_outputs = base_model.layers[128].output

model_inputs = base_inputs
x = GlobalAveragePooling2D()(base_outputs)
x = Dropout(0.3)(x)
x = Dense(128, activation= 'selu')(x)
x = Dropout(0.2)(x)
x = Dense(64, activation= 'selu')(x)
x = Dropout(0.5)(x)
model_outputs = Dense(4, activation= 'softmax')(x) # Dense layer with filters equal to number of classes and 'softmax' activation function

model_checkpoint_callback1 = tf.keras.callbacks.ModelCheckpoint(
    filepath='modelXpection.h5',
    save_weights_only=True,
    monitor='val_loss',
    save_best_only=True
    )

Xception = Model(inputs = model_inputs, outputs=model_outputs) # pass model_inputs to the inputs and model_outputs to the outputs
Xception.summary()

Xception.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.00001),
              loss=tf.keras.losses.CategoricalCrossentropy(),
              metrics=['acc'])

historyXception = Xception.fit(train_generator,
                     validation_data=validation_generator,
                     class_weight = class_weight,
                     epochs=100,
                     callbacks=[early_stopping, model_checkpoint_callback1])

plot_history(historyXception)

Xception.evaluate(test_generator, return_dict=True)

y_predict3=Xception.predict(test_generator)
y_predict3=np.argmax(y_predict3,axis=1)

conf_mat3 = confusion_matrix(y_test, y_predict3)
confusion_map(conf_mat3)

"""## Transfer learning - VGG19"""

#transfered VGG19
tf.keras.backend.clear_session()

base_model = tf.keras.applications.VGG19(
    include_top=False, # wether to include last layers in the network
    weights='imagenet', # wether to use pre-trainned weights
    input_shape=(256,256, 3),
    classes=4
)
# fine tuning:
# Dissable training in the layers of the base_model
base_model.trainable = False

base_inputs = base_model.layers[0].input
base_outputs = base_model.layers[19].output

model_inputs = base_inputs
x = GlobalAveragePooling2D()(base_outputs)
x = Dropout(0.3)(x)
x = Dense(128, activation= 'selu')(x)
x = Dropout(0.2)(x)
x = Dense(64, activation= 'selu')(x)
x = Dropout(0.5)(x)
model_outputs = Dense(4, activation= 'softmax')(x) # Dense layer with filters equal to number of classes and 'softmax' activation function


model_checkpoint_callback2 = tf.keras.callbacks.ModelCheckpoint(
    filepath='modelVGG19.h5',
    save_weights_only=True,
    monitor='val_loss',
    save_best_only=True
    )

VGG19 = Model(inputs = model_inputs, outputs=model_outputs) # pass model_inputs to the inputs and model_outputs to the outputs
VGG19.summary()

VGG19.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
              loss=tf.keras.losses.CategoricalCrossentropy(),
              metrics=['acc'])

historyVGG19 = VGG19.fit(train_generator,
                     validation_data=validation_generator,
                     epochs=100,
                     class_weight = class_weight,
                     callbacks=[early_stopping, model_checkpoint_callback2])

plot_history(historyVGG19)

VGG19.evaluate(test_generator, return_dict=True)

y_predict4=VGG19.predict(test_generator)
y_predict4=np.argmax(y_predict4,axis=1)

conf_mat4 = confusion_matrix(y_test, y_predict4)
confusion_map(conf_mat4)

"""## Comparison of classification quality results for the created networks"""

res = np.array([[0.17,  0.94,  ],
                [0.38,  0.69,  ],
                [0.21,  0.82,  ],
                [0.41,  0.62]])

plt.figure(figsize=(8,8))
ax1=plt.subplot()
sns.heatmap(res,annot=True,fmt='g',ax=ax1)
ax1.set_xlabel('Results for the test set'); ax1.set_ylabel('Architectures');
ax1.set_title('Comparison');
ax1.xaxis.set_ticklabels(['Loss','Accuracy']);
ax1.yaxis.set_ticklabels(['Simple CNN','ResNET', 'Xception', 'VGG19']);

"""#### Based on the obtained results we can say that the best classification results are for the **small convolutional network**. Xception also did well with classification. The worst in this ranking were VGG19 and ResNET

## **Hyperparameter tuning**
#### Hyperparameter tuning was performed for the architecture that received the best results.
"""

# Commented out IPython magic to ensure Python compatibility.
#Hyperparameter tunning for best architecture - Tensorboard Grid Search
# First, we need to load the TensorBoard notebook extension
# %load_ext tensorboard
logdir = "logs/hparam_tuning/"
if os.path.isdir(logdir):
  shutil.rmtree(logdir)

HP_dropout = hp.HParam('dropout',hp.RealInterval(0.3,0.6))# interval for dropout
HP_num_units = hp.HParam('filters',hp.Discrete([64,256,512]))
HP_learning_rate = hp.HParam('learning_rate',hp.Discrete([1e-05,1e-03]))
HP_metric = hp.Metric('epoch_accuracy',
                      group='validation',
                      display_name='val_accurcy')

"""#### In hyperparameter tuning we diecided to test three parameters:
- dropout rate (0.3; 0.6)
- learning rate (1e-05; 1e-03)
- numer of filters (64, 256, 512)

#### Due to the long calculation time and limitations related to the use of collab, we set the number of epochs to only 12 and reduced the number of filters and dropout rate to be tested.
"""

with tf.summary.create_file_writer(logdir).as_default():
  hp.hparams_config(hparams=[HP_num_units,HP_dropout,HP_learning_rate],
                    metrics=[HP_metric],
                    )

def simple_CNN(filters, learning_rate, dropout_rate):
  model = tf.keras.Sequential()
  model.add(layers.Conv2D(filters, (3, 3), activation='relu',
   input_shape=(256,256, 3)))
  model.add(layers.MaxPooling2D((2, 2)))
  model.add(layers.Conv2D(filters, (3, 3), activation='relu')) #32
  model.add(layers.MaxPooling2D((2, 2)))
  model.add(layers.Conv2D(filters, (3, 3), activation='relu'))
  model.add(layers.MaxPooling2D((2, 2)))
  model.add(layers.Conv2D(filters, (3, 3), activation='relu'))
  model.add(layers.MaxPooling2D((2, 2)))
  model.add(layers.Flatten())
  model.add(layers.Dropout(dropout_rate))
  model.add(layers.Dense(filters, activation='relu')) #64
  model.add(layers.Dense(NO_CLASSES, activation='softmax'))


  model.compile(optimizer = tf.keras.optimizers.Adam(learning_rate),
                       loss='categorical_crossentropy', metrics=['acc'])
  return model

NO_CLASSES=4
count=0

for num_unit in HP_num_units.domain.values:
  for dropout_rate in tf.linspace(HP_dropout.domain.min_value,HP_dropout.domain.max_value,2):
    dropout_rate=dropout_rate.numpy()
    for learn_rate in HP_learning_rate.domain.values:
      hparams={
          HP_num_units:num_unit,
          HP_dropout:dropout_rate,
          HP_learning_rate:learn_rate
      }

      #define the model
      model=simple_CNN(filters=num_unit,learning_rate=learn_rate,dropout_rate=dropout_rate)
      logdir_model="logs/hparam_tuning/model_{}".format(count)
      print("Train model: Filters: {} Dropout: {} Learnin rate: {}".format(num_unit,dropout_rate,learn_rate))
      #callbacks
      callback=tf.keras.callbacks.TensorBoard(logdir_model)
      hp_callback=hp.KerasCallback(logdir_model,hparams)

      # fit the model
      model.fit(train_generator,
                validation_data=validation_generator,
                epochs=12,
                class_weight=class_weight,
                callbacks=[early_stopping,hp_callback,callback])

      count +=1

# Commented out IPython magic to ensure Python compatibility.
# %tensorboard --logdir logs/hparam_tuning/

"""#### Unfortunately, due to limitations related to the use of colab, we were able to perform oonly 8 different combinations of parameters:
- Filters: 64, Dropout: 0.30, Learning rate: 1e-05
- Filters: 64, Dropout: 0.30, Learning rate: 0.001
- Filters: 64, Dropout: 0.60, Learning rate: 1e-05
- Filters: 64, Dropout: 0.60, Learning rate: 0.001
- Filters: 256, Dropout: 0.30, Learning rate: 1e-05
- Filters: 256, Dropout: 0.30, Learning rate: 0.001
- Filters: 256, Dropout: 0.60, Learning rate: 1e-05
- Filters: 256, Dropout: 0.60, Learning rate: 0.001

Therefore, we had to choose the best parameters based on what we were able to do. It can be seen that after going through a given number of epochs, the best results were achieved by the model with parameters:
**Filters: 64, Dropout: 0.30, Learning rate: 0.001**
In fact these parameters are not very different from the original ones we defined at the beggining, because only dropout rate is different. I guess that means we were able to choose the right parameters at the beginning :)

We know that these results may not be fully reriable because we have set a small number of epochs and a relatively small number of parameters to check. However, due to the long computation time and the previously mentioned GPU constraints, we were unable to do more.

## Training simple CNN with best parameters:
"""

num_unit=64
learn_rate=0.001
dropout_rate=0.3

callback_best = tf.keras.callbacks.ModelCheckpoint(
    filepath='best_simpleCNN.h5',
    save_weights_only=True,
    monitor='val_loss',
    save_best_only=True
    )

model=simple_CNN(filters=num_unit,learning_rate=learn_rate,dropout_rate=dropout_rate)
print("Train model: Filters: {}, Dropout: {}, Learning rate: {}".format(num_unit,dropout_rate,learn_rate))
best_history = model.fit(train_generator,
               validation_data=validation_generator,
               epochs=100,
               class_weight=class_weight,
               callbacks=[early_stopping,callback_best])

plot_history(best_history)

model.evaluate(test_generator, return_dict=True)

y_predict5=model.predict(test_generator)
y_predict5=np.argmax(y_predict5,axis=1)

conf_mat5 = confusion_matrix(y_test, y_predict5)
confusion_map(conf_mat5)

"""### Discussion and conclusion

####As we can see, the final quality of the model after hyperparameter tuning is slightly worse than the original model, which may be due to the fact that the tuning was carried out on a small number of parameters and a small number of epochs. Generally, it can be said that the classification quality obtained by us is very good for a simple convolutional network. Surprisingly, more complex networks such as ResNET, Xception or VGG19 gave worse results than ordinary CNN. It is possible that for this classification problem a simple neural network is quite sufficient to obtain satisfactory results.

#### Based on the obtained results, it can be concluded that the operation of the network depends to a large extent on the set parameters. It also be seen that depending on the selected architecture, get different classification quality for the same training and test sets. In this case, the best network is a simple convolutional network with maxpooling and dropout. Moreover, it should be mentioned that creating the most optimal architecture for a given problem is a very time-consuming task and many aspects must be takien into account.
"""