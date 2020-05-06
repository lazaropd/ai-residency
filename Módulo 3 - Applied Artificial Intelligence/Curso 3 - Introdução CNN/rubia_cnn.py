
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # avoid all those experimental tf2 messages
#os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'


import gc
import cv2
import time
import joblib
import random
import imutils
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf

import tensorflow.keras as keras
import tensorflow.keras.models as models
import tensorflow.keras.layers as layers

from tensorflow.keras.utils import to_categorical

from tensorflow.keras.preprocessing.image import load_img, img_to_array, ImageDataGenerator

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

from joblib.externals.loky import get_reusable_executor





class cnn_keras():
    
    def __init__(self, X, y, gpu=False, workers=-3):        
        self.release()
        self.gpu = gpu
        self.workers = workers
        self.check()
        self.loadData(X, y)
        self.describe()
        
    def loadData(self, X, y):
        #self.X = tf.convert_to_tensor(X)
        #self.y = tf.convert_to_tensor(y)  
        self.X = X
        self.y = y
        
    def release(self):
        get_reusable_executor().shutdown(wait=True)
        tf.keras.backend.clear_session()
        gc.collect()

    def check(self):
        self.importers()  
            
    def describe(self):
        print('\nNeural network setup:')
        print('{:25s}'.format('X data shape:'), self.X.shape)
        print('{:25s}'.format('y data shape:'), self.y.shape)
        
    def importers(self):        
        print('\nVersions:')
        print("Keras :" , keras.__version__)
        print("Tensorflow :" , tf.__version__) 
        if self.workers < 0: 
            self.workers = joblib.cpu_count() + self.workers
            print('\nNegative workers means all available except N-1 (%d)' % self.workers)
        if not self.gpu: 
            tf.config.threading.set_inter_op_parallelism_threads(self.workers)
            tf.config.threading.set_intra_op_parallelism_threads(self.workers)
            print('\nGPU disabled!')
            os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
            if len(tf.config.list_physical_devices('GPU')) > 0:
                print('You have at least one available GPU device not in use!')
            print('Using %d CPU workers' % self.workers)
        else:
            if len(tf.config.list_physical_devices('GPU')) < 1:
                print('\nGPU was not found. Using CPU instead!')
            else:
                print('\nGPU enabled!')
                if len(tf.config.list_physical_devices('GPU')) < self.workers or self.workers < 0:
                    self.workers = len(tf.config.list_physical_devices('GPU'))
                    print('Number of workers adjusted to fit the GPUs available')
                print('Using %d GPU workers' % self.workers)
                self.testGPU()
        self.multicore = True if self.workers > 1 else False
        print('Multiprocessing status:', self.multicore)
        
    def testGPU(self):
        cpu_slot, gpu_slot = 0, 0
        with tf.device('/CPU:' + str(cpu_slot)):
            start = time.time()
            tf.eye(10000,10000)
            end = time.time() - start
            print('CPU test for EYE(10000): ', end)
        with tf.device('/GPU:' + str(cpu_slot)):
            start = time.time()
            tf.eye(10000,10000)
            end = time.time() - start
            print('GPU test for EYE(10000): ', end)         

    def printHistory(self, history, metric='accuracy'):
        fig, axs = plt.subplots(nrows=1, ncols=2, figsize = (14, 7))
        print('\nPerformance (train/test): %.2f / %.2f' % (history[metric][-1], history['val_%s'%metric][-1]))
        axs[0].plot(history[metric], label='Train')
        axs[0].plot(history['val_%s'%metric], label='Test')
        axs[0].set_xlabel('Epoch')
        axs[0].set_title('Model Performance')
        axs[0].legend()
        axs[1].plot(history['loss'], label='Train')
        axs[1].plot(history['val_loss'], label='Test')
        axs[1].set_xlabel('Epoch')
        axs[1].set_title('Model Loss')
        axs[1].legend()
        plt.show()   

    def printPerformanceC(self, X, y, model=None):
        if model:
            y_true = y if y.shape[1] == 1 else np.argmax(y, axis=1)
            y_pred = model.predict_classes(X)
            report = classification_report(y_true, y_pred, output_dict=True)
            fig, ax = plt.subplots(figsize=(12, 6))
            plt.title('Confusion Matrix')
            #print(y_true.shape, y_true)
            #print(y_pred.shape, y_pred)
            sns.heatmap(confusion_matrix(y_true, y_pred), annot=True, cmap='YlGn', fmt='d',)
            plt.xlabel('Predicted')
            plt.ylabel('True Class')
            plt.yticks(rotation=90)
            plt.show()
            fig, ax = plt.subplots(figsize=(12, 6))
            plt.title('Classification Report')
            sns.heatmap(pd.DataFrame(report).iloc[0:3].T, annot=True, vmin=0, vmax=1, cmap='BrBG', fmt='.2g')
            plt.xlabel('Score')
            plt.show()
            print('ACCURACY: ', round(accuracy_score(y_true, y_pred)*100, 1), '%')


run = False

if run:

    from tensorflow.keras.datasets import mnist

    (train_images, train_labels), (test_images, test_labels) = mnist.load_data()

    train_images = train_images.reshape((60000, 28, 28, 1))
    train_images = train_images.astype('float32') / 255

    test_images = test_images.reshape((10000, 28, 28, 1))
    test_images = test_images.astype('float32') / 255

    train_labels = to_categorical(train_labels)
    test_labels = to_categorical(test_labels)

    k = cnn_keras(X=train_images, y=train_labels, gpu=False, workers=1)

    model = models.Sequential()
    model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Flatten())
    model.add(layers.Dense(10, activation='softmax'))
    model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])

    print('\nTraining started...')
    model.fit(k.X, k.y, epochs=5, batch_size=64, verbose=2, validation_split=0.2, 
            shuffle=True, workers=k.workers, use_multiprocessing=k.multicore)

