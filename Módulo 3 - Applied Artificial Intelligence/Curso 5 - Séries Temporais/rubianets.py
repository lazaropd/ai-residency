# import the necessary packages

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # avoid all those experimental tf2 messages
#os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

import gc
import joblib
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from IPython.display import display, HTML, clear_output

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Conv1D, Conv2D
from tensorflow.keras.layers import MaxPooling1D, MaxPooling2D
from tensorflow.keras.layers import GlobalAveragePooling1D
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Embedding
from tensorflow.keras.layers import SimpleRNN
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Bidirectional

from tensorflow.keras import backend as K

from tensorflow.keras.optimizers import Adam, RMSprop, SGD
from tensorflow.keras.regularizers import l1, l2

from tensorflow.keras.callbacks import Callback

from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

from joblib.externals.loky import get_reusable_executor



class simplerKeras():
    
    def __init__(self, X, y, gpu=False, workers=-3, gpu_test=True):        
        self.release()
        self.gpu = gpu
        self.workers = workers
        self.gpu_test = gpu_test
        self.check()
        self.loadData(X, y)
        if isinstance(X, pd.DataFrame) and isinstance(y, pd.DataFrame):
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
                if self.gpu_test:
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
            if y.ndim == 1:
                y_true = y
            else:
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



class formatCallbackNotebook(Callback):
    
    def __init__(self, epochs, start, report='', timeout=0):
        self.epochs = epochs
        self.start = start
        self.report = report
        self.timeout = timeout
        
    def on_epoch_end(self, epoch, logs):
        clear_output(wait=True)
        elapsed = time.time() - self.start
        eta = (self.epochs - epoch - 1) * (elapsed) / (epoch + 1)
        results = {}
        for k, v in logs.items():
            results.update({k: round(v, 3)})
        message = self.report
        message += '\nCurrent epoch: %d/%d (ETA: %d seg)' % (epoch+1, self.epochs, eta) + "\nCurrent results: " + str(results)
        if elapsed/60 > self.timeout and self.timeout > 0:
            message += '\nTimed out after %d epochs' % epoch
            self.model.stop_training = True
        display(HTML(message.replace('\n','<br>')))





class StridedNetOne:
	
    @staticmethod
    def build(width, height, depth, classes, reg, init="he_normal", last_act='softmax'):
        model = Sequential()
        inputShape = (height, width, depth)
        chanDim = -1
        if K.image_data_format() == "channels_first":
            inputShape = (depth, height, width)
            chanDim = 1

        model.add(Conv2D(16, (3, 3), strides=(2, 2), padding="valid",
            kernel_initializer=init, kernel_regularizer=reg,
            input_shape=inputShape))
        model.add(Activation("relu"))
        model.add(BatchNormalization(axis=chanDim))
        model.add(Dropout(0.25))

        model.add(Conv2D(32, (3, 3), strides=(2, 2), padding="same",
            kernel_initializer=init, kernel_regularizer=reg))
        model.add(Activation("relu"))
        model.add(BatchNormalization(axis=chanDim))
        model.add(Dropout(0.25))

        model.add(Flatten())
        model.add(Dense(classes))
        model.add(Activation(last_act))

        return model


class StridedNetTwo:
	
    @staticmethod
    def build(width, height, depth, classes, reg, init="he_normal", last_act='softmax'):
        model = Sequential()
        inputShape = (height, width, depth)
        chanDim = -1
        if K.image_data_format() == "channels_first":
            inputShape = (depth, height, width)
            chanDim = 1

        model.add(Conv2D(16, (3, 3), padding="valid",
            kernel_initializer=init, kernel_regularizer=reg,
            input_shape=inputShape))
        model.add(Activation("relu"))
        model.add(BatchNormalization(axis=chanDim))
        model.add(Conv2D(16, (1, 1), padding="same",
            kernel_initializer=init, kernel_regularizer=reg))
        model.add(Activation("relu"))
        model.add(BatchNormalization(axis=chanDim))
        model.add(Conv2D(16, (3, 3), strides=(2, 2), padding="same",
            kernel_initializer=init, kernel_regularizer=reg))
        model.add(Activation("relu"))
        model.add(BatchNormalization(axis=chanDim))
        model.add(Dropout(0.25))

        model.add(Conv2D(32, (3, 3), strides=(2, 2), padding="same",
            kernel_initializer=init, kernel_regularizer=reg))
        model.add(Activation("relu"))
        model.add(BatchNormalization(axis=chanDim))
        model.add(Dropout(0.25))

        model.add(Flatten())
        model.add(Dense(classes))
        model.add(Activation(last_act))

        return model


class StridedNetThree:
	
    @staticmethod
    def build(width, height, depth, classes, reg, init="he_normal", last_act='softmax'):
        model = Sequential()
        inputShape = (height, width, depth)
        chanDim = -1
        if K.image_data_format() == "channels_first":
            inputShape = (depth, height, width)
            chanDim = 1

        model.add(Conv2D(16, (5, 5), padding="valid",
            kernel_initializer=init, kernel_regularizer=reg,
            input_shape=inputShape))
        model.add(Activation("relu"))
        model.add(BatchNormalization(axis=chanDim))
        model.add(Conv2D(16, (1, 1), padding="same",
            kernel_initializer=init, kernel_regularizer=reg))
        model.add(Activation("relu"))
        model.add(BatchNormalization(axis=chanDim))
        model.add(Conv2D(16, (3, 3), strides=(2, 2), padding="same",
            kernel_initializer=init, kernel_regularizer=reg))
        model.add(Activation("relu"))
        model.add(BatchNormalization(axis=chanDim))
        model.add(Dropout(0.25))

        model.add(Conv2D(32, (3, 3), padding="same",
            kernel_initializer=init, kernel_regularizer=reg))
        model.add(Activation("relu"))
        model.add(BatchNormalization(axis=chanDim))
        model.add(Conv2D(32, (3, 3), strides=(2, 2), padding="same",
            kernel_initializer=init, kernel_regularizer=reg))
        model.add(Activation("relu"))
        model.add(BatchNormalization(axis=chanDim))
        model.add(Dropout(0.25))

        model.add(Conv2D(64, (3, 3), strides=(2, 2), padding="same",
            kernel_initializer=init, kernel_regularizer=reg))
        model.add(Activation("relu"))
        model.add(BatchNormalization(axis=chanDim))
        model.add(Dropout(0.25))

        model.add(Flatten())
        model.add(Dense(4*classes, kernel_initializer=init))
        model.add(Activation("relu"))
        model.add(BatchNormalization())
        model.add(Dropout(0.5))
        model.add(Dense(classes))
        model.add(Activation(last_act))

        return model


class StridedNetFour:
	
    @staticmethod
    def build(width, height, depth, classes, reg, init="he_normal", last_act='softmax'):
        model = Sequential()
        inputShape = (height, width, depth)
        chanDim = -1
        if K.image_data_format() == "channels_first":
            inputShape = (depth, height, width)
            chanDim = 1

        model.add(Conv2D(16, (5, 5), padding="valid",
            kernel_initializer=init, kernel_regularizer=reg,
            input_shape=inputShape))
        model.add(Activation("relu"))
        model.add(BatchNormalization(axis=chanDim))
        model.add(Conv2D(16, (1, 1), padding="same",
            kernel_initializer=init, kernel_regularizer=reg))
        model.add(Activation("relu"))
        model.add(BatchNormalization(axis=chanDim))
        model.add(Conv2D(16, (3, 3), strides=(2, 2), padding="same",
            kernel_initializer=init, kernel_regularizer=reg))
        model.add(Activation("relu"))
        model.add(BatchNormalization(axis=chanDim))
        model.add(Dropout(0.25))

        model.add(Conv2D(32, (3, 3), padding="same",
            kernel_initializer=init, kernel_regularizer=reg))
        model.add(Activation("relu"))
        model.add(BatchNormalization(axis=chanDim))
        model.add(Conv2D(32, (1, 1), padding="same",
            kernel_initializer=init, kernel_regularizer=reg))
        model.add(Activation("relu"))
        model.add(BatchNormalization(axis=chanDim))
        model.add(Conv2D(32, (3, 3), strides=(2, 2), padding="same",
            kernel_initializer=init, kernel_regularizer=reg))
        model.add(Activation("relu"))
        model.add(BatchNormalization(axis=chanDim))
        model.add(Dropout(0.25))

        model.add(Conv2D(64, (3, 3), padding="same",
            kernel_initializer=init, kernel_regularizer=reg))
        model.add(Activation("relu"))
        model.add(BatchNormalization(axis=chanDim))
        model.add(Conv2D(64, (3, 3), strides=(2, 2), padding="same",
            kernel_initializer=init, kernel_regularizer=reg))
        model.add(Activation("relu"))
        model.add(BatchNormalization(axis=chanDim))
        model.add(Dropout(0.25))

        model.add(Flatten())
        model.add(Dense(8*classes, kernel_initializer=init))
        model.add(Activation("relu"))
        model.add(BatchNormalization())
        model.add(Dropout(0.5))
        model.add(Dense(4*classes, kernel_initializer=init))
        model.add(Activation("relu"))
        model.add(BatchNormalization())
        model.add(Dropout(0.5))
        model.add(Dense(classes))
        model.add(Activation(last_act))

        return model


class StridedNetFive:
	
    @staticmethod
    def build(width, height, depth, classes, reg, init="he_normal", last_act='softmax'):
        model = Sequential()
        inputShape = (height, width, depth)
        chanDim = -1
        if K.image_data_format() == "channels_first":
            inputShape = (depth, height, width)
            chanDim = 1

        model.add(Conv2D(16, (7, 7), padding="valid",
            kernel_initializer=init, kernel_regularizer=reg,
            input_shape=inputShape))
        model.add(Activation("relu"))
        model.add(BatchNormalization(axis=chanDim))
        model.add(Conv2D(16, (1, 1), padding="same",
            kernel_initializer=init, kernel_regularizer=reg))
        model.add(Activation("relu"))
        model.add(BatchNormalization(axis=chanDim))
        model.add(Conv2D(16, (3, 3), strides=(2, 2), padding="same",
            kernel_initializer=init, kernel_regularizer=reg))
        model.add(Activation("relu"))
        model.add(BatchNormalization(axis=chanDim))
        model.add(Dropout(0.25))

        model.add(Conv2D(32, (3, 3), padding="same",
            kernel_initializer=init, kernel_regularizer=reg))
        model.add(Activation("relu"))
        model.add(BatchNormalization(axis=chanDim))
        model.add(Conv2D(32, (1, 1), padding="same",
            kernel_initializer=init, kernel_regularizer=reg))
        model.add(Activation("relu"))
        model.add(BatchNormalization(axis=chanDim))
        model.add(Conv2D(32, (3, 3), strides=(2, 2), padding="same",
            kernel_initializer=init, kernel_regularizer=reg))
        model.add(Activation("relu"))
        model.add(BatchNormalization(axis=chanDim))
        model.add(Dropout(0.25))

        model.add(Conv2D(64, (3, 3), padding="same",
            kernel_initializer=init, kernel_regularizer=reg))
        model.add(Activation("relu"))
        model.add(BatchNormalization(axis=chanDim))
        model.add(Conv2D(64, (3, 3), strides=(2, 2), padding="same",
            kernel_initializer=init, kernel_regularizer=reg))
        model.add(Activation("relu"))
        model.add(BatchNormalization(axis=chanDim))
        model.add(Dropout(0.25))

        model.add(Conv2D(128, (3, 3), padding="same",
            kernel_initializer=init, kernel_regularizer=reg))
        model.add(Activation("relu"))
        model.add(BatchNormalization(axis=chanDim))
        model.add(Conv2D(128, (3, 3), strides=(2, 2), padding="same",
            kernel_initializer=init, kernel_regularizer=reg))
        model.add(Activation("relu"))
        model.add(BatchNormalization(axis=chanDim))
        model.add(Dropout(0.25))

        model.add(Conv2D(256, (3, 3), padding="same",
            kernel_initializer=init, kernel_regularizer=reg))
        model.add(Activation("relu"))
        model.add(BatchNormalization(axis=chanDim))
        model.add(Conv2D(256, (3, 3), strides=(2, 2), padding="same",
            kernel_initializer=init, kernel_regularizer=reg))
        model.add(Activation("relu"))
        model.add(BatchNormalization(axis=chanDim))
        model.add(Dropout(0.25))

        model.add(Flatten())
        model.add(Dense(16*classes, kernel_initializer=init))
        model.add(Activation("relu"))
        model.add(BatchNormalization())
        model.add(Dropout(0.5))
        model.add(Dense(4*classes, kernel_initializer=init))
        model.add(Activation("relu"))
        model.add(BatchNormalization())
        model.add(Dropout(0.5))
        model.add(Dense(classes))
        model.add(Activation(last_act))

        return model


class StridedNetSix:
	
    @staticmethod
    def build(width, height, depth, classes, reg, init="he_normal", last_act='softmax'):
        model = Sequential()
        inputShape = (height, width, depth)
        chanDim = -1
        if K.image_data_format() == "channels_first":
            inputShape = (depth, height, width)
            chanDim = 1

        model.add(Conv2D(16, (7, 7), padding="valid",
            kernel_initializer=init, kernel_regularizer=reg,
            input_shape=inputShape))
        model.add(Activation("relu"))
        model.add(BatchNormalization(axis=chanDim))
        model.add(Conv2D(16, (1, 1), padding="same",
            kernel_initializer=init, kernel_regularizer=reg))
        model.add(Activation("relu"))
        model.add(BatchNormalization(axis=chanDim))
        model.add(Conv2D(16, (3, 3), strides=(2, 2), padding="same",
            kernel_initializer=init, kernel_regularizer=reg))
        model.add(Activation("relu"))
        model.add(BatchNormalization(axis=chanDim))
        model.add(Dropout(0.25))

        model.add(Conv2D(32, (3, 3), padding="same",
            kernel_initializer=init, kernel_regularizer=reg))
        model.add(Activation("relu"))
        model.add(BatchNormalization(axis=chanDim))
        model.add(Conv2D(32, (1, 1), padding="same",
            kernel_initializer=init, kernel_regularizer=reg))
        model.add(Activation("relu"))
        model.add(BatchNormalization(axis=chanDim))
        model.add(Conv2D(32, (3, 3), strides=(2, 2), padding="same",
            kernel_initializer=init, kernel_regularizer=reg))
        model.add(Activation("relu"))
        model.add(BatchNormalization(axis=chanDim))
        model.add(Dropout(0.25))

        model.add(Conv2D(64, (3, 3), padding="same",
            kernel_initializer=init, kernel_regularizer=reg))
        model.add(Activation("relu"))
        model.add(BatchNormalization(axis=chanDim))
        model.add(Conv2D(64, (1, 1), padding="same",
            kernel_initializer=init, kernel_regularizer=reg))
        model.add(Activation("relu"))
        model.add(BatchNormalization(axis=chanDim))
        model.add(Conv2D(64, (3, 3), strides=(2, 2), padding="same",
            kernel_initializer=init, kernel_regularizer=reg))
        model.add(Activation("relu"))
        model.add(BatchNormalization(axis=chanDim))
        model.add(Dropout(0.25))

        model.add(Conv2D(128, (3, 3), padding="same",
            kernel_initializer=init, kernel_regularizer=reg))
        model.add(Activation("relu"))
        model.add(BatchNormalization(axis=chanDim))
        model.add(Conv2D(128, (1, 1), padding="same",
            kernel_initializer=init, kernel_regularizer=reg))
        model.add(Activation("relu"))
        model.add(BatchNormalization(axis=chanDim))
        model.add(Conv2D(128, (3, 3), strides=(2, 2), padding="same",
            kernel_initializer=init, kernel_regularizer=reg))
        model.add(Activation("relu"))
        model.add(BatchNormalization(axis=chanDim))
        model.add(Dropout(0.25))

        model.add(Conv2D(256, (3, 3), padding="same",
            kernel_initializer=init, kernel_regularizer=reg))
        model.add(Activation("relu"))
        model.add(BatchNormalization(axis=chanDim))
        model.add(Conv2D(256, (3, 3), padding="same",
            kernel_initializer=init, kernel_regularizer=reg))
        model.add(Activation("relu"))
        model.add(BatchNormalization(axis=chanDim))
        model.add(Dropout(0.25))

        model.add(Conv2D(512, (3, 3), padding="same",
            kernel_initializer=init, kernel_regularizer=reg))
        model.add(Activation("relu"))
        model.add(BatchNormalization(axis=chanDim))
        model.add(Conv2D(512, (3, 3), strides=(2, 2), padding="same",
            kernel_initializer=init, kernel_regularizer=reg))
        model.add(Activation("relu"))
        model.add(BatchNormalization(axis=chanDim))
        model.add(Dropout(0.25))

        model.add(Flatten())
        model.add(Dense(16*classes, kernel_initializer=init))
        model.add(Activation("relu"))
        model.add(BatchNormalization())
        model.add(Dropout(0.5))
        model.add(Dense(4*classes, kernel_initializer=init))
        model.add(Activation("relu"))
        model.add(BatchNormalization())
        model.add(Dropout(0.5))
        model.add(Dense(classes))
        model.add(Activation(last_act))

        return model


class PooledThree:
	
    @staticmethod
    def build(width, height, depth, classes, reg, init="he_normal", last_act='softmax'):
        model = Sequential()
        inputShape = (height, width, depth)
        chanDim = -1
        if K.image_data_format() == "channels_first":
            inputShape = (depth, height, width)
            chanDim = 1

        model.add(Conv2D(32, (3, 3), 
            kernel_initializer=init, kernel_regularizer=reg,
            input_shape=inputShape))
        model.add(Activation("relu"))
        model.add(MaxPooling2D((2, 2)))

        model.add(Conv2D(64, (3, 3), 
            kernel_initializer=init, kernel_regularizer=reg))
        model.add(Activation("relu"))
        model.add(MaxPooling2D((2, 2)))

        model.add(Conv2D(128, (3, 3), 
            kernel_initializer=init, kernel_regularizer=reg))
        model.add(Activation("relu"))
        model.add(MaxPooling2D((2, 2)))

        model.add(Conv2D(128, (3, 3), 
            kernel_initializer=init, kernel_regularizer=reg))
        model.add(Activation("relu"))
        model.add(MaxPooling2D((2, 2)))

        model.add(Flatten())
        model.add(Dense(512, activation='relu'))
        model.add(Dense(classes))
        model.add(Activation(last_act))

        return model



class EmbeddedOne:
	
    @staticmethod
    def build(input_dim, output_dim, input_len, classes, reg, init="he_normal", last_act='softmax'):
        model = Sequential()

        model.add(Embedding(input_dim=input_dim, output_dim=output_dim, input_length=input_len,
                            embeddings_initializer=init, activity_regularizer=reg))

        model.add(GlobalAveragePooling1D()) 

        model.add(Dense(16, activation='relu'))

        model.add(Dense(1, activation=last_act))

        return model


class EmbeddedTwo:
	
    @staticmethod
    def build(input_dim, output_dim, input_len, classes, reg, init="he_normal", last_act='softmax'):
        model = Sequential()

        model.add(Embedding(input_dim=input_dim, output_dim=output_dim, input_length=input_len,
                            embeddings_initializer=init, activity_regularizer=reg))

        model.add(Flatten())
        model.add(Dense(256, activation='relu'))
        model.add(Dense(16, activation='relu'))

        model.add(Dense(1, activation=last_act))

        return model


class EmbeddedThree:
	
    @staticmethod
    def build(input_dim, output_dim, input_len, classes, reg, init="he_normal", last_act='softmax'):
        model = Sequential()

        model.add(Embedding(input_dim=input_dim, output_dim=output_dim, input_length=input_len,
                            embeddings_initializer=init, activity_regularizer=reg))

        model.add(Conv1D(output_dim, 3, padding='same', activation='relu'))
        model.add(MaxPooling1D())

        model.add(Flatten())
        model.add(Dense(256, activation='relu'))
        model.add(Dropout(0.5))

        model.add(Dense(128, activation='relu'))
        model.add(Dropout(0.5))

        model.add(Dense(16, activation='relu'))
        model.add(Dropout(0.5))

        model.add(Dense(1, activation=last_act))

        return model


class EmbeddedFour:
	
    @staticmethod
    def build(input_dim, output_dim, input_len, classes, reg, init="he_normal", last_act='softmax'):
        model = Sequential()

        model.add(Embedding(input_dim=input_dim, output_dim=output_dim, input_length=input_len,
                            embeddings_initializer=init, activity_regularizer=reg))

        model.add(Conv1D(output_dim, 3, padding='valid', activation='relu'))
        model.add(MaxPooling1D())

        model.add(Conv1D(2*output_dim, 3, padding='valid', activation='relu'))
        model.add(MaxPooling1D())

        model.add(Flatten())
        
        model.add(Dense(256, activation='relu'))
        model.add(Dropout(0.5))

        model.add(Dense(128, activation='relu'))
        model.add(Dropout(0.5))

        model.add(Dense(64, activation='relu'))
        model.add(Dropout(0.5))

        model.add(Dense(32, activation='relu'))
        model.add(Dropout(0.5))

        model.add(Dense(16, activation='relu'))
        model.add(Dropout(0.5))

        model.add(Dense(1, activation=last_act))

        return model


class EmbeddedFive:
	
    @staticmethod
    def build(input_dim, output_dim, input_len, classes, reg, init="he_normal", last_act='softmax'):
        model = Sequential()

        model.add(Embedding(input_dim=input_dim, output_dim=output_dim, input_length=input_len,
                            embeddings_initializer=init, activity_regularizer=reg))

        model.add(Conv1D(output_dim, 3, padding='valid', activation='relu'))
        model.add(MaxPooling1D())

        model.add(Conv1D(2*output_dim, 3, padding='valid', activation='relu'))
        model.add(MaxPooling1D())

        model.add(Flatten())
        
        model.add(Dense(16, activation='relu'))
        model.add(Dropout(0.5))

        model.add(Dense(64, activation='relu'))
        model.add(Dropout(0.5))
        
        model.add(Dense(256, activation='relu'))
        model.add(Dropout(0.5))

        model.add(Dense(128, activation='relu'))
        model.add(Dropout(0.5))

        model.add(Dense(64, activation='relu'))
        model.add(Dropout(0.5))

        model.add(Dense(32, activation='relu'))
        model.add(Dropout(0.5))

        model.add(Dense(16, activation='relu'))
        model.add(Dropout(0.5))

        model.add(Dense(1, activation=last_act))

        return model



class RecurrentOne:
	
    @staticmethod
    def build(input_dim, output_dim, input_len, lstm_out, classes, reg, bidirectional=False, dropout=0.1, init="he_normal", last_act='softmax'):
        model = Sequential()

        model.add(Embedding(input_dim=input_dim, output_dim=output_dim, input_length=input_len,
                            embeddings_initializer=init, activity_regularizer=reg))

        model.add(LSTM(lstm_out))

        model.add(Dense(classes, activation=last_act))

        return model


class RecurrentTwo:
	
    @staticmethod
    def build(input_dim, output_dim, input_len, lstm_out, classes, reg, bidirectional=True, dropout=0.1, init="he_normal", last_act='softmax'):
        model = Sequential()

        model.add(Embedding(input_dim=input_dim, output_dim=output_dim, input_length=input_len,
                            embeddings_initializer=init, activity_regularizer=reg))

        model.add(Bidirectional(LSTM(lstm_out, return_sequences=True, dropout=dropout)))
        model.add(Bidirectional(LSTM(lstm_out)))

        model.add(Dense(classes, activation=last_act))

        return model


class RecurrentThree:
	
    @staticmethod
    def build(input_dim, output_dim, input_len, lstm_out, classes, reg, bidirectional=False, dropout=0.1, init="he_normal", last_act='softmax'):
        model = Sequential()

        model.add(Embedding(input_dim=input_dim, output_dim=output_dim, input_length=input_len,
                            embeddings_initializer=init, activity_regularizer=reg))

        model.add(LSTM(lstm_out, return_sequences=True, dropout=dropout))
        model.add(LSTM(lstm_out))

        model.add(Dense(classes, activation=last_act))

        return model


class RecurrentFour:
	
    @staticmethod
    def build(input_dim, output_dim, input_len, lstm_out, classes, reg, bidirectional=False, dropout=0.1, init="he_normal", last_act='softmax'):
        model = Sequential()

        model.add(Embedding(input_dim=input_dim, output_dim=output_dim, input_length=input_len,
                            embeddings_initializer=init, activity_regularizer=reg))

        if bidirectional:
            model.add(Bidirectional(LSTM(lstm_out, return_sequences=True, dropout=0.1)))
            model.add(Bidirectional(LSTM(lstm_out)))
        else:
            model.add(LSTM(lstm_out, return_sequences=True, dropout=dropout))
            model.add(LSTM(lstm_out))

        model.add(Dense(4*classes, activation='relu'))
        model.add(Dropout(dropout))

        model.add(Dense(classes, activation=last_act))

        return model

