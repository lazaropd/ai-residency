
import tensorflow as tf
from tensorflow import keras

import numpy as np
import os

from urllib.parse import urlparse

import mlflow
import mlflow.keras



mlflow.set_experiment("fashion")


fashion_mnist = keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

# scale the values to 0.0 to 1.0
train_images = train_images / 255.0
test_images = test_images / 255.0

# reshape for feeding into the model
train_images = train_images.reshape(train_images.shape[0], 28*28)
test_images = test_images.reshape(test_images.shape[0], 28*28)

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

model = keras.Sequential([
  keras.layers.Reshape((28, 28, 1), input_shape=(28*28,)),
  keras.layers.Conv2D(input_shape=(28,28,1), filters=8, kernel_size=3, 
                      strides=2, activation='relu', name='Conv1'),
  keras.layers.Flatten(),
  keras.layers.Dense(10, activation=tf.nn.softmax, name='Softmax')
])

testing = False
epochs = 10

model.compile(optimizer='adam', 
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
model.fit(train_images, train_labels, epochs=epochs)

test_loss, test_acc = model.evaluate(test_images, test_labels)
print('\nTest accuracy: {}'.format(test_acc))


with mlflow.start_run(run_name='Adam Sparse - no HL') as run:

  mlflow.log_param('strides', 2)
  mlflow.log_metric("loss", test_loss)
  mlflow.log_metric("acc", test_acc)

  tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme
  # Model registry does not work with file store
  if tracking_url_type_store != "file":
      mlflow.keras.log_model(model, "model", registered_model_name="fashionmnist")
  else:
      mlflow.keras.log_model(model, "model")

  run_id = run.info.run_uuid
  experiment_id = run.info.experiment_id
  mlflow.end_run()
  print(mlflow.get_artifact_uri())
  print("runID: %s | experimentID: %s" % (run_id, experiment_id))


