
import tensorflow as tf


print(tf.config.list_physical_devices('GPU'))

print(('Is your GPU available for use?\n{0}').format(
    'Yes, your GPU is available: True' if tf.test.is_gpu_available() == True else 'No, your GPU is NOT available: False'
))

print(('\nYour devices that are available:\n{0}').format(
    [device.name for device in tf.config.experimental.list_physical_devices()]
))


import time

cpu_slot = 0
gpu_slot = 0

# Using CPU at slot 0
with tf.device('/CPU:' + str(cpu_slot)):
    # Starting a timer
    start = time.time()

    # Doing operations on CPU
    print(tf.eye(10000,10000))

    # Printing how long it took with CPU
    end = time.time() - start
    print("CPU: ", end)

# Using the GPU at slot 0
with tf.device('/GPU:' + str(gpu_slot)):
    # Starting a timer
    start = time.time()

    # Doing operations on CPU

    print(tf.eye(10000,10000))

    # Printing how long it took with CPU
    end = time.time() - start
    print("GPU: ",end)


print('Test ended')