import tensorflow as tf; 
print(tf.__version__)

# Some Matrix A
A = tf.constant([[3, 7],
                 [1, 9]])

A = tf.transpose(A)

print(('The transposed matrix A:\n{0}').format(A))
