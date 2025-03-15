import numpy as np
import tensorflow as tf
#import os

tf.__version__
#os.environ['MY_VARIABLE'] = 'my_value'

##---------------------------------------------------
# 1. Initialize parameters variable and optimizer with tensorflow
# ** is square
# Call keras API, class optimizers from tensorflow
# GradientTape() use tape for record forward propagration, reverse the recorded value to perform backward propagation
# Adam(0.1) is optimizer set to Adam algorithm with learning_rate 0.1
# tape.gradient() computes gradient using input cost and set of trainable variables to be computed
# zip takes 2 arguments and pairing them
# optimizer.apply_gradients() to compute gradient descent and update variables with gradients

w = tf.Variable(initial_value=0, trainable=None, dtype=tf.float32, shape=None)
optimizer = tf.keras.optimizers.Adam(learning_rate=0.1)

def train_step():
  with tf.GradientTape() as tape:
    cost = w ** 2 - 10 * w + 25
  trainable_variables = [w]
  grads = tape.gradient(cost, trainable_variables)
  optimizer.apply_gradients(zip(grads, trainable_variables))

print(w)

train_step()
print(w)

for i in range(1000):
  train_step()
print(w) # w reach almost 5.0000001, which is almost minimum value for the cost function

#---------------------------------------------------------------------------------------
# 2. Tensorflow application in computing cost function and update variables with input X
# Call optimizers method: RMSprop(), Adam(), SGD(,momentum=beta), SGD()
# optimizer.minimize(cost, variable) will minimize the cost based on input variable
# Sketch the computation graph, compute cost_func in forward propagation with given equation
# Then, Tensorflow automatically determine all backward propagation to calculate gradients, optimize the variables
"""
w = tf.Variable(initial_value=0.0, trainable=True, dtype=tf.float32)  # Make w trainable
x = np.array([1.0, -10.0, 25.0], dtype=np.float32)
optimizer = tf.keras.optimizers.RMSprop(learning_rate=0.1)

def cost_func():
    cost = (x[0] * w) ** 2 - (x[1] * w) + x[2]
    return cost

def training(x, w, optimizer):
    for i in range(1000):
        optimizer.minimize(cost_func, [w])  # Pass the function itself
    return w

w = training(x, w, optimizer)
print(w)"
"""

#----------------------------------------------------------------------------------------------
# 3. Test Optimizer with Tensorflow and Keras API

w = tf.Variable(initial_value=0.0, trainable=True, dtype=tf.float32)  # Corrected: trainable=True
x = np.array([1.0, -10.0, 25.0], dtype=np.float32)
optimizer = tf.keras.optimizers.Adam(learning_rate=0.1)

def cost_func():
    cost = (x[0] * w) ** 2 - (x[1] * w) + x[2]
    return cost

def training(x, w, optimizer):
    for i in range(1000):
        optimizer.minimize(cost_func, [w])  # Corrected: Call the function
    return w

w = training(x, w, optimizer)
print(w)

final_cost = cost_func().numpy()
print(f"Final Cost: {final_cost}")