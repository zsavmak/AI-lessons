#tensorflow_net.py

import tensorflow as tf
from math import pi

class Net:
    def __init__(self):
        self.layers = []
    
    def add(self,l):
        self.layers.append(l)
        
    def forward(self,x):
        for l in self.layers:
            x = l.forward(x)
        return x
    
    def backward(self,z):
        for l in self.layers[::-1]:
            z = l.backward(z)
        return z
    
    def update(self, lr):
        for l in self.layers:
            if hasattr(l, "update"):
                l.update(lr)


class PerceptronLayer:
    def __init__(self, in_features, out_features, name="fc"):
        self.weights = tf.Variable(
            tf.random.normal([in_features, out_features], stddev=0.01), 
            name=f"{name}_w"
        )
        self.bias = tf.Variable(tf.zeros([out_features]), name=f"{name}_b")
        self.input = None
        self.output = None
    
    def forward(self, x):
        self.input = x
        self.output = output = tf.add(tf.matmul(x, self.weights), self.bias)
        return self.output
    
    def backward(self, grad_output):
        self.grad_weights = tf.matmul(tf.transpose(self.input), grad_output)
        self.grad_bias = tf.reduce_sum(grad_output, axis=0)
        grad_input = tf.matmul(grad_output, tf.transpose(self.weights))
        return grad_input
    
    def update(self, lr):
        self.weights.assign_sub(lr * self.grad_weights)
        self.bias.assign_sub(lr * self.grad_bias)


class CrossEntropyLoss:
    def forward(self,logits,y):
        self.y = y
        z = tf.subtract(logits, tf.reduce_max(logits, axis=1, keepdims=True))
        log_sum_exp = tf.math.log(tf.reduce_sum(tf.exp(z), axis=1, keepdims=True))
        self.log_probs = tf.subtract(z, log_sum_exp)
        loss = -tf.reduce_mean(tf.gather_nd(self.log_probs, tf.stack([tf.range(tf.shape(y)[0]), y], axis=1)))
        return loss
    def backward(self):
        if self.log_probs is None:
            raise RuntimeError("Необходимо сначала выполнить forward pass.")
        probs = tf.exp(self.log_probs) # N×C
        indices = tf.stack([tf.range(tf.shape(self.y)[0]), self.y], axis=1) # N×2
        scatter = tf.scatter_nd(
            indices = indices, 
            updates=tf.ones_like(self.y, dtype=tf.float32),
            shape=tf.shape(probs)
        )
        grad = (probs - scatter) / tf.cast(tf.shape(self.y)[0], tf.float32)
        return grad


class Softmax:
    def __call__(self, z):
        z_max = tf.reduce_max(z, axis=1, keepdims=True)
        exp_z = tf.exp(z - z_max)
        return exp_z / tf.reduce_sum(exp_z, axis=1, keepdims=True)


class ReLULayer:
    def __init__(self):
        self.input = None
    def forward(self, x):
        self.input = x
        return tf.maximum(0, x)
    def backward(self, grad_output):
        return grad_output * tf.cast(self.input > 0, tf.float32)



class GELULayer:
    def __init__(self):
        self.input = None

    def forward(self, x):
        self.input = x
        sqrt_2 = tf.sqrt(tf.constant(2.0, dtype=x.dtype))
        return 0.5 * x * (1.0 + tf.math.erf(x / sqrt_2))

    def backward(self, grad_output):
        if self.input is None:
            raise RuntimeError("Необходимо сначала выполнить forward pass.")

        x = self.input
        sqrt_2 = tf.sqrt(tf.constant(2.0, dtype=x.dtype))
        sqrt_2_pi = tf.sqrt(tf.constant(2.0 / pi, dtype=x.dtype))

        erf_derivative = sqrt_2_pi * tf.exp(-0.5 * tf.square(x))

        grad_gelu = 0.5 * (1.0 + tf.math.erf(x / sqrt_2)) + 0.5 * x * erf_derivative
        return grad_output * grad_gelu