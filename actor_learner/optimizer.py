import tensorflow as tf

from tools import Gradient

class Optimzer:
    def __init__(self, type, FLAGS, use_gradient=False):
        self.type = type
        self.FLAG = FLAGS
        self.use_gradient = use_gradient
    
    def __call__(self, learning_rate):
        opt = None

        if self.type == 'rms':
            opt = tf.train.RMSPropOptimizer(learning_rate, self.FLAG.decay, self.FLAG.momentum, self.FLAG.epsilon)

        elif self.type == 'adam':
            opt = tf.train.AdamOptimizer(learning_rate)
        
        if self.use_gradient:
            opt = Gradient(opt)
        
        assert opt != None
        
        return opt
