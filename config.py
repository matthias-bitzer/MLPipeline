import layers
import tensorflow as tf

class Config(object):

    def __init__(self):
        self.input_dim = [None,28,28,1]
        self.output_dim = [None,10]
        self.batch_size= 60
        self.epochs = 30
        self.architecture_dict = {"layers":[(layers.conv_conv_max_3x3,{"filters":10,"stride":1},False,0),(layers.conv_conv_max_3x3,{"filters":60,"stride":1},False,0),(layers.fully_connected,{"units":40,"activation":tf.nn.relu},False,0),(layers.fully_connected,{"units":10,"activation":tf.nn.sigmoid},False,0)]}
        self.step_size = 1e-5
        self.model_name = "model1"