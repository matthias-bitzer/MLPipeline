import tensorflow as tf
import log


def fully_connected(input,dict,num):
    units = dict["units"]
    activation = dict["activation"]
    with tf.variable_scope(__name__+str(num)):
        with tf.variable_scope("fully_connected"):
            output = tf.layers.flatten(inputs=input)
            output = tf.layers.dense(inputs=output,units=units,activation=activation,kernel_initializer=tf.truncated_normal_initializer,reuse=tf.AUTO_REUSE)
            log.build_log(str(output.get_shape().as_list())+" --- "+str(tf.get_variable_scope().name))
    return output


def conv_conv_max_3x3(input,dict,num):
    filters = dict["filters"]
    stride = dict["stride"]
    with tf.variable_scope(__name__+str(num)):
        with tf.variable_scope('conv1') as scope:
            output_conv_1 = tf.layers.conv2d(inputs=input,filters=filters,kernel_size=3,strides=[stride,stride],padding="SAME",kernel_initializer=tf.truncated_normal_initializer,reuse=False)
            log.build_log(str(output_conv_1.get_shape().as_list())+" --- "+str(tf.get_variable_scope().name))
        with tf.variable_scope('conv2'):
            output_conv_2 = tf.layers.conv2d(inputs=output_conv_1,filters=filters,kernel_size=3,strides=[stride,stride],padding="SAME",kernel_initializer=tf.truncated_normal_initializer,reuse=False)
            log.build_log(str(output_conv_2.get_shape().as_list())+" --- "+str(tf.get_variable_scope().name))
        with tf.variable_scope('maxpool'):
            output = tf.layers.max_pooling2d(inputs=output_conv_2,pool_size=2,strides=[2,2],padding='SAME')
            log.build_log(str(output.get_shape().as_list())+" --- "+str(tf.get_variable_scope().name))
    return output


def conv_conv_max_5x5(input,dict,num):
    filters = dict["filters"]
    stride = dict["stride"]
    with tf.variable_scope(__name__+str(num)):
        with tf.variable_scope('conv1'):
            output_conv_1 = tf.layers.conv2d(inputs=input,filters=filters,kernel_size=5,strides=[stride,stride],padding="SAME",kernel_initializer=tf.truncated_normal_initializer,reuse=False)
            log.build_log(str(output_conv_1.get_shape().as_list()))
        with tf.variable_scope('conv2'):
            output_conv_2 = tf.layers.conv2d(inputs=output_conv_1,filters=filters,kernel_size=5,strides=[stride,stride],padding="SAME",kernel_initializer=tf.truncated_normal_initializer,reuse=False)
            log.build_log(str(output_conv_2.get_shape().as_list()))
        with tf.variable_scope('maxpool'):
            output = tf.layers.max_pooling2d(inputs=output_conv_2,pool_size=2,strides=[2,2],padding='SAME')
            log.build_log(str(output.get_shape().as_list()))
    return output


def conv_conv_max_8x8(input,dict,num):
    filters = dict["filters"]
    stride = dict["stride"]
    with tf.variable_scope(__name__+str(num)):
        with tf.variable_scope('conv1'):
            output_conv_1 = tf.layers.conv2d(inputs=input,filters=filters,kernel_size=8,strides=[stride,stride],padding="SAME",kernel_initializer=tf.truncated_normal_initializer,reuse=False)
            log.build_log(str(output_conv_1.get_shape().as_list()))
        with tf.variable_scope('conv2'):
            output_conv_2 = tf.layers.conv2d(inputs=output_conv_1,filters=filters,kernel_size=8,strides=[stride,stride],padding="SAME",kernel_initializer=tf.truncated_normal_initializer,reuse=False)
            log.build_log(str(output_conv_2.get_shape().as_list()))
        with tf.variable_scope('maxpool'):
            output = tf.layers.max_pooling2d(inputs=output_conv_2,pool_size=2,strides=[2,2],padding='SAME')
            log.build_log(str(output.get_shape().as_list()))
    return output


