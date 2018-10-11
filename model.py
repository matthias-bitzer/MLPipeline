import tensorflow as tf
import layers
import forward
import PATH
import dataLoader
import log

class Model(object):

    def __init__(self,config):
        self.input_dim = config.input_dim
        self.output_dim = config.output_dim
        self.architecture_dict = config.architecture_dict
        self.step_size = config.step_size
        self.graph = tf.Graph()
        self.sess = tf.Session(graph=self.graph)
        self.model_name = config.model_name
        self.model_save_path= PATH.save_path+"/"+self.model_name
        self.epochs = config.epochs
        self.batch_size = config.batch_size
        self.graph_build = False

    def __del__(self):
        self.sess.close()

    def build_graph(self):
        with self.graph.as_default():
            with tf.name_scope("input") as scope:
                input = tf.placeholder(dtype=tf.float32, shape=self.input_dim, name=scope)
                log.build_log(str(input.get_shape().as_list()) + " --- " + str(input.name))
            with tf.name_scope("label") as scope:
                label = tf.placeholder(dtype=tf.float32, shape=self.output_dim, name=scope)
            with tf.name_scope("forward_pass"):
                output = forward.forward_pass(input,self.architecture_dict)
            with tf.name_scope("output") as scope:
                output = tf.identity(output,name=scope)
                log.build_log(str(output.get_shape().as_list()) + " --- " + str(output.name))
            with tf.name_scope("loss") as scope:
                loss = tf.losses.mean_squared_error(labels=label,predictions=output)
                loss = tf.identity(loss,name=scope)
            with tf.name_scope("optimizer") as scope:
                optimizer = tf.train.AdamOptimizer(learning_rate=self.step_size,name=scope).minimize(loss)
            tf.train.export_meta_graph(self.model_save_path + "/methagraph")
            self.saver = tf.train.Saver(max_to_keep=0)
            init_op = tf.global_variables_initializer()
            self.sess.run(init_op)
            self.graph_build = True



    def load_graph(self):
        with self.graph.as_default():
            tf.train.import_meta_graph(self.model_save_path + "/methagraph")
            self.saver = tf.train.Saver(max_to_keep=0)
            init_op = tf.global_variables_initializer()
            self.sess.run(init_op)
            self.graph_build = True


    def load_weights(self):
        self.saver.restore(self.sess,self.model_save_path+"/checkpoint")

    def predict(self,input):
        input_tensor =  self.graph.get_tensor_by_name("input")
        output_tensor = self.graph.get_tensor_by_name("output")
        prediction = self.sess.run(output_tensor,feed_dict={input_tensor: input})
        return prediction

    def train(self):
        if(not self.graph_build):
            raise Exception("Graph has to be build before Training")
        input_tensor =  self.graph.get_tensor_by_name("input:0")
        label_tensor = self.graph.get_tensor_by_name("label:0")
        loss_tensor = self.graph.get_tensor_by_name("loss:0")
        optimizer_operation = self.graph.get_operation_by_name("optimizer")
        self.data_loader = dataLoader.DataLoader()
        log.train_log("Training of model "+self.model_name+" started")

        for epoch in range(self.epochs):
            epoch_loss = 0
            n=int(self.data_loader.train_size/self.batch_size)
            for i in range(n):
                input_batch, label_batch = self.data_loader.get_batch(self.batch_size)
                _,loss = self.sess.run([optimizer_operation,loss_tensor],feed_dict={input_tensor: input_batch,label_tensor: label_batch})
                epoch_loss += loss
            log.train_log("Epoch "+str(epoch)+": "+str(epoch_loss/(n*self.batch_size)))
            self.saver.save(self.sess,self.model_save_path+"/checkpoint",global_step=epoch)

        log.train_log("Training of model "+self.model_name+" finished")












