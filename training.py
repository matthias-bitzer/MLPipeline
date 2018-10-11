import model


def initial_training(config):
    model_object = model.Model(config)
    model_object.build_graph()
    model_object.train()

def continue_training(config):
    model_object = model.Model(config)
    model_object.load_graph()
    model_object.load_weights()
    model_object.train()

def show_graph(config):
    model_object = model.Model(config)
    model_object.build_graph()
