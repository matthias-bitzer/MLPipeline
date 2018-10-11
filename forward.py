import tensorflow as tf

def forward_pass(forward,architecture_dict):
    layer_list = architecture_dict["layers"]

    layer_tensor_list = [forward]
    layer_num = 0
    for layer_tuple in layer_list:
        layer, dict, add_identity, index_of_layer_for_identity = layer_tuple
        forward = layer(forward,dict,layer_num)
        if add_identity:
            forward = forward + layer_tensor_list[index_of_layer_for_identity]
        layer_tensor_list.append(forward)
        layer_num+=1

    return forward
