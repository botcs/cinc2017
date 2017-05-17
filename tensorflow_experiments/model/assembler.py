import tensorflow as tf
import json
import resnet
import cnn
import rnn
import classifier
from misc import variable_size_window


def get_model_logits(seq_len, input_op, **params):
    ####################################
    # Variable length feature extraction
    var_features = seq_len, input_op
    if params.get('resnet'):
        var_features = resnet.get_resnet_output(
            *var_features, **params['resnet'])

    elif params.get('fcn'):
        var_features = cnn.get_output(
            *var_features, return_name=False, **params['fcn'])

    ####################################
    # Fixed length feature extraction
    if params.get('rnn'):
        pass
    elif params.get('partition_num'):
        # Instead of global averaging,
        # Use variable sized windows to make equal fractions of input
        # Preserving the raw locality of the features
        var_features = variable_size_window(
            *var_features, params['partition_num'])
        # seq_len is no longer needed, therefore discarded
        features = tf.reduce_mean(var_features[1], axis=2)
        print('Local reduce:')
        print(features)

    classifier_in = tf.contrib.layers.flatten(features)
    fc_params = params['fc_params']
    logits = classifier.get_logits_and_pred(
        classifier_in, return_name=False, **fc_params)[0]
    return logits
