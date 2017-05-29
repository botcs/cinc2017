import tensorflow as tf
import json
from . import resnet
from . import cnn
from . import rnn
from . import classifier
from .misc import variable_size_window


def get_model_logits(seq_len, input_op, **params):
    is_training = tf.Variable(True, trainable=False, name='is_training')
    tf.add_to_collection('inference_vars', is_training)
    ####################################
    # Variable length feature extraction
    var_features = seq_len, input_op
    if params.get('resnet'):
        with tf.variable_scope('ResNet'):
            var_features = resnet.get_resnet_output(
                var_features[0], var_features[1], **params['resnet'])

    elif params.get('fcn'):
        with tf.variable_scope('FCN'):
            var_features = cnn.get_output(
                var_features[0], var_features[1],
                return_name=False, **params['fcn'])

    ####################################
    # Fixed length feature extraction
    if params.get('rnn'):
        pass
    elif params.get('partition_num'):
        # Instead of global averaging,
        # Use variable sized windows to make equal fractions of input
        # Preserving the raw locality of the features
        var_features = variable_size_window(
            var_features[0], var_features[1], params['partition_num'])
        # seq_len is no longer needed, therefore discarded
        features = tf.reduce_mean(var_features[1], axis=2)
        tf.add_to_collection('activations', features)
        print('Local reduce:')
        print(features)
    else:
        raise ValueError('No fixed length feature extraction specified')

    classifier_in = tf.contrib.layers.flatten(features)
    fc_params = params['fc_params']
    logits = classifier.get_logits_and_pred(
        classifier_in, return_name=False, **fc_params)[0]
    return logits
