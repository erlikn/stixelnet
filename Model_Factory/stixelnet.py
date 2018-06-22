# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Builds the calusa_heatmap network.

Summary of available functions:

 # Compute input images and labels for training. If you would like to run
 # evaluations, use inputs() instead.
 inputs, labels = inputs()

 # Compute inference on the model inputs to make a prediction.
 predictions = inference(inputs)

 # Compute the total loss of the prediction with respect to the labels.
 loss = loss(predictions, labels)

 # Create a graph to run one step of training with respect to the loss.
 train_op = train(loss, global_step)
"""
# pylint: disable=missing-docstring
from __future__ import absolute_import
from __future__ import division

import tensorflow as tf
import numpy as np

import Model_Factory.model_base as model_base

USE_FP_16 = False

# If a model is trained with multiple GPUs, prefix all Op names with tower_name
# to differentiate the operations. Note that this prefix is removed from the
# names of the summaries when visualizing a model.
TOWER_NAME = 'tower'

def inference(images, **kwargs): #batchSize=None, phase='train', outLayer=[13,13], existingParams=[]
    modelShape = kwargs.get('modelShape')
    wd = None #0.0002
    USE_FP_16 = kwargs.get('usefp16')
    dtype = tf.float16 if USE_FP_16 else tf.float32

    batchSize = kwargs.get('activeBatchSize', None)

    #######################################
    ############# CONV_1
    fireOut, prevExpandDim = model_base.conv_fire_module('conv1', images, kwargs.get('imageDepthChannels'),
                                                                  {'cnn11x5': modelShape[0]},
                                                                  wd, **kwargs)
    # # calc batch norm CONV_1
    # if kwargs.get('batchNorm'):
    #     fireOut = model_base.batch_norm('batchnorm1', fireOut, dtype)

    ###### Pooling1 8x4 with stride 8x4
    fireOut = tf.nn.max_pool(fireOut, ksize=[1, 8, 4, 1], strides=[1, 8, 4, 1],
                          padding='SAME', name='maxpool1')
    #######################################
    #######################################
    ############# CONV_2
    fireOut, prevExpandDim = model_base.conv_fire_module('conv2', fireOut, prevExpandDim,
                                                                  {'cnn3x5': modelShape[1]},
                                                                  wd, **kwargs)
    # # calc batch norm CONV_2
    # if kwargs.get('batchNorm'):
    #     fireOut = model_base.batch_norm('batchnorm2', fireOut, dtype)

    ###### Pooling1 8x4 with stride 8x4
    fireOut = tf.nn.max_pool(fireOut, ksize=[1, 4, 3, 1], strides=[1, 4, 3, 1],
                          padding='SAME', name='maxpool2')

    #######################################
    ###### Prepare for fully connected layers
    # Reshape firout - flatten
    prevExpandDim = (kwargs.get('imageDepthRows')//(2*2*2))*(kwargs.get('imageDepthCols')//(2*2*2))*prevExpandDim
    fireOut = tf.reshape(fireOut, [batchSize, -1])

    #######################################
    ############# FC1 layer with 1024 outputs
    fireOut, prevExpandDim = model_base.fc_fire_module('fc1', fireOut, prevExpandDim,
                                                       {'fc': modelShape[2]},
                                                       wd, **kwargs)
    # calc batch norm FC1
    #if kwargs.get('batchNorm'):
    #    fireOut = model_base.batch_norm('batchnorm9', fireOut, dtype)
    ###### DROPOUT after FC1
    with tf.name_scope("drop"):
        keepProb = tf.constant(kwargs.get('dropOutKeepRate') if kwargs.get('phase') == 'train' else 1.0, dtype=dtype)
        fireOut = tf.nn.dropout(fireOut, keepProb, name="dropout")
    
    #######################################
    #######################################
    ############# FC2 layer with 2048 outputs
    fireOut, prevExpandDim = model_base.fc_fire_module('fc2', fireOut, prevExpandDim,
                                                             {'fc': modelShape[3]},
                                                             wd, **kwargs)
    ###### DROPOUT after FC2
    with tf.name_scope("drop"):
        keepProb = tf.constant(kwargs.get('dropOutKeepRate') if kwargs.get('phase') == 'train' else 1.0, dtype=dtype)
        fireOut = tf.nn.dropout(fireOut, keepProb, name="dropout")

    #######################################
    #######################################
    ############# FC3 layer with 50 outputs ---- OUTPUT LAYER
    fireOut, prevExpandDim = model_base.fc_fire_module('fc3_output', fireOut, prevExpandDim,
                                                             {'fc': kwargs.get('networkOutputSize')},
                                                             wd, **kwargs)
    #######################################
    return fireOut

def loss(pred, target, **kwargs): # batchSize=Sne
    """Add L2Loss to all the trainable variables.
    Add summary for "Loss" and "Loss/avg".
    Args:
      logits: Logits from inference().
      labels: Labels from distorted_inputs or inputs(). 1-D tensor
              of shape [batch_size, heatmap_size ]
    Returns:
      Loss tensor of type float.
    """
    return model_base.loss(pred, target, **kwargs)

def train(loss, globalStep, **kwargs):
    return model_base.train(loss, globalStep, **kwargs)

def test(loss, globalStep, **kwargs):
    return model_base.test(loss, globalStep, **kwargs)
