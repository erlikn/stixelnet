from __future__ import absolute_import
from __future__ import division

import tensorflow as tf
import numpy as np

def add_loss_summaries(total_loss, batchSize):
    """Add summaries for losses in calusa_heatmap model.
    Generates moving average for all losses and associated summaries for
    visualizing the performance of the network.
    Args:
      total_loss: Total loss from loss().
    Returns:
      loss_averages_op: op for generating moving averages of losses.
    """
    # Compute the moving average of all individual losses and the total loss.
    loss_averages = tf.train.ExponentialMovingAverage(0.9, name='Average')
    losses = tf.get_collection('losses')
    loss_averages_op = loss_averages.apply(losses + [total_loss])

    # Individual average loss
#    lossPixelIndividual = tf.sqrt(tf.multiply(total_loss, 2/(batchSize*4))) # dvidied by (8/2) = 4 which is equal to sum of 2 of them then sqrt will result in euclidean pixel error
#    tf.summary.scalar('Average_Pixel_Error_Real', lossPixelIndividual)

    # Attach a scalar summary to all individual losses and the total loss; do the
    # same for the averaged version of the losses.
    for l in losses + [total_loss]:
        # Name each loss as '(raw)' and name the moving average version of the loss
        # as the original loss name.
        tf.summary.scalar(l.op.name + '_raw', l)
        tf.summary.scalar(l.op.name, loss_averages.average(l))

    return loss_averages_op

def _l2_loss(pred, tval): # batchSize=Sne
    """Add L2Loss to all the trainable variables.
    
    Add summary for "Loss" and "Loss/avg".
    Args:
      logits: Logits from inference().
      labels: Labels from distorted_inputs or inputs(). 1-D tensor
              of shape [batch_size, heatmap_size ]
    
    Returns:
      Loss tensor of type float.
    """
    #if not batch_size:
    #    batch_size = kwargs.get('train_batch_size')
    
    #l1_loss = tf.abs(tf.subtract(logits, HAB), name="abs_loss")
    #l1_loss_mean = tf.reduce_mean(l1_loss, name='abs_loss_mean')
    #tf.add_to_collection('losses', l2_loss_mean)

    l2_loss = tf.nn.l2_loss(tf.subtract(pred, tval), name="loss_l2")
    tf.add_to_collection('losses', l2_loss)

    #l2_loss_mean = tf.reduce_mean(l2_loss, name='l2_loss_mean')
    #tf.add_to_collection('losses', l2_loss_mean)

    #mse = tf.reduce_mean(tf.square(logits - HAB), name="mse")
    #tf.add_to_collection('losses', mse)

    # Calculate the average cross entropy loss across the batch.
    # labels = tf.cast(labels, tf.int64)
    # cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
    #     logits, labels, name='cross_entropy_per_example')
    # cross_entropy_mean = tf.reduce_mean(cross_entropy, name='cross_entropy')
    # tf.add_to_collection('losses', cross_entropy_mean)

    # The total loss is defined as the cross entropy loss plus all of the weight
    # decay terms (L2 loss).
    return tf.add_n(tf.get_collection('losses'), name='loss_total')


def _weighted_L2_loss(tMatP, tMatT, activeBatchSize):
    mask = np.array([[100, 100, 100, 1, 100, 100, 100, 1, 100, 100, 100, 1]], dtype=np.float32)
    mask = np.repeat(mask, activeBatchSize, axis=0)
    tMatP = tf.multiply(mask, tMatP)
    tMatT = tf.multiply(mask, tMatT)
    return _l2_loss(tMatP, tMatT)

def _weighted_params_L2_loss(targetP, targetT, activeBatchSize):
    # Alpha, Beta, Gamma are -Pi to Pi periodic radians - mod over pi to remove periodicity
    #mask = np.array([[np.pi, np.pi, np.pi, 1, 1, 1]], dtype=np.float32)
    #mask = np.repeat(mask, kwargs.get('activeBatchSize'), axis=0)
    #targetP = tf_mod(targetP, mask)
    #targetT = tf_mod(targetT, mask)
    # Importance weigting on angles as they have smaller values
    mask = np.array([[100, 100, 100, 1, 1, 1]], dtype=np.float32)
    #mask = np.array([[1000, 1000, 1000, 100, 100, 100]], dtype=np.float32)
    mask = np.repeat(mask, activeBatchSize, axis=0)
    targetP = tf.multiply(targetP, mask)
    targetT = tf.multiply(targetT, mask)
    return _l2_loss(targetP, targetT)

def _weighted_params_L2_loss_nTuple_last(targetP, targetT, nTuple, activeBatchSize):
    # Alpha, Beta, Gamma are -Pi to Pi periodic radians - mod over pi to remove periodicity
    #mask = np.array([[np.pi, np.pi, np.pi, 1, 1, 1]], dtype=np.float32)
    #mask = np.repeat(mask, kwargs.get('activeBatchSize'), axis=0)
    #targetP = tf_mod(targetP, mask)
    #targetT = tf_mod(targetT, mask)
    # Importance weigting on angles as they have smaller values
    mask = np.array([[100, 100, 100, 1, 1, 1]], dtype=np.float32)
    mask = np.repeat(mask, activeBatchSize, axis=0)
    targetP = tf.multiply(targetP, mask)
    targetT = tf.multiply(targetT, mask)
    return _l2_loss(targetP, targetT)

def _weighted_params_L2_loss_nTuple_all(targetP, targetT, nTuple, activeBatchSize):
    # Alpha, Beta, Gamma are -Pi to Pi periodic radians - mod over pi to remove periodicity
    #mask = np.array([[np.pi, np.pi, np.pi, 1, 1, 1]], dtype=np.float32)
    #mask = np.repeat(mask, kwargs.get('activeBatchSize'), axis=0)
    #targetP = tf_mod(targetP, mask)
    #targetT = tf_mod(targetT, mask)
    # Importance weigting on angles as they have smaller values
    mask = np.array([[100, 100, 100, 1, 1, 1]], dtype=np.float32)
    mask = np.repeat(mask, nTuple-1, axis=0).reshape((nTuple-1)*6)
    mask = np.repeat(mask, activeBatchSize, axis=0)
    targetP = tf.multiply(targetP, mask)
    targetT = tf.multiply(targetT, mask)
    return _l2_loss(targetP, targetT)

def _params_classification_l2_loss_nTuple(targetP, targetT, nTuple, activeBatchSize):
    '''
    TargetT dimensions are [activeBatchSize, rows=6, cols=32, nTuple]
    '''
    # Alpha, Beta, Gamma are -Pi to Pi periodic radians - mod over pi to remove periodicity
    #mask = np.array([[np.pi, np.pi, np.pi, 1, 1, 1]], dtype=np.float32)
    #mask = np.repeat(mask, kwargs.get('activeBatchSize'), axis=0)
    #targetP = tf_mod(targetP, mask)
    #targetT = tf_mod(targetT, mask)
    # Importance weigting on angles as they have smaller values
    #mask = np.array([[100, 100, 100, 1, 1, 1]], dtype=np.float32)
    #mask = np.repeat(mask, nTuple-1, axis=0).reshape((nTuple-1)*6)
    #mask = np.repeat(mask, activeBatchSize, axis=0)
    #targetP = tf.multiply(targetP, mask)
    #targetT = tf.multiply(targetT, mask)
    targetT = tf.cast(targetT, tf.float32)
    return _l2_loss(targetP, targetT)

def _params_classification_softmaxCrossentropy_loss_nTuple(targetP, targetT, nTuple, activeBatchSize):
    '''
    Takes in the targetP and targetT and calculates the softmax-cross entropy loss for each parameter
    and sums them for each instance and sum again for each tuple in the batch
    TargetT dimensions are [activeBatchSize, rows=6, cols=32, nTuple]
    '''
    targetT = tf.cast(targetT, tf.float32)
    targetP = tf.cast(targetP, tf.float32)
    ############################
    # Alternatively, instead of sum, we could use squared_sum to penalize harsher
    ############################
    # ---> [activeBatchSize, rows=6, cols=32, nTuple]
    # Calculate softmax-cross entropy loss for each parameter (cols dimension -> cols)
    # ---> [activeBatchSize, rows=6, nTuple]
    # Then calculate sum of parameter losses for each batch (last 2 dimensions -> ntuple, rows), and returns an array of [activeBatchSize] size
    # ---> [activeBatchSize]
    smce_loss = tf.nn.l2_loss(tf.nn.softmax_cross_entropy_with_logits(logits=targetP, labels=targetT, dim=2), name="loss_smce_l2")
    #smce_loss = tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits(logits=targetP, labels=targetT, dim=2), name="loss_smce_sum")
    return smce_loss

def loss(pred, tval, **kwargs):
    """
    Choose the proper loss function and call it.
    """
    lossFunction = kwargs.get('lossFunction')
    if lossFunction == 'L2':
        return _l2_loss(pred, tval)
    if lossFunction == 'Weighted_L2_loss':
        return _weighted_L2_loss(pred, tval, kwargs.get('activeBatchSize'))
    if lossFunction == 'Weighted_Params_L2_loss':
        return _weighted_params_L2_loss(pred, tval, kwargs.get('activeBatchSize'))
    if lossFunction == 'Weighted_Params_L2_loss_nTuple_last':
        return _weighted_params_L2_loss_nTuple_last(pred, tval, kwargs.get('numTuple'), kwargs.get('activeBatchSize'))
    if lossFunction == 'Weighted_Params_L2_loss_nTuple_all':
        return _weighted_params_L2_loss_nTuple_all(pred, tval, kwargs.get('numTuple'), kwargs.get('activeBatchSize'))
    if lossFunction == '_params_classification_l2_loss_nTuple':
        return _params_classification_l2_loss_nTuple(pred, tval, kwargs.get('numTuple'), kwargs.get('activeBatchSize'))
    if lossFunction == '_params_classification_softmaxCrossentropy_loss_nTuple':
        if kwargs.get('lastTuple'):
            return _params_classification_softmaxCrossentropy_loss_nTuple(pred, tval, 1, kwargs.get('activeBatchSize'))
        else:
            return _params_classification_softmaxCrossentropy_loss_nTuple(pred, tval, kwargs.get('numTuple'), kwargs.get('activeBatchSize'))
            
