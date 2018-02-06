# Python 3.4
import sys
sys.path.append("/usr/local/lib/python3.4/site-packages/")
import cv2 as cv2
from os import listdir
from os.path import isfile, join
from os import walk
import os
import json
import collections
import math
import random
from shutil import copy
import numpy as np
import matplotlib.pyplot as plt
import csv
import tensorflow as tf

from joblib import Parallel, delayed
import multiprocessing

import Data_IO.tfrecord_io as tfrecord_io
import Data_IO.kitti_shared as kitti

def _apply_prediction(pclB, targetT, targetP, **kwargs):
    '''
    Transform pclB, Calculate new targetT based on targetP, Create new depth image
    Return:
        - New PCLB
        - New targetT
        - New depthImageB
    '''
    # remove trailing zeros
    pclA = kitti.remove_trailing_zeros(pclB)
    # get transformed pclB based on targetP
    tMatP = kitti._get_tmat_from_params(targetP) 
    pclBTransformed = kitti.transform_pcl(pclB, tMatP)
    # get new depth image of transformed pclB
    depthImageB, _ = kitti.get_depth_image_pano_pclView(pclBTransformed)
    pclBTransformed = kitti._zero_pad(pclBTransformed, kwargs.get('pclCols')-pclBTransformed.shape[1])
    # get residual Target
    #tMatResB2A = kitti.get_residual_tMat_Bp2B2A(targetP, targetT) # first is A, second is B
    targetResP2T = targetT - targetP
    return pclBTransformed, targetResP2T, depthImageB

def output_clsf(batchImages, batchPcl, bTargetT, bTargetP, bRngs, batchTFrecFileIDs, **kwargs):
    """
    TODO: SIMILAR TO DATA INPUT -> WE NEED A QUEUE RUNNER TO WRITE THIS OFF TO BE FASTER

    Everything evaluated
    Warp second image based on predicted HAB and write to the new address
    Args:
    Returns:
    Raises:
      ValueError: If no dataDir
    """
    num_cores = multiprocessing.cpu_count() - 2
    Parallel(n_jobs=num_cores)(delayed(output_loop_clsf)(batchImages, batchPcl, bTargetT, bTargetP, bRngs, batchTFrecFileIDs, i, **kwargs) for i in range(kwargs.get('activeBatchSize')))
    #for i in range(kwargs.get('activeBatchSize')):
    #    output_loop(batchImages, batchPcl, bTargetT, bTargetP, bRngs, batchTFrecFileIDs, i, **kwargs)
    return

def output_loop_clsf(batchImages, batchPcl, bTargetT, bTargetP, bRngs, batchTFrecFileIDs, i, **kwargs):
    """
    TODO: SIMILAR TO DATA INPUT -> WE NEED A QUEUE RUNNER TO WRITE THIS OFF TO BE FASTER

    Everything evaluated
    Warp second image based on predicted HAB and write to the new address
    Args:
        batchImages:        batch of depth images
        batchPcl:           batch of PCL files
        bTargetT:           batch of target logits (binary)
        bTargetP:           batch of predicted logits (binary)
        bRngs:              barch of ranges
        batchTFrecFileIDs:  batch of tf_record_IDs
        i:                  ID of the batch
        kwargs:             Model parameters
    Returns:
        N/A
    Raises:
        ValueError: If no dataDir
    """
    numTuples = kwargs.get('numTuple')
    # Two tasks:
    #   1 - Use the predicted bTargetP[b, 6, 32, nt] and brngs[b, 6, 33, nt] to get the parameters
    #   2 - Update the rngs based on the predicted values for each image and save them

    # find argmax for the bTargeP and use it to get the corresponding params
    if modelParams.get('lastTuple'):
        # for training on last tuple
        predParam = kitti.get_params_from_binarylogits(bTargetP[i], bRngs[i,:,:,numTuples-2:numTuples-1])
        # get updated ranges
        newRanges = kitti.get_updated_ranges(bTargetP[i], bRngs[i,:,:,numTuples-2:numTuples-1])
    else:
        # for training on all tuples
        predParam = kitti.get_params_from_binarylogits(bTargetP[i], bRngs[i])
        # get updated ranges
        newRanges = kitti.get_updated_ranges(bTargetP[i], bRngs[i])
    # Apply the prediction from extracted parameters 
    

    # split for depth dimension
    pclBTransformed, targetRes, depthBTransformed = _apply_prediction(batchPcl[i,:,:,numTuples-1], bTargetT[i,:,numTuples-2], bTargetP[i], **kwargs)
    outBatchPcl = batchPcl.copy()
    outBatchImages = batchImages.copy()
    outTargetT = bTargetT.copy()
    outBatchPcl[i,:,:,numTuples-1] = pclBTransformed
    outBatchImages[i,:,:,numTuples-1] = depthBTransformed
    outTargetT[i,:,numTuples-2] = targetRes
    # Write each Tensorflow record
    filename = str(batchTFrecFileIDs[i][0]+100) + "_" + str(batchTFrecFileIDs[i][1]+100000) + "_" + str(batchTFrecFileIDs[i][2]+100000)
    tfrecord_io.tfrecord_writer_ntuple(batchTFrecFileIDs[i],
                                       outBatchPcl[i],
                                       outBatchImages[i],
                                       outTargetT[i],
                                       kwargs.get('warpedOutputFolder')+'/',
                                       numTuples,
                                       filename)

    if kwargs.get('phase') == 'train':
        folderTmat = kwargs.get('tMatTrainDir')
    else:
        folderTmat = kwargs.get('tMatTestDir')
    write_predictions(batchTFrecFileIDs[i], bTargetP[i], folderTmat)
    return

def write_json_file(filename, datafile):
    filename = 'Model_Settings/../'+filename
    datafile = collections.OrderedDict(sorted(datafile.items()))
    with open(filename, 'w') as outFile:
        json.dump(datafile, outFile, indent = 0)

def _set_folders(folderPath):
    if not os.path.exists(folderPath):
        os.makedirs(folderPath)

def write_predictions(tfrecID, targetP, folderOut):
    """
    Write prediction outputs to generate path map
    """
    _set_folders(folderOut)
    dataJson = {'seq' : tfrecID[0].tolist(),
                'idx' : tfrecID[1].tolist(),
                'idxNext' : tfrecID[2].tolist(),
                'tmat' : targetP.tolist()}
    write_json_file(folderOut + '/' + str(tfrecID[0]) + '_' + str(tfrecID[1]) + '_' + str(tfrecID[2]) +'.json', dataJson)
    return
