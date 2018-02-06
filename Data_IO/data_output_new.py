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

def _apply_prediction_tmat(pclA, targetT, targetP, **kwargs):
    '''
    Transform pclA, Calculate new targetT based on targetP, Create new depth image
    Return:
        - New PCLA
        - New targetT
        - New depthImage
    '''
    # remove trailing zeros
    pclA = kitti.remove_trailing_zeros(pclA)
    # get transformed pclA based on targetP
    tMatP = kitti._get_tmat_from_params(targetP) 
    pclATransformed = kitti.transform_pcl(pclA, tMatP)
    # get new depth image of transformed pclA
    depthImageA, _ = kitti.get_depth_image_pano_pclView(pclATransformed)
    pclATransformed = kitti._zero_pad(pclATransformed, kwargs.get('pclCols')-pclATransformed.shape[1])
    # get residual Target
    tMatT = kitti._get_tmat_from_params(targetT) 
    tMatResA2B = kitti.get_residual_tMat_A2B(tMatT.reshape([12]), tMatP.reshape([12]))
    targetResP2T = kitti._get_params_from_tmat(tMatResA2B)
    return pclATransformed, targetResP2T, depthImageA

def _apply_prediction_periodic(pclA, targetT, targetP, **kwargs):
    '''
    Transform pclA, Calculate new targetT based on targetP, Create new depth image
    Return:
        - New PCLA
        - New targetT
        - New depthImage
    '''
    # remove trailing zeros
    pclA = kitti.remove_trailing_zeros(pclA)
    # get transformed pclA based on targetP
    tMatP = kitti._get_tmat_from_params(targetP) 
    pclATransformed = kitti.transform_pcl(pclA, tMatP)
    # get new depth image of transformed pclA
    depthImageA, _ = kitti.get_depth_image_pano_pclView(pclATransformed)
    pclATransformed = kitti._zero_pad(pclATransformed, kwargs.get('pclCols')-pclATransformed.shape[1])
    # get residual Target
    #tMatResA2B = kitti.get_residual_tMat_A2B(targetT, targetP)
    targetP[0] = targetP[0]%np.pi
    targetP[1] = targetP[1]%np.pi
    targetP[2] = targetP[2]%np.pi
    targetResP2T = targetT - targetP
    return pclATransformed, targetResP2T, depthImageA

def _apply_prediction(pclA, targetT, targetP, **kwargs):
    '''
    Transform pclA, Calculate new targetT based on targetP, Create new depth image
    Return:
        - New PCLA
        - New targetT
        - New depthImage
    '''
    # remove trailing zeros
    pclA = kitti.remove_trailing_zeros(pclA)
    # get transformed pclA based on targetP
    tMatP = kitti._get_tmat_from_params(targetP) 
    pclATransformed = kitti.transform_pcl(pclA, tMatP)
    # get new depth image of transformed pclA
    depthImageA, _ = kitti.get_depth_image_pano_pclView(pclATransformed)
    pclATransformed = kitti._zero_pad(pclATransformed, kwargs.get('pclCols')-pclATransformed.shape[1])
    # get residual Target
    #tMatResA2B = kitti.get_residual_tMat_A2B(targetT, targetP)
    targetResP2T = targetT - targetP
    return pclATransformed, targetResP2T, depthImageA

def output(batchImages, batchPclA, batchPclB, bargetT, targetP, batchTFrecFileIDs, **kwargs):
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
    ## Output for 2 images A and B as in batch
#    Parallel(n_jobs=num_cores)(delayed(output_loop)(batchImages, batchPclA, batchPclB, bargetT, targetP, batchTFrecFileIDs, i, **kwargs) for i in range(kwargs.get('activeBatchSize')))
    ## Output for 3 images A, Diff, and B as in batch
    Parallel(n_jobs=num_cores)(delayed(output_loop_diff)(batchImages, batchPclA, batchPclB, bargetT, targetP, batchTFrecFileIDs, i, **kwargs) for i in range(kwargs.get('activeBatchSize')))
    #for i in range(kwargs.get('activeBatchSize')):
    #    output_loop(batchImages, batchPclA, batchPclB, bargetT, targetP, batchTFrecFileIDs, i, **kwargs)
    return

def output_loop(batchImages, batchPclA, batchPclB, bargetT, targetP, batchTFrecFileIDs, i, **kwargs):
    """
    TODO: SIMILAR TO DATA INPUT -> WE NEED A QUEUE RUNNER TO WRITE THIS OFF TO BE FASTER

    Everything evaluated
    Warp second image based on predicted HAB and write to the new address
    Args:
    Returns:
    Raises:
      ValueError: If no dataDir
    """
    # split for depth dimension
    depthA, depthB = np.asarray(np.split(batchImages[i], 2, axis=2))
    depthB = depthB.reshape(kwargs.get('imageDepthRows'), kwargs.get('imageDepthCols'))
    pclATransformed, targetRes, depthATransformed = _apply_prediction_periodic(batchPclA[i], bargetT[i], targetP[i], **kwargs)
    # Write each Tensorflow record
    filename = str(batchTFrecFileIDs[i][0]) + "_" + str(batchTFrecFileIDs[i][1]) + "_" + str(batchTFrecFileIDs[i][2])
    tfrecord_io.tfrecord_writer(batchTFrecFileIDs[i],
                                pclATransformed, batchPclB[i],
                                depthATransformed, depthB,
                                targetRes,
                              
       kwargs.get('warpedOutputFolder')+'/', filename)
    if kwargs.get('phase') == 'train':
        folderTmat = kwargs.get('tMatTrainDir')
    else:
        folderTmat = kwargs.get('tMatTestDir')
    write_predictions(batchTFrecFileIDs[i], targetP[i], folderTmat)
    return

def output_loop_diff(batchImages, batchPclA, batchPclB, bargetT, targetP, batchTFrecFileIDs, i, **kwargs):
    """
    TODO: SIMILAR TO DATA INPUT -> WE NEED A QUEUE RUNNER TO WRITE THIS OFF TO BE FASTER

    Everything evaluated
    Warp second image based on predicted HAB and write to the new address
    Args:
    Returns:
    Raises:
      ValueError: If no dataDir
    """
    # split for depth dimension
    depthA, depthDiff, depthB = np.asarray(np.split(batchImages[i], 3, axis=2))
    depthB = depthB.reshape(kwargs.get('imageDepthRows'), kwargs.get('imageDepthCols'))
    pclATransformed, targetRes, depthATransformed = _apply_prediction(batchPclA[i], bargetT[i], targetP[i], **kwargs)
    # Write each Tensorflow record
    filename = str(batchTFrecFileIDs[i][0]) + "_" + str(batchTFrecFileIDs[i][1]) + "_" + str(batchTFrecFileIDs[i][2])
    tfrecord_io.tfrecord_writer(batchTFrecFileIDs[i],
                                pclATransformed, batchPclB[i],
                                depthATransformed, depthB,
                                targetRes,
                              
       kwargs.get('warpedOutputFolder')+'/', filename)
    if kwargs.get('phase') == 'train':
        folderTmat = kwargs.get('tMatTrainDir')
    else:
        folderTmat = kwargs.get('tMatTestDir')
    write_predictions(batchTFrecFileIDs[i], targetP[i], folderTmat)
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
