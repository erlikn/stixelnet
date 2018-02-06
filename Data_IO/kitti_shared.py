# Python 3.4
import sys
sys.path.append("/usr/local/lib/python3.4/site-packages/")
import cv2 as cv2
from os import listdir
from os.path import isfile, join
from os import walk
from datetime import datetime
import time
import math
import random
from shutil import copy
import numpy as np
import csv

import struct
from scipy import spatial


import Data_IO.kitti_shared_ext_clsf_range as ext_clsf_range
############################################################################
# xyz[0]/rXYZ out of [-1,1]  this is reveresd
MIN_X_R = -1
MAX_X_R = 1
# xyz[1]/rXYZ out of [-0.12,0.4097]  this is reveresd
MIN_Y_R = -0.12
MAX_Y_R = 0.4097
# Z in range [0.01, 100]
MIN_Z = 0.01
MAX_Z = 100

IMG_ROWS = 64  # makes image of 2x64 = 128
IMG_COLS = 512
PCL_COLS = 62074 # All PCL files should have rows
PCL_ROWS = 3



############################################################################
def _get_tMat_A_2_B(tMatA2o, tMatB2o):
    '''
    3x4 , 3x4 = 3x4
    tMatA2o A -> O (source pcl is in A), tMatB2o B -> O (target pcl will be in B)
    return tMat A -> B
    '''
    # tMatA2o: A -> Orig
    # tMatB2o: B -> Orig ==> inv(tMatB2o): Orig -> B
    # inv(tMatB2o) * tMatA2o : A -> B
    tMatA2o = np.append(tMatA2o, [[0, 0, 0, 1]], axis=0)
    tMatB2o = np.append(tMatB2o, [[0, 0, 0, 1]], axis=0)
    tMatA2B = np.matmul(np.linalg.inv(tMatB2o), tMatA2o)
    tMatA2B = np.delete(tMatA2B, tMatA2B.shape[0]-1, 0)
    return tMatA2B
################# TMAT TO/FROM PARAMS
def _get_params_from_tmat(tmat):
    """
    tmat is a 3x4 matrix
    Output is a 6 valued vector

    For yaw, pitch, roll (alpha, beta, gamma)
    http://planning.cs.uiuc.edu/node103.html
    For dX, dY, dZ
    Last column
    """
    dX = tmat[0][3]
    dY = tmat[1][3]
    dZ = tmat[2][3]
    alpha_yaw = np.arctan2(tmat[1][0], tmat[0][0])
    beta_pitch = np.arctan2(-tmat[2][0], np.sqrt((tmat[2][1]*tmat[2][1])+(tmat[2][2]*tmat[2][2])))
    gamma_roll = np.arctan2(tmat[2][1], tmat[2][2])
    return np.array([alpha_yaw, beta_pitch, gamma_roll, dX, dY, dZ], dtype=np.float32)
def _get_tmat_from_params(abgxyz):
    """
    abgxyz is a 6 valued vector: Alpha_yaw, Beta_pitch, Gamma_roll, DeltaX, DeltaY, DeltaZ
    Output is a 3x4 tmat
    For rotation matrix:
    http://planning.cs.uiuc.edu/node102.html
    For Translation side:
    dX, dY, dZ are last column
    """
    a = abgxyz[0]
    b = abgxyz[1]
    g = abgxyz[2]
    dx = abgxyz[3]
    dy = abgxyz[4]
    dz = abgxyz[5]
    tmat = np.array([
              [np.cos(a)*np.cos(b), (np.cos(a)*np.sin(b)*np.sin(g))-(np.sin(a)*np.cos(g)), (np.cos(a)*np.sin(b)*np.cos(g))+(np.sin(a)*np.sin(g)), dx],
              [np.sin(a)*np.cos(b), (np.sin(a)*np.sin(b)*np.sin(g))+(np.cos(a)*np.cos(g)), (np.sin(a)*np.sin(b)*np.cos(g))-(np.cos(a)*np.sin(g)), dy],
              [-np.sin(b),          np.cos(b)*np.sin(g),                                   np.cos(b)*np.cos(g),                                   dz]
           ], dtype=np.float32)
    return tmat

############################################################################
def get_pose_path(poseFolder, seqID):
    return poseFolder + seqID + ".txt"
def get_pose_data(posePath):
    return np.loadtxt(open(posePath, "r"), delimiter=" ")
############################################################################
def _get_3x4_tmat(poseRow):
    return poseRow.reshape([3,4])
def _add_row4_tmat(pose3x4):
    return np.append(pose3x4, [[0, 0, 0, 1]], axis=0)
def _remove_row4_tmat(pose4x4):
    return np.delete(pose4x4, pose4x4.shape[0]-1, 0)
def get_residual_tMat_A2B(tMatA, tMatB):
    '''
        Input: 3x4, 3x4
        To get residual transformation E:
        T = P x E => (P.inv) x T = (P.inv) x P x E => (P.inv) x T = I x E => (P.inv) x T = E

        return E as residual tMat 3x4
    '''
    # get tMat in the correct form
    tMatA = _add_row4_tmat(_get_3x4_tmat(tMatA))
    tMatB = _add_row4_tmat(_get_3x4_tmat(tMatB))
    tMatResA2B = np.matmul(np.linalg.inv(tMatB), tMatA)
    tMatResA2B = _remove_row4_tmat(tMatResA2B)
    return tMatResA2B

def get_residual_tMat_Bp2B2A(tMatB2A, tMatB2Bp):
    '''
        Input: 3x4, 3x4
        return E as residual tMat 3x4
    '''
    # get tMat in the correct form
    tMatB2A = _add_row4_tmat(_get_3x4_tmat(tMatB2A))
    tMatB2Bp = _add_row4_tmat(_get_3x4_tmat(tMatB2Bp))
    tMatResBp2A = np.matmul(tMatB2A, np.linalg.inv(tMatB2Bp))
    tMatResBp2A = _remove_row4_tmat(tMatResBp2A)
    return tMatResBp2A
############################################################################
def transform_pcl(xyz, tMat):
    '''
    NEW XYZ = tMAT x XYZ
    pointcloud i 3xN, and tMat 3x4
    '''
    tMat = _add_row4_tmat(_get_3x4_tmat(tMat))
    # append a ones row to xyz
    xyz = np.append(xyz, np.ones(shape=[1, xyz.shape[1]]), axis=0)
    xyz = np.matmul(tMat, xyz)
    # remove last row
    xyz = np.delete(xyz, xyz.shape[0]-1, 0)
    return xyz
############################################################################
def _zero_pad(xyz, num):
    '''
    Append xyz with num 0s to have unified pcl length of PCL_COLS
    '''
    if num < 0:
        print("xyz shape is", xyz.shape)
        print("MAX PCL_COLS is", PCL_COLS)
        raise ValueError('Error... PCL_COLS should be the unified max of the whole system')
    elif num > 0:
        pad = np.zeros([xyz.shape[0], num], dtype=float)
        xyz = np.append(xyz, pad, axis=1)
    # if num is 0 -> do nothing
    return xyz

def _normalize_Z_weighted(z):
    '''
    As we have higher accuracy measuring closer points
    map closer points with higher resolution
    0---20---40---60---80---100
     40%  25%  20%  --15%---
    '''
    for i in range(0, z.shape[0]):
        if z[i] < 20:
            z[i] = (0.4*z[i])/20
        elif z[i] < 40:
            z[i] = ((0.25*(z[i]-20))/20)+0.4
        elif z[i] < 60:
            z[i] = (0.2*(z[i]-40))+0.65
        else:
            z[i] = (0.15*(z[i]-60))+0.85
    return z

def _add_corner_points(xyz, rXYZ):
    '''
    MOST RECENT CODE A10333
    Add MAX RANGE for xyz[0]/rXYZ out of [-1,1]
    Add MIN RANGE for xyz[1]/rXYZ out of [-0.12,0.4097]
    '''
    ### Add Two corner points with z=0 and x=rand and y calculated based on a, For max min locations
    ### Add Two min-max depth point to correctly normalize distance values
    ### Will be removed after histograms

    xyz = np.append(xyz, [[MIN_Y_R], [MIN_Y_R], [0]], axis=1) # z not needed
    rXYZ = np.append(rXYZ, 1)
    xyz = np.append(xyz, [[MAX_X_R], [MAX_Y_R], [0]], axis=1) # z not needed
    rXYZ = np.append(rXYZ, 1)
    #z = 0.0
    #x = 2.0
    #a = 0.43
    #y = np.sqrt(((a*a)*((x*x)*(x*x)))/(1-(a*a)))
    #xyz = np.append(xyz, [[x], [y], [z]], axis=1)
    #rXYZ = np.append(rXYZ, np.sqrt((x*x)+(y*y)+(z*z)))
    #x = -2.0
    #a = -0.1645
    #y = np.sqrt(((a*a)*((x*x)*(x*x)))/(1-(a*a)))
    #xyz = np.append(xyz, [[x], [y], [z]], axis=1)
    #rXYZ = np.append(rXYZ, np.sqrt((x*x)+(y*y)+(z*z)))

    xyz = np.append(xyz, [[0], [0], [MIN_Z]], axis=1)
    rXYZ = np.append(rXYZ, MIN_Z*MIN_Z)
    xyz = np.append(xyz, [[0], [0], [MAX_Z]], axis=1)
    rXYZ = np.append(rXYZ, MAX_Z*MAX_Z)
    return xyz, rXYZ

def _remove_corner_points(xyz):
    xyz = np.delete(xyz, xyz.shape[1]-1, 1)
    xyz = np.delete(xyz, xyz.shape[1]-1, 1)
    xyz = np.delete(xyz, xyz.shape[1]-1, 1)
    xyz = np.delete(xyz, xyz.shape[1]-1, 1)
    return xyz

def _get_plane_view(xyz, rXYZ):
    ### Flatten to a plane
    # 0 left-right, 1 is up-down, 2 is forward-back
    xT = (xyz[0]/rXYZ).reshape([1, xyz.shape[1]])
    yT = (xyz[1]/rXYZ).reshape([1, xyz.shape[1]])
    zT = rXYZ.reshape([1, xyz.shape[1]])
    planeView = np.append(np.append(xT, yT, axis=0), zT, axis=0)
    return planeView

def _make_image(depthview, rXYZ):
    '''
    Get depthview and generate a depthImage
    '''
    '''
    We found that the plane slop is in between [ -0.1645 , 0.43 ] # [top, down]
    So any point beyond this should be trimmed.
    And all points while converting to depthmap should be grouped in this range for Y
    Regarding X, we set all points with z > 0. This means slops for X are inf

    We add 2 points to the list holding 2 corners of the image plane
    normalize points to chunks and then remove the auxiliary points

    [-9.42337227   14.5816927   30.03821182  $ 0.42028627  $  34.69466782]
    [-1.5519526    -0.26304439  0.28228107   $ -0.16448526 $  1.59919727]
    '''
    ### Flatten to a plane
    depthview = _get_plane_view(depthview, rXYZ)
    ##### Project to image coordinates using histograms
    ### Add maximas and minimas. Remove after histograms ----
    depthview, rXYZ = _add_corner_points(depthview, rXYZ)
    # Normalize to 0~1
    depthview[0] = (depthview[0] - np.min(depthview[0]))/(np.max(depthview[0]) - np.min(depthview[0]))
    depthview[1] = (depthview[1] - np.min(depthview[1]))/(np.max(depthview[1]) - np.min(depthview[1]))
    # there roughly should be 64 height bins group them in 64 clusters
    _, xBinEdges = np.histogram(depthview[0], 512)
    _, yBinEdges = np.histogram(depthview[1], 64)
    xCent = np.ndarray(shape=xBinEdges.shape[0]-1)
    for i in range(0, xCent.shape[0]):
        xCent[i] = (xBinEdges[i]+xBinEdges[i+1])/2
    yCent = np.ndarray(shape=yBinEdges.shape[0]-1)
    for i in range(0, yCent.shape[0]):
        yCent[i] = (yBinEdges[i]+yBinEdges[i+1])/2
    # make image of size 128x512 : 64 -> 128 (double sampling the height)
    depthImage = np.zeros(shape=[128, 512])
    # normalize range values
    #depthview[2] = (depthview[2]-np.min(depthview[2]))/(np.max(depthview[2])-np.min(depthview[2]))
    depthview[2] = _normalize_Z_weighted(depthview[2])
    depthview[2] = 1-depthview[2]
    ### Remove maximas and minimas. -------------------------
    depthview = _remove_corner_points(depthview)
    # sorts ascending
    idxs = np.argsort(depthview[2], kind='mergesort')
    # assign range to pixels
    for i in range(depthview.shape[1]-1, -1, -1): # traverse descending
        yidx = np.argmin(np.abs(yCent-depthview[1, idxs[i]]))
        xidx = np.argmin(np.abs(xCent-depthview[0, idxs[i]]))
        # hieght is 2x64
        yidx = yidx*2
        depthImage[yidx, xidx] = depthview[2, idxs[i]]
        depthImage[yidx+1, xidx] = depthview[2, idxs[i]]
    return depthImage

def get_depth_image_pano_pclView(xyz, height=1.6):
    '''
    Gets a point cloud
    Keeps points higher than 'height' value and located on the positive Z=0 plane
    Returns correstempMaxponding depthMap and pclView
    '''
    '''
    MOST RECENT CODE A10333
    remove any point who has xyz[0]/rXYZ out of [-1,1]
    remove any point who has xyz[1]/rXYZ out of [-0.12,0.4097]
    '''
    # calc rXYZ
    rXYZ = np.linalg.norm(xyz, axis=0)
    xyz = xyz.transpose()
    first = True
    pclview = np.ndarray(shape=[xyz.shape[1],0], dtype=np.float32)
    for i in range(xyz.shape[0]):
        # xyz[i][2] >= 0 means all the points who have depth larger than 0 (positive depth plane)
        if (xyz[i][2] >= 0) and (xyz[i][1] < height) and (rXYZ[i] > 0) and (xyz[i][0]/rXYZ[i] > -1) and (xyz[i][0]/rXYZ[i] < 1) and (xyz[i][1]/rXYZ[i] > -0.12) and (xyz[i][1]/rXYZ[i] < 0.4097): # frontal view & above ground & x in range & y in range
            if first:
                pclview = xyz[i].reshape(xyz.shape[1], 1)
                first = False
            else:
                pclview = np.append(pclview, xyz[i].reshape(xyz.shape[1], 1), axis=1)
    rPclview = np.linalg.norm(pclview, axis=0)
    depthImage = _make_image(pclview, rPclview)
    pclview = _zero_pad(pclview, PCL_COLS-pclview.shape[1])
    return depthImage, pclview

############################################################################
def remove_trailing_zeros(xyz):
    '''Remove trailing 0 points'''
    condition = (xyz[0] != 0) | (xyz[1] != 0) | (xyz[2] != 0)
    condition = [[condition], [condition], [condition]]
    xyz = np.extract(condition, xyz)
    xyz = xyz.reshape([3, -1])
    return xyz

############################################################################
# BIN_multi_bit_target
def get_multi_bit_target(pose, BIN_rng, BIN_SIZE):
    '''
    pose: target n=6 dof pose
    BIN_rng: n=6 * BIN_SIZE=32 matrix showing 32 bins for each dimension of pose
    BIN_SIZE: 32, number of the bins for each dimension of pose
    '''
    bits = np.zeros(shape=[len(pose), BIN_SIZE], dtype=np.int8)
    for labdex in range(len(pose)):
        for bindex in range(BIN_SIZE+1):
            if (pose[labdex] < BIN_rng[labdex][bindex]):
                bits[labdex][bindex-1] = 1
                break
    return bits

# BIN_ranges
def get_multi_bit_ranges(BIN_max, BIN_min, BIN_SIZE):
    BIN_rng = list()
    for i in range(len(BIN_min)):
        BIN_rng.append(np.append(np.arange(BIN_min[i], BIN_max[i], (BIN_max[i]-BIN_min[i])/BIN_SIZE), [BIN_max[i]], axis=0))
    BIN_rng = np.asarray(BIN_rng, np.float32)
    return BIN_rng

def get_bin_min_max():
    # return the updated min max ranges
    BIN_min = [-0.021, -0.075, -0.027, -0.24, -0.20, -2.74]
    BIN_max = [ 0.019,  0.084,  0.023,  0.30,  0.20, 0.018]
    return BIN_max, BIN_min

def get_updated_ranges(logits, ranges):
    '''
    Get updated ranges for each tuple and each parameter based on calculated scores (logits) and ranges.
    Args:
        binPreds: predicted logits selecting bins [6, 32, nT]
        ranges: ranges for current model to exhibit bins [6, 33, nt]
    Output:
        params: [6, nT]
    '''
    binSize = logits.shape[1]
    # Get the updates for each tuple and each parameter
    for nt in range(logits.shape[2]):
        for pid in range(logits.shape[0]):
            # Update ranges
            ranges[pid,:,nt] = ext_clsf_range.get_new_ranges(ranges[pid,:,nt], logits[pid, :, nt], binSize)
    return ranges

def get_params_from_binarylogits(binPreds, ranges):
    '''
    Extract parameters from binary predicted logits and ranges.
    Args:
        binPreds: binary predicted logits selecting bins [6, 32, nT]
        ranges: ranges for current model to exhibit bins [6, 33, nt]
    Output:
        params: [6, nT]
    '''
    # find argmax for the bTargeP and use it to get the corresponding params
    # argmax over 2nd dim (bins)
    argmax = np.argmax(binPreds, axis=1) 
    params = np.ndarray([binPreds.shape[0], 1, binPreds.shape[2]])
    for parID in range(binPreds.shape[0]):
        for tupID in range(binPreds.shape[2]):
            # rngs 0-33, argmax 0-32 => always choses lower one, is this matching with kitti shared extended?????
            params[parID, 0, tupID] = ranges[parID, argmax[parID, tupID], tupID]
    return params