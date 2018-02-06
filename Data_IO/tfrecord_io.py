# Python 3.4
import sys
sys.path.append("/usr/local/lib/python3.4/site-packages/")
import cv2 as cv2
from os import listdir
from os.path import isfile, join
from os import walk
from shutil import copy
import numpy as np
import tensorflow as tf


def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def _int64_array(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))

def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

# to get HAB and pOrig
def _float_nparray(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))

def _decode_byte_string(filename):
    """Decode and preprocess one filename.
    Args:
      filename: Binary string Tensor
    Returns:
      String Tensor containing the image in float32
    """
    tfname = tf.decode_raw(filename, tf.uint8)
    return tfname

def _decode_byte_string(filename, r, c, d):
    """Decode and preprocess one filename.
    Args:
      filename: Binary string Tensor
    Returns:
      String Tensor containing the image in float32
    """
    tfname = tf.decode_raw(filename, tf.uint8)
    tfname = tf.reshape(tfname, [r, c, d])
    tfname.set_shape([r, c, d])
    return tfname

def _decode_byte_image(image, height, width, depth):
    """Decode and preprocess one image for evaluation or training.
    Args:
      imageBuffer: Binary string Tensor
      Height, Widath, Channels <----- GLOBAL VARIABLES ARE USED DUE TO SET_SHAPE REQUIREMENTS
    Returns:
      3-D float Tensor containing the image in float32
    """
    image = tf.decode_raw(image, tf.float32)
    if depth > 1:
        image = tf.reshape(image, [height, width, depth])
        image.set_shape([height, width, depth])
    else:
        image = tf.reshape(image, [height, width])
        image.set_shape([height, width])
    return image

def _get_pcl(pcl, rows, cols):
    """
    Decode and put point cloud in the right form. nx4
    """
    #pcl = tf.decode_raw(pcl, tf.float32)
    pcl = tf.reshape(pcl, [rows, cols])
    pcl.set_shape([rows, cols])
    return pcl

def parse_example_proto(exampleSerialized, **kwargs):
    """
        ID: python list with size 2
        pclA: numpy matrix of size 3xPCL_COLS
        pclB: numpy matrix of size 3xPCL_COLS
        image: numpy matrix of 128x512x2
            imgDepthA: numpy matrix of size 128x512
            imgDepthB: numpy matrix of size 128x512
        targetABGXYZ: numpy matrix of size 3x4 = 12

        'ID': _int64_feature(IDList),
        'pclA': _float_nparray(pclAList),
        'pclB': _float_nparray(pclBList),
        'image': _bytes_feature(flatImageList)
        'targetABGXYZ': _float_nparray(targetList), # 2D np array
    """
    """
    KWARGS:
        imageDepthRows = 128
        imageDepthCols = 512
        imageDepthChannels = 2

        pclRows = 3
        pclCols = 62074

        targetABGXYZ = 6
    """
    featureMap = {
        'fileID': tf.FixedLenFeature([3], dtype=tf.int64),
        'images': tf.FixedLenFeature([], dtype=tf.string),
        'pclA': tf.FixedLenFeature([kwargs.get('pclRows')*kwargs.get('pclCols')], dtype=tf.float32),
        'pclB': tf.FixedLenFeature([kwargs.get('pclRows')*kwargs.get('pclCols')], dtype=tf.float32),
        'targetABGXYZ': tf.FixedLenFeature([6], dtype=tf.float32)
        }
    features = tf.parse_single_example(exampleSerialized, featureMap)
    fileID = features['fileID']
    images = _decode_byte_image(features['images'],
                                kwargs.get('imageDepthRows'),
                                kwargs.get('imageDepthCols'),
                                kwargs.get('imageDepthChannels'))
    pclA = _get_pcl(features['pclA'], kwargs.get('pclRows'), kwargs.get('pclCols'))
    pclB = _get_pcl(features['pclB'], kwargs.get('pclRows'), kwargs.get('pclCols'))
    target = features['targetABGXYZ']
    # PCLs will hold padded [0, 0, 0, 0] points at the end that will be ignored during usage
    # However, they will be kept to unify matrix col size for valid tensor operations 
    return images, pclA, pclB, target, fileID

def parse_example_proto_DIFF(exampleSerialized, **kwargs):
    """
        ID: python list with size 2
        pclA: numpy matrix of size 3xPCL_COLS
        pclB: numpy matrix of size 3xPCL_COLS
        image: numpy matrix of 128x512x2
            imgDepthA: numpy matrix of size 128x512
            imgDepthB: numpy matrix of size 128x512
        targetABGXYZ: numpy matrix of size 3x4 = 12

        'ID': _int64_feature(IDList),
        'pclA': _float_nparray(pclAList),
        'pclB': _float_nparray(pclBList),
        'image': _bytes_feature(flatImageList)
        'targetABGXYZ': _float_nparray(targetList), # 2D np array
    """
    """
    KWARGS:
        imageDepthRows = 128
        imageDepthCols = 512
        imageDepthChannels = 2

        pclRows = 3
        pclCols = 62074

        targetABGXYZ = 6
    """
    featureMap = {
        'fileID': tf.FixedLenFeature([3], dtype=tf.int64),
        'images': tf.FixedLenFeature([], dtype=tf.string),
        'pclA': tf.FixedLenFeature([kwargs.get('pclRows')*kwargs.get('pclCols')], dtype=tf.float32),
        'pclB': tf.FixedLenFeature([kwargs.get('pclRows')*kwargs.get('pclCols')], dtype=tf.float32),
        'targetABGXYZ': tf.FixedLenFeature([6], dtype=tf.float32)
        }
    features = tf.parse_single_example(exampleSerialized, featureMap)
    fileID = features['fileID']
    images = _decode_byte_image(features['images'],
                                kwargs.get('imageDepthRows'),
                                kwargs.get('imageDepthCols'),
                                kwargs.get('imageDepthChannels')-1)
    pclA = _get_pcl(features['pclA'], kwargs.get('pclRows'), kwargs.get('pclCols'))
    pclB = _get_pcl(features['pclB'], kwargs.get('pclRows'), kwargs.get('pclCols'))
    target = features['targetABGXYZ']

    # PCLs will hold padded [0, 0, 0, 0] points at the end that will be ignored during usage
    # However, they will be kept to unify matrix col size for valid tensor operations 
    return images, pclA, pclB, target, fileID

def tfrecord_writer(fileID,
                    pclA, pclB,
                    imgDepthA, imgDepthB,
                    targetABGXYZ,
                    tfRecFolder, tfFileName):
    """
    Converts a dataset to tfrecords
    fileID = seqID, i, i+1
    imgDepthA, imgDepthB => int8 a.k.a. char 128x512
    targetABGXYZ => will be converted to float32 with size 6
    pclA, pclB => will be converted to float16 with size 3xPCLCOLS
    """
    tfRecordPath = tfRecFolder + tfFileName + ".tfrecords"
    # Depth Images
    rows = imgDepthA.shape[0]
    cols = imgDepthA.shape[1]
    depth = 2
    stackedImage = np.stack((imgDepthA, imgDepthB), axis=2) #3D array (hieght, width, channels)
    flatImage = stackedImage.reshape(rows*cols*depth)
    flatImage = np.asarray(flatImage, np.float32)
    flatImageList = flatImage.tostring()
    # Point Clouds
    pclA = pclA.reshape(pclA.shape[0]*pclA.shape[1]) # 3 x PCL_COLS
    pclAlist = pclA.tolist()
    pclB = pclB.reshape(pclB.shape[0]*pclB.shape[1]) # 3 x PCL_COLS
    pclBlist = pclB.tolist()
    # Target Transformation
    targetABGXYZList = targetABGXYZ.tolist()

    writer = tf.python_io.TFRecordWriter(tfRecordPath)
    example = tf.train.Example(features=tf.train.Features(feature={
        'fileID': _int64_array(fileID),
        'images': _bytes_feature(flatImageList),
        'pclA': _float_nparray(pclAlist), # 2D np array
        'pclB': _float_nparray(pclBlist), # 2D np array
        'targetABGXYZ': _float_nparray(targetABGXYZList) # 2D np array
        }))
    writer.write(example.SerializeToString())
    writer.close()






########################## N-TUPLE
def _get_ntuple(data, rows, cols, ntuple):
    """
    Put data in the right form.
    """
    data = tf.reshape(data, [rows, cols, ntuple])
    data.set_shape([rows, cols, ntuple])
    return data

def _get_pcl_ntuple(pcl, rows, cols, ntuple):
    """
    Decode and put point cloud in the right form. nx4
    """
    #pcl = tf.decode_raw(pcl, tf.float32)
    pcl = tf.reshape(pcl, [rows, cols, ntuple])
    pcl.set_shape([rows, cols, ntuple])
    return pcl

def _get_target_ntuple(target, rows, ntuple):
    """
    Decode and put target in the right form. 6xn
    """
    target = tf.reshape(target, [rows, ntuple])
    target.set_shape([rows, ntuple])
    return target

def parse_example_proto_ntuple(exampleSerialized, **kwargs):
    """
    Converts a dataset to tfrecords
    fileID = seqID, i, i+1
    imgDepth => int8 a.k.a. char (numTuple x 128 x 512)
    tMatTarget => will be converted to float32 with size numTuple x 6
    pcl => will be converted to float16 with size (numTuple x 3 x PCLCOLS)
    """
    """
    KWARGS:
        imageDepthRows = 128
        imageDepthCols = 512
        imageDepthChannels = ntuple

        pclRows = 3
        pclCols = 62074

        targetABGXYZ = ntupleX6
    """
    numTuples = kwargs.get('numTuple')
    featureMap = {
        'fileID': tf.FixedLenFeature([3], dtype=tf.int64),
        'images': tf.FixedLenFeature([], dtype=tf.string),
        'pcl': tf.FixedLenFeature([kwargs.get('pclRows')*kwargs.get('pclCols')*numTuples], dtype=tf.float32),
        'targetn6': tf.FixedLenFeature([(numTuples-1) * 6], dtype=tf.float32)
        }
    features = tf.parse_single_example(exampleSerialized, featureMap)
    fileID = features['fileID']
    images = _decode_byte_image(features['images'],
                                kwargs.get('imageDepthRows'),
                                kwargs.get('imageDepthCols'),
                                kwargs.get('imageDepthChannels'))
    pcl = _get_pcl_ntuple(features['pcl'], kwargs.get('pclRows'), kwargs.get('pclCols'), numTuples)
    target = _get_target_ntuple(features['targetn6'], kwargs.get('logicalOutputSize'), numTuples-1)
    # PCLs will hold padded [0, 0, 0, 0] points at the end that will be ignored during usage
    # However, they will be kept to unify matrix col size for valid tensor operations 
    return images, pcl, target, fileID


def tfrecord_writer_ntuple(fileID, pcl, imgDepth, tMatTarget, tfRecFolder, numTuples, tfFileName):
    """
    Converts a dataset to tfrecords
    fileID = seqID, i, i+1
    imgDepth => int8 a.k.a. char (numTuple x 128 x 512)
    tMatTarget => will be converted to float32 with size numTuple x 6
    pcl => will be converted to float16 with size (numTuple x 3 x PCLCOLS)
    """
    tfRecordPath = tfRecFolder + tfFileName + ".tfrecords"
    # Depth Images
    rows = imgDepth.shape[0]
    cols = imgDepth.shape[1]
    flatImage = imgDepth.reshape(rows*cols*numTuples)
    flatImage = np.asarray(flatImage, np.float32)
    flatImageList = flatImage.tostring()
    # Point Clouds
    pcl = pcl.reshape(pcl.shape[0]*pcl.shape[1]*numTuples) # 3 x PCL_COLS
    pclList = pcl.tolist()
    # Target Transformation
    tMatTarget = tMatTarget.reshape(tMatTarget.shape[0]*(numTuples-1))
    tMatTargetList = tMatTarget.tolist()

    writer = tf.python_io.TFRecordWriter(tfRecordPath)
    example = tf.train.Example(features=tf.train.Features(feature={
        'fileID': _int64_array(fileID),
        'images': _bytets_feature(flatImageList),
        'pcl': _float_nparray(pclList), # 2D np array
        'targetn6': _float_nparray(tMatTargetList) # 2D np array
        }))
    writer.write(example.SerializeToString())
    writer.close()



############################################
############################################
############                  ##############
############  CLASSIFICATION  ##############
############                  ##############
############################################
############################################
def parse_example_proto_ntuple_classification(exampleSerialized, **kwargs):
    """
        Converts a dataset to tfrecords
        fileID = seqID, i, i+1
        imgDepth => int8 a.k.a. char  (rows_128 x cols_512 x n)
        tMatTarget => will be converted to float32 with size (6D_data x n-1)
        bitTarget => (6D_data x bins x n-1)
        rng => (6D_data x bins+1 x n-1)
        pcl => will be converted to float16 with size  (rows_XYZ x cols_Points x n)
    """
    """
        KWARGS:
            imageDepthRows = 128
            imageDepthCols = 512
            imageDepthChannels = ntuple
            pclRows = 3
            pclCols = 62074
            targetABGXYZ = ntupleX6
    """
    numTuples = kwargs.get('numTuple')
    featureMap = {
        'fileID': tf.FixedLenFeature([3], dtype=tf.int64),
        'images': tf.FixedLenFeature([], dtype=tf.string),
        'pcl': tf.FixedLenFeature([kwargs.get('pclRows')*kwargs.get('pclCols')*numTuples], dtype=tf.float32),
        'targetn6': tf.FixedLenFeature([kwargs.get('logicalOutputSize')*(numTuples-1)], dtype=tf.float32),
        'bitTarget': tf.FixedLenFeature([], dtype=tf.string),
        'rngs': tf.FixedLenFeature([kwargs.get('logicalOutputSize')*(kwargs.get('classificationModel').get('binSize')+1)*(numTuples-1)], dtype=tf.float32)
        }
    features = tf.parse_single_example(exampleSerialized, featureMap)
    # seqID, i, i+1
    fileID = features['fileID']
    # (rows_128 x cols_512 x n)
    images = _decode_byte_image(features['images'],
                                kwargs.get('imageDepthRows'),
                                kwargs.get('imageDepthCols'),
                                numTuples) # kwargs.get('imageDepthChannels'))
    # (rows_XYZ x cols_Points x n)
    pcl = _get_ntuple(features['pcl'], kwargs.get('pclRows'), kwargs.get('pclCols'), numTuples)
    # (6D_data x n-1)
    target = _get_target_ntuple(features['targetn6'], kwargs.get('logicalOutputSize'), numTuples-1)
    # (6D_data x bins x n-1)
    bitTarget = _decode_byte_string(features['bitTarget'], kwargs.get('logicalOutputSize'), kwargs.get('classificationModel').get('binSize'), (numTuples-1))
    # (6D_data x bins+1 x n-1)
    rngs = _get_ntuple(features['rngs'], kwargs.get('logicalOutputSize'), kwargs.get('classificationModel').get('binSize')+1, numTuples-1)

    # PCLs will hold padded [0, 0, 0, 0] points at the end that will be ignored during usage
    # However, they will be kept to unify matrix col size for valid tensor operations 
    return images, pcl, target, bitTarget, rngs, fileID

def tfrecord_writer_ntuple_classification(fileID, pcl, imgDepth, tMatTarget, bitTarget, rng, tfRecFolder, numTuples, tfFileName):
    """
    Converts a dataset to tfrecords
    fileID = seqID, i, i+1
    imgDepth => int8 a.k.a. char  (rows_128 x cols_512 x n)
    tMatTarget => will be converted to float32 with size (6D_data x n-1)
    bitTarget => (6D_data x bins x n-1)
    rng => (6D_data x bins+1 x n-1)
    pcl => will be converted to float16 with size  (rows_XYZ x cols_Points x n)
    """
    tfRecordPath = tfRecFolder + tfFileName + ".tfrecords"
    # Depth Images
    rows = imgDepth.shape[0]
    cols = imgDepth.shape[1]
    flatImage = imgDepth.reshape(rows*cols*numTuples)
    flatImage = np.asarray(flatImage, np.float32)
    flatImageList = flatImage.tostring()
    # Point Clouds
    pcl = pcl.reshape(pcl.shape[0]*pcl.shape[1]*numTuples) # 3 x PCL_COLS
    pclList = pcl.tolist()
    # Target Transformation
    tMatTarget = tMatTarget.reshape(tMatTarget.shape[0]*(numTuples-1))
    tMatTargetList = tMatTarget.tolist()

    bitTarget = bitTarget.reshape(bitTarget.shape[0]*bitTarget.shape[1]*bitTarget.shape[2])
    bitTargetList = bitTarget.tostring()
    rng = rng.reshape(rng.shape[0]*rng.shape[1]*rng.shape[2])
    rngList = rng.tolist()
    
    writer = tf.python_io.TFRecordWriter(tfRecordPath)
    example = tf.train.Example(features=tf.train.Features(feature={
        'fileID': _int64_array(fileID),
        'images': _bytes_feature(flatImageList),
        'pcl': _float_nparray(pclList), # 2D np array
        'targetn6': _float_nparray(tMatTargetList), # 2D np array
        'bitTarget': _bytes_feature(bitTargetList),
        'rngs': _float_nparray(rngList)
        }))
    writer.write(example.SerializeToString())
    writer.close()