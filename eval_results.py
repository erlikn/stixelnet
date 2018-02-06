from datetime import datetime
import os.path
import time
import json
import importlib
from os import listdir                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                          
from os.path import isfile, join
print(os.getcwd())
import numpy as np
from six.moves import xrange  # pylint: disable=redefined-builtin

import Data_IO.kitti_shared as kitti

# import json_maker, update json files and read requested json file
import Model_Settings.json_maker as json_maker
json_maker.recompile_json_files()

# 170706_ITR_B_1.json Best performer
#jsonsToRead = ['170706_ITR_B_1.json',
#               '170706_ITR_B_2.json'#,
#               #'170706_ITR_B_3.json'
#               ]

# BAD zigzagy as the orientation and translation are seperate
jsonsToRead = ['170720_ITR_B_1.json',
               '170720_ITR_B_2.json'
               ]

#jsonsToRead = ['170719_ITR_B_1.json',
#               '170719_ITR_B_2.json'
#               ]

def read_model_params(jsonToRead):
    print("Reading %s" % jsonToRead)
    with open('Model_Settings/'+jsonToRead) as data_file:
        modelParams = json.load(data_file)
    _get_control_params(modelParams)
    print('Evaluation phase for %s' % modelParams['phase'])
    print('Ground truth input: %s' % modelParams['gTruthDir'])
    if modelParams['phase'] == 'train':
        print('Train sequences:', seqIDtrain)
        print('Prediction input: %s' % modelParams['tMatDir'])
    else:
        print('Test sequences:' % seqIDtest)
        print('Prediction Input: %s' % modelParams['tMatDir'])
    print(modelParams['modelName'])
    #if input("IS PRESENTED INFORMATION VALID? ") != "yes":
    #    print("Please consider updating the provided information!")
    #    return
    return modelParams
############# SET PRINT PRECISION
np.set_printoptions(precision=4, suppress=True)
############# STATE
PHASE = 'train' # 'train' or 'test'
############# PATHS
pclPath = '../Data/kitti/pointcloud/'
posePath = '../Data/kitti/poses/'
seqIDtrain = ['00', '01', '02', '03', '04', '05', '06', '07', '08']
seqIDtest = ['09', '10']
####################################################
def _get_file_names(readFolder, fileFormat):
    #print(readFolder)
    #print(fileFormat)
    filenames = [f for f in listdir(readFolder) if (isfile(join(readFolder, f)) and fileFormat in f)]
    return filenames

def _get_all_predictions(pFilenames, modelParams):
    """
    read all predictions of all sequences to a list
    """
    predAllList = list()
    predAllListTemp = list()
    for i in range(9):
        predAllListTemp.append(list())
    for i in range(0,len(pFilenames)):
        with open(modelParams['tMatDir']+'/'+pFilenames[i]) as data_file:
            tMatJson = json.load(data_file)
        predAllListTemp[int(tMatJson['seq'])].append(tMatJson)
    for i in range(9):       
        seqList = sorted(predAllListTemp[i], key=lambda k: k['idx'])
        predAllList.append(seqList)
    return predAllList

def _get_pose_from_param(pParam):
    poses = list()
    for i in range(pParam.shape[0]):
        pposep = pParam[i]
        #print(pposep)
        poses.append(kitti._get_tmat_from_params(pposep).reshape(3*4))
    return poses

def _get_param_from_pose(poselist):
    params = list()
    for i in range(len(poselist)-1):
        poseA2B = kitti._get_tMat_A_2_B(kitti._get_3x4_tmat(poselist[i]),kitti._get_3x4_tmat(poselist[i+1]))
        params.append(kitti._get_params_from_tmat(poseA2B))
    return params

def _get_prediction(predAllList, seqID):
    """
    get prediction for an specific sequence
    """
    return predAllList[int(seqID)]

def _get_tMat_A_2_B(tMatA2o, tMatB2o):
    '''
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

def _get_tMat_B_2_O(tMatA2o, tMatA2B):
    '''
    tMatA2o A -> O (target pcl will be in O), tMatA2B A -> B (source pcl is in B)
    return tMat B -> O
    '''
    # tMatA2o: A -> Orig
    # tMatA2B: A -> B ==> inv(tMatA2B): B -> A
    # tMatA2o * inv(tMatA2B) : B -> O
    tMatA2o = np.append(tMatA2o, [[0, 0, 0, 1]], axis=0)
    tMatA2B = np.append(tMatA2B, [[0, 0, 0, 1]], axis=0)
    tMatB2o = np.matmul(tMatA2o, np.linalg.inv(tMatA2B))
    tMatB2o = np.delete(tMatB2o, tMatB2o.shape[0]-1, 0)
    return tMatB2o

#def _get_gt_map_seq(gPose2o):
#    origin = np.array([[0], [0], [0]], dtype=np.float32)
#    pathMap = np.ndarray(shape=[3,0], dtype=np.float32)
#    # because inv(OrigTo1)*OrigTo1*(0,0,0)=(0,0,0) so simply append (0,0,0) 
#    pathMap = np.append(pathMap, origin, axis=1)
#    for i in range(len(gPose2o)-1):
#        poseA = kitti._get_3x4_tmat(gPose2o[i])
#        poseB = kitti._get_3x4_tmat(gPose2o[i+1])
#        pose = _get_tMat_A_2_B(poseA, poseB)
#        #oAtNextFrame = kitti.transform_pcl(pathMap, pose)
#        #oAtOrigFrame = kitti.transform_pcl(oAtNextFrame, gPose2o[i+1])
#        pathMap = np.append(pathMap, oAtOrigFrame, axis=1)
#    #### PathMap consists of all points in the origin frame coordinates
#    return pathMap

def _get_gt_map(gtPose):
    """
    get the ground truth path map
    pose are w.r.t. the origin
    """
    origin = np.array([[0], [0], [0]], dtype=np.float32)
    pathMap = np.ndarray(shape=[3,0], dtype=np.float32)
    pathMap = np.append(pathMap, origin, axis=1)
    for i in range(len(gtPose)):
        pose = kitti._get_3x4_tmat(gtPose[i])
        pointT = kitti.transform_pcl(origin, pose)
        pathMap = np.append(pathMap, pointT, axis=1)
    return pathMap

def _get_gt_map_backwards(gtPose):
    """
    iterate backwards to transform step by step backwards
    """
    origin = np.array([[0], [0], [0]], dtype=np.float32)
    pathMap = np.array([[0], [0], [0]], dtype=np.float32)
    for i in range(gtPose.shape[0]-2,-1,-1):
        poseA = kitti._get_3x4_tmat(gtPose[i])
        poseB = kitti._get_3x4_tmat(gtPose[i+1])
        poseB2A = _get_tMat_A_2_B(poseB, poseA)
        pathMap = kitti.transform_pcl(pathMap, poseB2A)
        pathMap = np.append(pathMap, origin, axis=1)
    #### PathMap consists of all points transformed to the frame 0 coordinates
    # transform them to origin access
    pathMap = kitti.transform_pcl(pathMap, gtPose[0])
    # add final origin
    pathMap = np.append(pathMap, origin, axis=1)
    return pathMap

def _get_p_map(pPose):
    """
    get the predicted truth path map
    poses are w.r.t. previous frame
    """
    origin = np.array([[0], [0], [0]], dtype=np.float32)
    pathMap = np.ndarray(shape=[3,0], dtype=np.float32)
    pathMap = np.append(pathMap, origin, axis=1)
    for i in range(len(pPose)):
        pose = kitti._get_3x4_tmat(np.array(pPose[i]['tmat']))
        origin = kitti.transform_pcl(origin, pose)
        pathMap = np.append(pathMap, origin, axis=1)
    return pathMap

def _get_p_map_w_orig(pPoseAB, gPose2o):
    """
    get the predicted truth path map
    poses are w.r.t. previous frame
    """
    ## origin = np.array([[0], [0], [0]], dtype=np.float32)
    ## pathMap = np.ndarray(shape=[3,0], dtype=np.float32)
    ## pathMap = np.append(pathMap, origin, axis=1)
    ## # Sequential transformations that takes points form frame i to i+1
    ## for i in range(len(pPoseAB)-1,-1,-1):
    ##     poseA2B = kitti._get_3x4_tmat(np.array(pPoseAB[i]))
    ##     pathMap = kitti.transform_pcl(pathMap, poseA2B)
    ##     pathMap = np.append(pathMap, origin, axis=1)
    ## #### PathMap consists of all points transformed to the last frame coordinates
    ## # transform points at last frame coordinates to origin frame
    ## #pathMap = kitti.transform_pcl(pathMap, gPose2o[len(gPose2o)-1])
    ## pathMap = kitti.transform_pcl(pathMap, gPose2o[0])

    origin = np.array([[0], [0], [0]], dtype=np.float32)
    pathMap = np.ndarray(shape=[3,0], dtype=np.float32)
    poseA2o = kitti._get_3x4_tmat(gPose2o[0])
    # because inv(OrigTo1)*OrigTo1*(0,0,0)=(0,0,0) so simply append (0,0,0)
    #gtLoc = kitti.transform_pcl(origin, gPose2o[0])
    #pathMap = np.append(pathMap, gtLoc, axis=1)
    pathMap = np.append(pathMap, origin, axis=1)
    for i in range(len(pPoseAB)):
        poseA2B = kitti._get_3x4_tmat(np.array(pPoseAB[i]))
        poseB2O = _get_tMat_B_2_O(poseA2o, poseA2B)
        oAtNextFrame = kitti.transform_pcl(origin, poseA2B)
        oAtOrigFrame = kitti.transform_pcl(origin, poseB2O)
        pathMap = np.append(pathMap, oAtOrigFrame, axis=1)
        poseA2o = poseB2O
    #### PathMap consists of all points in the origin frame coordinates
    return pathMap

def _get_p_map_w_orig_points(pPoseAB, gPose2o):
    """
    Original Coordinates are used and only transformation for each frame is plotted
    len(pPoseAB) == len(gPose2o)+1
    get the predicted truth path map
    poses are w.r.t. previous frame
    """
    origin = np.array([[0], [0], [0]], dtype=np.float32)
    pathMap = np.ndarray(shape=[3,0], dtype=np.float32)
    gtLoc = kitti.transform_pcl(origin, gPose2o[0])
    pathMap = np.append(pathMap, gtLoc, axis=1)
    # Sequential transformations that takes points form frame i to i+1
    for i in range(len(pPoseAB)):
        poseA2B = kitti._get_3x4_tmat(np.array(pPoseAB[i]))
        oAtNextFrame = kitti.transform_pcl(origin, poseA2B)
        oAtOrigFrame = kitti.transform_pcl(oAtNextFrame, gPose2o[i+1])
        pathMap = np.append(pathMap, oAtOrigFrame, axis=1)
    #### PathMap consists of all points in the origin frame coordinates
    return pathMap

def _get_control_params(modelParams):
    """
    Get control parameters for the specific task
    """
    modelParams['phase'] = PHASE
    #params['shardMeta'] = model_cnn.getShardsMetaInfo(FLAGS.dataDir, params['phase'])

    modelParams['existingParams'] = None
    modelParams['gTruthDir'] = posePath

    if modelParams['phase'] == 'train':
        modelParams['activeBatchSize'] = modelParams['trainBatchSize']
        modelParams['maxSteps'] = modelParams['trainMaxSteps']
        modelParams['numExamples'] = modelParams['numTrainDatasetExamples']
        modelParams['dataDir'] = modelParams['trainDataDir']
        modelParams['warpedOutputFolder'] = modelParams['warpedTrainDataDir']
        modelParams['tMatDir'] = modelParams['tMatTrainDir']
        modelParams['seqIDs'] = seqIDtrain

    if modelParams['phase'] == 'test':
        modelParams['activeBatchSize'] = modelParams['testBatchSize']
        modelParams['maxSteps'] = modelParams['testMaxSteps']
        modelParams['numExamples'] = modelParams['numTestDatasetExamples']
        modelParams['dataDir'] = modelParams['testDataDir']
        modelParams['warpedOutputFolder'] = modelParams['warpedTestDataDir']
        modelParams['tMatDir'] = modelParams['tMatTestDir']
        modelParams['seqIDs'] = seqIDtest
    return modelParams

def evaluate(modelParamsList, prevPParam=list()):
    # Read all prediction posefiles and sort them based on the seqID and frameID
    seqIDs = modelParamsList[0]['seqIDs']
    predPosesList = list()
    gtPosePathList = list()
    print("Reading predictions from files")
    for trnItr in range(len(modelParamsList)):
        print("          Pred Iteration ", str(trnItr+1))
        modelParams = modelParamsList[trnItr]
        pFilenames = _get_file_names(modelParams['tMatDir'], "")
        predPosesList.append(_get_all_predictions(pFilenames, modelParams))
    # For each sequence
    for i in range(len(seqIDs)):
        PParamList = list()
        pMapSeqList = list()
        pMapSeqWgtFramesList = list()
        print("Processing sequences: {0} / {1}".format(i+1, len(modelParams['seqIDs'])))
        for trnItr in range(len(modelParamsList)):
            modelParams = modelParamsList[trnItr]
            predPoses = predPosesList[trnItr]
            ###### Read groundtruth posefile for a seqID
            # create map
            # Get ground truth information (only once per sequence)
            if trnItr == 0:
                gtPosePath = kitti.get_pose_path(modelParams['gTruthDir'], modelParams['seqIDs'][i])
                gtPose = kitti.get_pose_data(gtPosePath)
                gtParam = _get_param_from_pose(gtPose)
                #print("GT pose count:", len(gtPose))
                #print("GT params count:", len(gtParam))
                gtMapOrig = _get_gt_map(gtPose) # w.r.t. Original
                #vis_path(gtMapOrig, 'GTORIG')
                #gtMapSeq = _get_gt_map_seq(gtPose) # w.r.t. sequential
                #vis_path(gtMapSeq, 'GTSEQ')
                #gtMapBack = _get_gt_map_backwards(gtPose) # w.r.t. Backwards
                #vis_path(gtMapBack, 'GTBACK')
                #vis_path_all(gtMapOrig, gtMapSeq, gtMapBack, ['GTORIG', 'GTSEQ', 'GTBACK'])
            ###### Get prediction map
            # create map
            pPoseParam = _get_prediction(predPoses, modelParams['seqIDs'][i])
            print("   Iteration: {0}".format(trnItr+1))
            print("         GT  pose  count  =", len(gtPose))
            print("         Pred param count =", len(pPoseParam))
            pParam = list()
            for j in range(len(pPoseParam)):
                pParam.append(pPoseParam[j]['tmat'])
            pParam = np.array(pParam)
            if (len(PParamList) != 0):
                pParam = pParam + PParamList[trnItr-1]
            print("         Abs error:", np.sum(np.abs(pParam-np.array(gtParam)), axis=0))
            print("         +/- error:", np.sum((pParam-np.array(gtParam)), axis=0))
            PParamList.append(pParam)
            pPose = _get_pose_from_param(pParam)
            # Use only sequential
            pMapSeqList.append(_get_p_map_w_orig(pPose, gtPose))
            # Use GTforLoc
            pMapSeqWgtFramesList.append(_get_p_map_w_orig_points(pPose, gtPose))

        # Visualize both
        print("   Displaying---")
        vis_path_all_perSeq(gtMapOrig, pMapSeqList, pMapSeqWgtFramesList, ['GT', 'PredSeq', 'PSeqWgtFrms'])
    return
################################
def vis_path_all_perSeq(gtxyz, p1xyzList, p2xyzList, legendNamesx3):
    import matplotlib.pyplot as plt
    plotList = list()
    legendNames = list()
    legendNames.append(legendNamesx3[0])
    gt, = plt.plot(gtxyz[0], gtxyz[1], 'r')
    plotList.append(gt)
    colorList1 = ['b', 'g', 'm', 'yellow']
    colorList2 = ['c', 'mediumaquamarine', 'orchid', 'goldenrod']
    for i in range(len(p1xyzList)):
        pred1, = plt.plot(p1xyzList[i][0], p1xyzList[i][1], colorList1[i])
        #pred2, = plt.plot(p2xyzList[i][0], p2xyzList[i][1], colorList2[i], alpha=0.5)
        plotList.append(pred1)
        legendNames.append(legendNamesx3[1]+"_"+str(i+1))
        #plotList.append(pred2)
        #legendNames.append(legendNamesx3[2]+"_"+str(i+1))
        
    plt.legend(plotList, legendNames)
    plt.show()

def vis_path_all(gtxyz, p1xyz, p2xyz, legendNamesx3):
    import matplotlib.pyplot as plt
    gt, = plt.plot(gtxyz[0], gtxyz[1], 'r')
    pred1, = plt.plot(p1xyz[0], p1xyz[1], 'b')
    pred2, = plt.plot(p2xyz[0], p2xyz[1], 'c', alpha=0.5)
    plt.legend([gt, pred1, pred2], legendNamesx3)
    plt.show()

def vis_path(xyz, graphType=""):
    import matplotlib.pyplot as plt
    graph = plt.plot(xyz[0], xyz[1], 'm')
    plt.legend(graph, [graphType])
    plt.show()


def main(argv=None):  # pylint: disable=unused-argumDt
    modelParamsList = list()
    for i in range(len(jsonsToRead)):
        modelParamsList.append(read_model_params(jsonsToRead[i]))
    evaluate(modelParamsList)

main()
