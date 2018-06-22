import json
import collections
import numpy as np
import os

def write_json_file(filename, datafile):
    filename = 'Model_Settings/'+filename
    datafile = collections.OrderedDict(sorted(datafile.items()))
    with open(filename, 'w') as outFile:
        json.dump(datafile, outFile, indent=0)

def _set_folders(folderPath):
    if not os.path.exists(folderPath):
        os.makedirs(folderPath)

####################################################################################
####################################################################################
####################################################################################

baseTrainDataDir = '../Data/train_tfrecords_clsf'
baseTestDataDir = '../Data/test_tfrecords_clsf'
trainLogDirBase = '../Data/logs/train_logs/'
testLogDirBase = '../Data/logs/test_logs/'

####################################################################################
####################################################################################
####################################################################################
reCompileJSON = True
####################################################################################
####################################################################################
####################################################################################

def write_stixelnet(runName):
    dataLocal = {
        # Data Parameters
        'numTrainDatasetExamples' : 6000,# -> 331K ground truth columns generated
        'numTestDatasetExamples' : 800,# -> 57k ground truth columns generated
        'trainDataDir' : '../Data/train_tfrecords',
        'testDataDir' : '../Data/test_tfrecords',
        'trainLogDir' : trainLogDirBase+'',
        'testLogDir' : testLogDirBase+'',
        'writeWarped' : False,
        'pretrainedModelCheckpointPath' : '',
        # Image Parameters
        'imageDepthRows' : 370,
        'imageDepthCols' : 24,
        'imageDepthChannels' : 3,
        'hMin' : 140,
        # Model Parameters
        'modelName' : 'stixelnet',
        'modelShape' : [64, 200, 1024, 2048],
        'batchNorm' : True,
        'weightNorm' : False,
        'optimizer' : 'MomentumOptimizer', # AdamOptimizer MomentumOptimizer GradientDescentOptimizer
        'momentum' : 0.9,
        'initialLearningRate' : 0.01,
        'learningRateDecayFactor' : 0.5,
        'numEpochsPerDecay' : 10000.0,
        'epsilon' : 0.1,
        'dropOutKeepRate' : 0.5,
        'clipNorm' : 1.0,
        'lossFunction' : 'L2',
        # Train Parameters
        'trainBatchSize' : 128,
        'testBatchSize' : 128,
        'outputSize' : 50, # Last layer
        'trainMaxSteps' : 1500, # 128*1500 = 192000 -> 32 runs over all training data
        'testMaxSteps' : 8, # 8*128 = 1024 -> 1 run over test data
        'usefp16' : False,
        'logDevicePlacement' : False,
        'classification' : {'Model' : False, 'binSize' : 0},
        'lastTuple' : False
        }
    dataLocal['testMaxSteps'] = int(np.ceil(dataLocal['numTestDatasetExamples']/dataLocal['testBatchSize']))

    dataLocal['writeWarpedImages'] = True
    # Iterative model only changes the wayoutput is written, 
    # so any model can be used by ease
    reCompileITR = True
    NOreCompileITR = False

    # binCenters --> 50 centers
    dataLocal['binCenters'] = [1,3,5,7,9,11,13,15,17,19]
    
    if runName == '180206':
        stx_180206(reCompileITR, trainLogDirBase, testLogDirBase, runName, dataLocal)
    #elif runName == '171003_ITR_B': # using 170706_ITR_B but with loss for all n-1 tuples
    #    dataLocal['classificationModel'] = {'Model' : True, 'binSize' : 32}
    #    itr_171003_ITR_B_clsf(reCompileITR, trainLogDirBase, testLogDirBase, runName, dataLocal)
    else:
        print("--error: Model name not found!")
        return False
    return True
    ##############
    ##############
    ##############

def stx_180206(reCompileITR, trainLogDirBase, testLogDirBase, runName, data):
    if reCompileITR:
        data['modelName'] = 'stixelnet'
        data['imageDepthChannels'] = 3
        data['optimizer'] = 'MomentumOptimizer' # AdamOptimizer MomentumOptimizer GradientDescentOptimizer
        data['modelShape'] = [64, 200, 1024, 2048]
        data['trainBatchSize'] = 128
        data['testBatchSize'] = 128
        data['numTrainDatasetExamples'] = 6000
        data['numTestDatasetExamples'] = 800
        data['logicalOutputSize'] = 50
        data['networkOutputSize'] = data['logicalOutputSize']
        data['lossFunction'] = "PL_loss"
        
        data['trainDataDir'] = baseTrainDataDir
        data['testDataDir'] = baseTestDataDir
        
        data['trainLogDir'] = trainLogDirBase + runName
        data['testLogDir'] = testLogDirBase + runName
        data['warpOriginalImage'] = True
        data['batchNorm'] = True
        data['weightNorm'] = False
        write_json_file(runName+'.json', data)

####################################################################################
####################################################################################
####################################################################################
####################################################################################

def recompile_json_files(runName):
    successItr = write_stixelnet(runName)
    if successItr:
        print("JSON files updated")
    return successItr
