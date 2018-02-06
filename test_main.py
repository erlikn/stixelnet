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
from __future__ import absolute_import
from __future__ import division

from datetime import datetime
import os.path
import time
import logging
import json
import csv
import importlib

import numpy as np
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf
#import tensorflow.python.debug as tf_debug


import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"]="0"

#from tensorflow.python.client import device_lib
#print(device_lib.list_local_devices())

# import input & output modules 
import Data_IO.data_input as data_input
import Data_IO.data_output as data_output


PHASE = 'test'

####################################################
FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_integer('printOutStep', 10,
                            """Number of batches to run.""")
tf.app.flags.DEFINE_integer('summaryWriteStep', 100,
                            """Number of batches to run.""")
tf.app.flags.DEFINE_integer('modelCheckpointStep', 1000,
                            """Number of batches to run.""")
tf.app.flags.DEFINE_integer('ProgressStepReportStep', 50,
                            """Number of batches to run.""")
####################################################
def _write_to_csv(filePath, dataOutRowList):
    with open(filePath, 'a') as fileToWrite:
        writer = csv.writer(fileToWrite, delimiter=',', lineterminator='\n',)
        writer.writerow(dataOutRowList)

def _set_control_params(modelParams):
    modelParams['phase'] = PHASE
    #params['shardMeta'] = model_cnn.getShardsMetaInfo(FLAGS.dataDir, params['phase'])

    modelParams['existingParams'] = None

    if modelParams['phase'] == 'train':
        modelParams['activeBatchSize'] = modelParams['trainBatchSize']
        modelParams['maxSteps'] = modelParams['trainMaxSteps']
        modelParams['numExamples'] = modelParams['numTrainDatasetExamples']
        modelParams['dataDir'] = modelParams['trainDataDir']
        modelParams['warpedOutputFolder'] = modelParams['warpedTrainDataDir']

    if modelParams['phase'] == 'test':
        modelParams['activeBatchSize'] = modelParams['testBatchSize']
        modelParams['maxSteps'] = modelParams['testMaxSteps']
        modelParams['numExamples'] = modelParams['numTestDatasetExamples']
        modelParams['dataDir'] = modelParams['testDataDir']
        modelParams['warpedOutputFolder'] = modelParams['warpedTestDataDir']
    return modelParams

def test(modelParams):
    # import corresponding model name as model_cnn, specifed at json file
    model_cnn = importlib.import_module('Model_Factory.'+modelParams['modelName'])

    if not os.path.exists(modelParams['dataDir']):
        raise ValueError("No such data directory %s" % modelParams['dataDir'])

    _setupLogging(os.path.join(modelParams['testLogDir'], "genlog"))

    with tf.Graph().as_default():
        # Get images and transformation for model_cnn.
        images, pclA, pclB, tMatT, tfrecFileIDs = data_input.inputs(**modelParams)
        # Build a Graph that computes the HAB predictions from the
        # inference model.
        tMatP = model_cnn.inference(images, **modelParams)

        # Calculate loss. 2 options:

        # use mask to get degrees significant
        loss = model_cnn.weighted_loss(tMatP, tMatT, **modelParams)

        # pcl based
        #loss = model_cnn.pcl_loss(pclA, tMatP, tMatT, **modelParams)

        # Create a saver.
        saver = tf.train.Saver(tf.global_variables())

        # Build the summary operation based on the TF collection of Summaries.
        summaryOp = tf.summary.merge_all()

        # Build an initialization operation to run below.
        #init = tf.initialize_all_variables()
        init = tf.global_variables_initializer()

        # Start running operations on the Graph.
        config = tf.ConfigProto(log_device_placement=modelParams['logDevicePlacement'])
        config.graph_options.optimizer_options.global_jit_level = tf.OptimizerOptions.ON_1
        sess = tf.Session(config=config)

        #sess = tf_debug.LocalCLIDebugWrapperSession(sess)
        #sess.add_tensor_filter("has_inf_or_nan", tf_debug.has_inf_or_nan)
        sess.run(init)

        # restore a saver.
        saver = tf.train.Saver(tf.global_variables())
        saver.restore(sess, modelParams['trainLogDir']+'/model.ckpt-'+str(modelParams['trainMaxSteps']-1))

        # Start the queue runners.
        tf.train.start_queue_runners(sess=sess)

        summaryWriter = tf.summary.FileWriter(modelParams['testLogDir'], sess.graph)

        lossValueSum = 0
        durationSum = 0
        durationSumAll = 0
        print('Warping images with batch size %d in %d steps' % (modelParams['activeBatchSize'], modelParams['maxSteps']))

        testValueSampleResults = list()
        stepFinal = 0
        for step in xrange(modelParams['maxSteps']):
            startTime = time.time()
            evImages, evPclA, evPclB, evtMatT, evtMatP, evtfrecFileIDs, evlossValue = sess.run([images, pclA, pclB, tMatT, tMatP, tfrecFileIDs, loss])
            duration = time.time() - startTime
            durationSum += duration
            lossValueSum += evlossValue

            #_write_to_csv(modelParams['testLogDir']+'/testRes'+jsonToRead.replace('.json', '_T.csv'), evtMatT)
            #_write_to_csv(modelParams['testLogDir']+'/testRes'+jsonToRead.replace('.json', '_P.csv'), evtMatP)

            # Write test outputs tfrecords
            #### put imageA, warpped imageB by pHAB, HAB-pHAB as new HAB, changed fileaddress tfrecFileIDs
            #if (step == 0):
            #    data_output.output_with_test_image_files(evImagesOrig, evImages, evPOrig, evtHAB, evpHAB, evtfrecFileIDs, **modelParams)
            #else:
            data_output.output(evImages, evPclA, evPclB, evtMatT, evtMatP, evtfrecFileIDs, **modelParams)
            duration = time.time() - startTime
            durationSumAll += duration

            # print out control outputs 
            if step % FLAGS.printOutStep == 0:
                numExamplesPerStep = modelParams['activeBatchSize']
                examplesPerSec = numExamplesPerStep / duration
                secPerBatch = float(duration)
                format_str = ('%s: step %d, loss = %.2f (%.1f examples/sec; %.3f '
                              'sec/batch) avg_err_over_time = %.2f')
                logging.info(format_str % (datetime.now(), step, evlossValue,
                                           examplesPerSec, secPerBatch,
                                           lossValueSum/(step+1)))
            # write summaries
            if (step % FLAGS.summaryWriteStep == 0) or ((step+1) == modelParams['maxSteps']):
                summaryStr = sess.run(summaryOp)
                summaryWriter.add_summary(summaryStr, step)

            # Print Progress Info
            if ((step % FLAGS.ProgressStepReportStep) == 0) or ((step+1) == modelParams['maxSteps']):
                print('Progress: %.2f%%, Loss: %.2f, Elapsed: %.2f mins, Training Completion in: %.2f mins' %
                        ((100*step)/modelParams['maxSteps'], lossValueSum/(step+1), durationSum/60,
                         (((durationSum*modelParams['maxSteps'])/(step+1))/60)-(durationSum/60)))
               # print('Total Elapsed: %.2f mins, Training Completion in: %.2f mins' % 
               #             durationSumAll/60, (((durationSumAll*stepsForOneDataRound)/(step+1))/60)-(durationSumAll/60))
            stepFinal = step

        step = stepFinal+1
        print('Average test error = %.2f - Average time per sample= %.2f s, Steps = %d, ex/sec = %.2f' %
                        (lossValueSum/(step), duration/(step*modelParams['activeBatchSize']), step, modelParams['numExamples']/durationSum))


def _setupLogging(logPath):
    # cleanup
    if os.path.isfile(logPath):
        os.remove(logPath)

    logging.basicConfig(level=logging.DEBUG,
                        format='%(asctime)s %(name)-12s %(levelname)-8s %(message)s',
                        datefmt='%m-%d %H:%M',
                        filename=logPath,
                        filemode='w')

    # also write out to the console
    console = logging.StreamHandler()
    console.setLevel(logging.DEBUG)
    console.setFormatter(logging.Formatter('%(asctime)s %(name)-12s %(levelname)-8s %(message)s'))

    # add the handler to the root logger
    logging.getLogger().addHandler(console)

    logging.info("Logging setup complete to %s" % logPath)

def main(argv=None):  # pylint: disable=unused-argumDt
    if (len(argv)<3):
        print("Enter 'model name' and 'iteration number'")
        return
    modelName = argv[1]
    itrNum = int(argv[2])
    if itrNum>4 or itrNum<0:
        print('iteration number should only be from 1 to 4 inclusive')
        return
    # import json_maker, update json files and read requested json file
    import Model_Settings.json_maker as json_maker
    if not json_maker.recompile_json_files(modelName, itrNum):
        return
    jsonToRead = modelName+'_'+str(itrNum)+'.json'
    print("Reading %s" % jsonToRead)
    with open('Model_Settings/'+jsonToRead) as data_file:
        modelParams = json.load(data_file)

    modelParams = _set_control_params(modelParams)
    
    print(modelParams['modelName'])
    print('Rounds on datase = %.1f' % float((modelParams['testBatchSize']*modelParams['testMaxSteps'])/modelParams['numTestDatasetExamples']))
    #print('Train Input: %s' % modelParams['trainDataDir'])
    print('Test  Input: %s' % modelParams['testDataDir'])
    print('Train Logs Input: %s' % modelParams['trainLogDir'])
    print('Test  Logs Output: %s' % modelParams['testLogDir'])
    #print('Train Warp Output: %s' % modelParams['warpedTrainDataDir'])
    print('Test  Warp Output: %s' % modelParams['warpedTestDataDir'])
    if input("(Overwrite WARNING) Did you change logs directory? (y) ") != "y":
        print("Please consider changing logs directory in order to avoid overwrite!")
        return
    if tf.gfile.Exists(modelParams['testLogDir']):
        tf.gfile.DeleteRecursively(modelParams['testLogDir'])
    tf.gfile.MakeDirs(modelParams['testLogDir'])
    test(modelParams)


if __name__ == '__main__':
    # looks up in the module named "__main__" (which is this module since its whats being run) in the sys.modules
    # list and invokes that modules main function (defined above)
    #    - in the process it parses the command line arguments using tensorflow.python.platform.flags.FLAGS
    #    - run can be called with the specific main and/or arguments
    tf.app.run()
