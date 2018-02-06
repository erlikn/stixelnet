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
import os, os.path
import time
import logging
import json
import importlib

import numpy as np
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf
#import tensorflow.python.debug as tf_debug

import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"]="1"

#from tensorflow.python.client import device_lib
#print(device_lib.list_local_devices())

# import input & output modules 
import Data_IO.data_input_ntuple as data_input
import Data_IO.data_output_ntuple as data_output

PHASE = 'train'

####################################################
####################################################
####################################################
####################################################
####################################################
FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_integer('printOutStep', 100,
                            """Number of batches to run.""")
tf.app.flags.DEFINE_integer('summaryWriteStep', 100,
                            """Number of batches to run.""")
tf.app.flags.DEFINE_integer('modelCheckpointStep', 1000,
                            """Number of batches to run.""")
tf.app.flags.DEFINE_integer('ProgressStepReportStep', 250,
                            """Number of batches to run.""")
tf.app.flags.DEFINE_integer('ProgressStepReportOutputWrite', 250,
                            """Number of batches to run.""")
####################################################
####################################################
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
####################################################
####################################################
####################################################
####################################################
####################################################
####################################################
def train(modelParams):
    # import corresponding model name as model_cnn, specifed at json file
    model_cnn = importlib.import_module('Model_Factory.'+modelParams['modelName'])
    
    if not os.path.exists(modelParams['dataDir']):
        raise ValueError("No such data directory %s" % modelParams['dataDir'])

    _setupLogging(os.path.join(modelParams['trainLogDir'], "genlog"))

    with tf.Graph().as_default():
        # track the number of train calls (basically number of batches processed)
        globalStep = tf.get_variable('globalStep',
                                     [],
                                     initializer=tf.constant_initializer(0),
                                     trainable=False)

        # Get images and transformation for model_cnn.
        images, pclA, targetT, bitTarget, rngs, tfrecFileIDs = data_input.inputs(**modelParams)
        print('Input        ready')
        # Build a Graph that computes the HAB predictions from the
        # inference model.
        targetP = model_cnn.inference(images, **modelParams)
        # Calculate loss. 2 options:

        # use mask to get degrees significant
        # What about adaptive mask to zoom into differences at each CNN stack !!!
        ########## model_cnn.loss is called in the loss function
        #loss = weighted_loss(targetP, targetT, **modelParams)
        # CLASSIFICATION
        if modelParams.get('lastTuple'):
            # for training on last tuple        
            loss = model_cnn.loss(targetP, bitTarget[:,:,:,modelParams['numTuple']-2:modelParams['numTuple']-1], **modelParams)
        else:
            # for training on all tuples
            loss = model_cnn.loss(targetP, bitTarget, **modelParams)
        print('--------targetP', targetP.get_shape())
        print('--------rngs', rngs.get_shape())
        return

        # Build a Graph that trains the model with one batch of examples and
        # updates the model parameters.
        opTrain = model_cnn.train(loss, globalStep, **modelParams)
        ##############################
        print('Training     ready')
        # Create a saver.
        saver = tf.train.Saver(tf.global_variables())
        print('Saver        ready')

        # Build the summary operation based on the TF collection of Summaries.
        summaryOp = tf.summary.merge_all()
        print('MergeSummary ready')

        # Build an initialization operation to run below.
        #init = tf.initialize_all_variables()
        init = tf.global_variables_initializer()

        #opCheck = tf.add_check_numerics_ops()
        # Start running operations on the Graph.
        config = tf.ConfigProto(log_device_placement=modelParams['logDevicePlacement'])
        config.gpu_options.allow_growth = True
        config.graph_options.optimizer_options.global_jit_level = tf.OptimizerOptions.ON_1
        sess = tf.Session(config=config)
        print('Session      ready')

        #sess = tf_debug.LocalCLIDebugWrapperSession(sess)
        #sess.add_tensor_filter("has_inf_or_nan", tf_debug.has_inf_or_nan)
        sess.run(init)


        # restore a saver.
        #saver.restore(sess, (modelParams['trainLogDir'].replace('_B_2','_B_1'))+'/model.ckpt-'+str(modelParams['trainMaxSteps']-1))
        #print('Ex-Model     loaded')

        # Start the queue runners.
        tf.train.start_queue_runners(sess=sess)
        print('QueueRunner  started')

        summaryWriter = tf.summary.FileWriter(modelParams['trainLogDir'], sess.graph)
        
        print('Training     started')
        durationSum = 0
        durationSumAll = 0
        for step in xrange(modelParams['maxSteps']):
            startTime = time.time()
            #_, lossValue = sess.run([opTrain, loss])
            _, lossValue, bitPreEV = sess.run([opTrain, loss, targetP])
            #print('lossValue', lossValue)
            #zzz = np.reshape(np.exp(bitPreEV[0]), (6,32))
            #print(zzz)
            #print(bitPreEV[0].shape)
            #for i in range(6):
            #    print(zzz[i].sum())
            duration = time.time() - startTime
            durationSum += duration
            assert not np.isnan(lossValue), 'Model diverged with loss = NaN'

            if step % FLAGS.printOutStep == 0:
                numExamplesPerStep = modelParams['activeBatchSize']
                examplesPerSec = numExamplesPerStep / duration
                secPerBatch = float(duration)
                format_str = ('%s: step %d, loss = %.2f (%.1f examples/sec; %.3f '
                              'sec/batch), loss/batch = %.2f')
                logging.info(format_str % (datetime.now(), step, lossValue,
                                           examplesPerSec, secPerBatch, lossValue/modelParams['activeBatchSize']))

            if step % FLAGS.summaryWriteStep == 0:
                summaryStr = sess.run(summaryOp)
                summaryWriter.add_summary(summaryStr, step)

            # Save the model checkpoint periodically.
            if step % FLAGS.modelCheckpointStep == 0 or (step + 1) == modelParams['maxSteps']:
                checkpointPath = os.path.join(modelParams['trainLogDir'], 'model.ckpt')
                saver.save(sess, checkpointPath, global_step=step)
            
            # Print Progress Info
            if ((step % FLAGS.ProgressStepReportStep) == 0) or ((step+1) == modelParams['maxSteps']):
                print('Progress: %.2f%%, Elapsed: %.2f mins, Training Completion in: %.2f mins --- %s' %
                        (
                            (100*step)/modelParams['maxSteps'],
                            durationSum/60,
                            (((durationSum*modelParams['maxSteps'])/(step+1))/60)-(durationSum/60),
                            datetime.now()
                        )
                    )

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
    print('Training steps = %.1f' % float(modelParams['trainMaxSteps']))
    print('Rounds on datase = %.1f' % float((modelParams['trainBatchSize']*modelParams['trainMaxSteps'])/modelParams['numTrainDatasetExamples']))
    print('lossFunction = ', modelParams['lossFunction'])
    print('Train Input: %s' % modelParams['trainDataDir'])
    #print('Test  Input: %s' % modelParams['testDataDir'])
    print('Train Logs Output: %s' % modelParams['trainLogDir'])
    #print('Test  Logs Output: %s' % modelParams['testLogDir'])
    print('Train Warp Output: %s' % modelParams['warpedTrainDataDir'])
    #print('Test  Warp Output: %s' % modelParams['warpedTestDataDir'])
    print('')
    print('')

    print('Train Main is built and Dataset is complied with n = 2 tuples!!!')
    print('')
    #if input("(Overwrite WARNING) Did you change logs directory? (y) ") != "y":
    #    print("Please consider changing logs directory in order to avoid overwrite!")
    #    return
    #if tf.gfile.Exists(modelParams['trainLogDir']):
    #    tf.gfile.DeleteRecursively(modelParams['trainLogDir'])
    #tf.gfile.MakeDirs(modelParams['trainLogDir'])
    train(modelParams)


if __name__ == '__main__':
    # looks up in the module named "__main__" (which is this module since its whats being run) in the sys.modules
    # list and invokes that modules main function (defined above)
    #    - in the process it parses the command line arguments using tensorflow.python.platform.flags.FLAGS
    #    - run can be called with the specific main and/or arguments
    tf.app.run()
