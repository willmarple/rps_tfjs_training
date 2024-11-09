import * as tf from '@tensorflow/tfjs'
import * as tfvis from '@tensorflow/tfjs-vis'
import {
  IMAGE_WIDTH,
  IMAGE_HEIGHT,
  NUM_CHANNELS,
  BATCH_SIZE
} from './constants'

export const train = async (model, data, numEpochs = 12) => {
  const metrics = ['loss', 'val_loss', 'accuracy', 'val_accuracy']
  const container = {
    name: 'Model Training',
    tab: 'Training',
    styles: {
      height: '1000px'
    }
  }
  
  const fitCallbacks = tfvis.show.fitCallbacks(container, metrics)

  // Get training batches with smaller batch size for advanced model
  const TRAIN_BATCH_SIZE = model.name === 'advancedModel' ? 32 : BATCH_SIZE
  const trainBatch = data.nextTrainBatch(TRAIN_BATCH_SIZE)
  const validationBatch = data.nextTestBatch(TRAIN_BATCH_SIZE)

  try {
    // Explicitly clean up any existing tensors
    tf.engine().startScope()
    
    // Reshape the input tensors
    const reshapedTrainXs = trainBatch.xs.reshape([
      TRAIN_BATCH_SIZE,
      IMAGE_WIDTH,
      IMAGE_HEIGHT,
      NUM_CHANNELS
    ])
    const reshapedValidationXs = validationBatch.xs.reshape([
      TRAIN_BATCH_SIZE,
      IMAGE_WIDTH,
      IMAGE_HEIGHT,
      NUM_CHANNELS
    ])

    // Start training
    const history = await model.fit(reshapedTrainXs, trainBatch.labels, {
      batchSize: TRAIN_BATCH_SIZE,
      validationData: [reshapedValidationXs, validationBatch.labels],
      epochs: numEpochs,
      callbacks: fitCallbacks
    })

    return history
  } finally {
    // Clean up tensors
    tf.engine().endScope()
    trainBatch.xs.dispose()
    trainBatch.labels.dispose()
    validationBatch.xs.dispose()
    validationBatch.labels.dispose()
  }
}
