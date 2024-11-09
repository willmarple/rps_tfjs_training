import * as tf from '@tensorflow/tfjs'
import {
  IMAGE_WIDTH,
  IMAGE_HEIGHT,
  NUM_CHANNELS,
  NUM_CLASSES
} from './constants'

export const getAdvancedModel = () => {
  const model = tf.sequential({
    name: 'advancedModel'
  })

  // First conv layer - reduced filters
  model.add(tf.layers.conv2d({
    inputShape: [IMAGE_WIDTH, IMAGE_HEIGHT, NUM_CHANNELS],
    kernelSize: 3,
    filters: 16,
    activation: 'relu',
    padding: 'same'
  }))
  model.add(tf.layers.maxPooling2d({ poolSize: 2 }))

  // Second conv layer - reduced filters
  model.add(tf.layers.conv2d({
    kernelSize: 3,
    filters: 32,
    activation: 'relu',
    padding: 'same'
  }))
  model.add(tf.layers.maxPooling2d({ poolSize: 2 }))

  // Flatten and dense layers - reduced units
  model.add(tf.layers.flatten())
  model.add(tf.layers.dense({ units: 32, activation: 'relu' }))
  model.add(tf.layers.dense({ units: NUM_CLASSES, activation: 'softmax' }))

  model.compile({
    optimizer: tf.train.adam(0.001),
    loss: 'categoricalCrossentropy',
    metrics: ['accuracy']
  })

  return model
}

// The classic MNIST style model
export const getSimpleModel = () => {
  const model = tf.sequential()

  // In the first layer of out convolutional neural network we have
  // to specify the input shape. Then we specify some parameters for
  // the convolution operation that takes place in this layer.
  model.add(
    tf.layers.conv2d({
      inputShape: [IMAGE_WIDTH, IMAGE_HEIGHT, NUM_CHANNELS],
      kernelSize: 5,
      filters: 8,
      strides: 1,
      activation: 'relu',
      kernelInitializer: 'varianceScaling'
    })
  )

  // The MaxPooling layer acts as a sort of downsampling using max values
  // in a region instead of averaging.
  model.add(tf.layers.maxPooling2d({ poolSize: [2, 2], strides: [2, 2] }))

  // Repeat another conv2d + maxPooling stack.
  // Note that we have more filters in the convolution.
  model.add(
    tf.layers.conv2d({
      kernelSize: 5,
      filters: 16,
      strides: 1,
      activation: 'relu',
      kernelInitializer: 'varianceScaling'
    })
  )
  model.add(tf.layers.maxPooling2d({ poolSize: [2, 2], strides: [2, 2] }))

  // Now we flatten the output from the 2D filters into a 1D vector to prepare
  // it for input into our last layer. This is common practice when feeding
  // higher dimensional data to a final classification output layer.
  model.add(tf.layers.flatten())

  // Our last layer is a dense layer which has 3 output units, one for each
  // output class (i.e. 0, 1, 2).
  const NUM_OUTPUT_CLASSES = 3
  model.add(
    tf.layers.dense({
      units: NUM_OUTPUT_CLASSES,
      kernelInitializer: 'varianceScaling',
      activation: 'softmax'
    })
  )

  // Choose an optimizer, loss function and accuracy metric,
  // then compile and return the model
  const optimizer = tf.train.adam()
  model.compile({
    optimizer: optimizer,
    loss: 'categoricalCrossentropy',
    metrics: ['accuracy']
  })

  return model
}
