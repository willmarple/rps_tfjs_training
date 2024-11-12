export const modelCode = `
// Create model
export const getModel = () => {
  const model = tf.sequential()

  // In the first layer of out convolutional neural network we have
  // to specify the input shape. Then we specify some parameters for
  // the convolution operation that takes place in this layer.
  model.add(
    tf.layers.conv2d({
      inputShape: [IMAGE_WIDTH, IMAGE_HEIGHT, 3],
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
`

export const loadDataAndShowExamples = `
// Load data and show examples
const handleLoadData = useCallback(async () => {
    setLoadDataMessage('Loading 10MB Data')
    // Wrap data loading in tf.tidy
    const data = await tf.tidy(() => {
      const dataset = new RPSDataset()
      return dataset
    })
    dataRef.current = data
    await data.load()
    tfvis.visor().open()
    await showExamples(data)
    setLoadDataMessage('Data Loaded!')
  }, [])

// Show examples
export const showExamples = async data => {
  // Create a container in the visor
  const surface = tfvis
    .visor()
    .surface({ name: 'RPS Data Examples', tab: 'Input Data' })

  // Add grid styling to the surface's drawArea
  const drawArea = surface.drawArea
  drawArea.style.display = 'grid'
  drawArea.style.gridTemplateColumns = 'repeat(5, 1fr)'
  drawArea.style.gap = '4px'
  drawArea.style.padding = '8px'

  // Get the examples
  const examples = data.nextTestBatch(42)
  const numExamples = examples.xs.shape[0]

  // Create canvas elements and display images
  for (let i = 0; i < numExamples; i++) {
    const imageTensor = tf.tidy(() => {
      // Get the tensor for this specific image and reshape correctly
      return examples.xs
        .slice([i, 0], [1, examples.xs.shape[1]])
        .reshape([IMAGE_WIDTH, IMAGE_HEIGHT, NUM_CHANNELS])
    })

    const canvas = document.createElement('canvas')
    canvas.width = IMAGE_WIDTH
    canvas.height = IMAGE_HEIGHT
    canvas.style = 'width: 100%; aspect-ratio: 1;'
    
    // Use toPixels instead of draw
    await tf.browser.toPixels(imageTensor, canvas)
    
    surface.drawArea.appendChild(canvas)
    imageTensor.dispose()
  }

  examples.xs.dispose()
  examples.labels.dispose()
}
`

export const trainModel = `
// Train model
export const train = async (model, data, numEpochs) => {
  const metrics = ['loss', 'val_loss', 'acc', 'val_acc'];
  const container = { name: 'Model Training', tab: 'Training' };
  const fitCallbacks = tfvis.show.fitCallbacks(container, metrics);

  const BATCH_SIZE = 32;
  const TARGET_ACCURACY = 0.95;

  const trainData = {
    xs: tf.tensor4d(data.trainImages, [data.trainIndices.length, 64, 64, 3]),
    labels: tf.tensor2d(data.trainLabels, [data.trainIndices.length, 3])
  };

  const testData = {
    xs: tf.tensor4d(data.testImages, [data.testIndices.length, 64, 64, 3]),
    labels: tf.tensor2d(data.testLabels, [data.testIndices.length, 3])
  };

  for (let epoch = 0; epoch < numEpochs; epoch++) {
    const result = await model.evaluate(trainData.xs, trainData.labels);
    const accuracy = result[1].dataSync()[0];
    
    console.log(\`Epoch \${epoch + 1}: accuracy = \${(accuracy * 100).toFixed(2)}%\`);

    if (accuracy >= TARGET_ACCURACY) {
      console.log(\`\\nReached target accuracy of \${(TARGET_ACCURACY * 100).toFixed(2)}%. Stopping training.\`);
      break;
    }

    await model.fit(trainData.xs, trainData.labels, {
      batchSize: BATCH_SIZE,
      epochs: 1,
      validationData: [testData.xs, testData.labels],
      callbacks: [fitCallbacks]
    });
  }

  // Clean up tensors
  trainData.xs.dispose();
  trainData.labels.dispose();
  testData.xs.dispose();
  testData.labels.dispose();
};
`

export const checkUntrainedModel = `
// Check untrained model
const handleCheckUntrainedModel = useCallback(async () => {
  if (!modelRef.current || !dataRef.current) {
      alert('Please create a model and load data first')
      return
    }

    try {
      const visor = tfvis.visor()
      visor.open()
      setIsVisorOpen(true)

      // Wrap test data creation in tf.tidy
      const testData = tf.tidy(() => ({
        nextTestBatch: (batchSize) => {
          const batch = dataRef.current.nextTestBatch(batchSize)
          return {
            xs: batch.xs.reshape([batchSize, IMAGE_WIDTH, IMAGE_HEIGHT, NUM_CHANNELS]),
            labels: batch.labels
          }
        }
      }))

      await showAccuracy(modelRef.current, testData)
      await showConfusion(modelRef.current, testData)

    } catch (error) {
      console.error('Error checking untrained model:', error)
      alert('Error checking untrained model: ' + error.message)
    }
  }, [])

  // TFJS VIS
  export const showAccuracy = async (model, data, title = 'Accuracy') => {
  const [preds, labels] = doPrediction(model, data)
  const classAccuracy = await tfvis.metrics.perClassAccuracy(labels, preds)
  const container = { name: title, tab: 'Evaluation' }
  tfvis.show.perClassAccuracy(container, classAccuracy, classNames)
  tfvis.visor().setActiveTab('Evaluation')

  labels.dispose()
}

export const showConfusion = async (
  model,
  data,
  title = 'Confusion Matrix'
) => {
  const [preds, labels] = doPrediction(model, data)
  const confusionMatrix = await tfvis.metrics.confusionMatrix(labels, preds)
  const container = { name: title, tab: 'Evaluation' }
  tfvis.render.confusionMatrix(container, {
    values: confusionMatrix,
    tickLabels: classNames
  })

  labels.dispose()
}
  `
