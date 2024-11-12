import * as tf from '@tensorflow/tfjs'
import * as tfvis from '@tensorflow/tfjs-vis'

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
    
    console.log(`Epoch ${epoch + 1}: accuracy = ${(accuracy * 100).toFixed(2)}%`);

    if (accuracy >= TARGET_ACCURACY) {
      console.log(`\nReached target accuracy of ${(TARGET_ACCURACY * 100).toFixed(2)}%. Stopping training.`);
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
