import { useState, useRef, useCallback, useEffect } from 'react'
import Webcam from 'react-webcam'
import * as tfvis from '@tensorflow/tfjs-vis'
import * as tf from '@tensorflow/tfjs'
import '@tensorflow/tfjs-backend-webgl'
import Editor from '@monaco-editor/react'
import { RPSDataset } from './tfjs/data.js'
import { getAdvancedModel, getSimpleModel } from './tfjs/models.js'
import { train } from './tfjs/train.js'
import {
  showAccuracy,
  showConfusion,
  showExamples,
  doSinglePrediction
} from './tfjs/evaluationHelpers.js'
import AdvancedModel from './AdvancedModel.jsx'
import { Disclosure, DisclosureButton, DisclosurePanel } from '@headlessui/react'
import "./Buttons.css"
import "./App.css"
import gant from "./corn.png";
import {
  IMAGE_WIDTH,
  IMAGE_HEIGHT,
  NUM_CHANNELS,
  NUM_CLASSES,
  TEST_BATCH_SIZE
} from './tfjs/constants'

const simpleModelCode = `
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
}`

const advancedModelCode = `
export const getAdvancedModel = () => {
  const model = tf.sequential({
    name: 'advancedModel'
  })

  // First conv layer
  model.add(
    tf.layers.conv2d({
      inputShape: [IMAGE_WIDTH, IMAGE_HEIGHT, NUM_CHANNELS],
      kernelSize: 3,
      filters: 32,
      strides: 1,
      activation: 'relu',
      kernelInitializer: 'varianceScaling'
    })
  )
  model.add(tf.layers.maxPooling2d({ poolSize: [2, 2], strides: [2, 2] }))

  // Second conv layer
  model.add(
    tf.layers.conv2d({
      kernelSize: 3,
      filters: 64,
      strides: 1,
      activation: 'relu',
      kernelInitializer: 'varianceScaling'
    })
  )
  model.add(tf.layers.maxPooling2d({ poolSize: [2, 2], strides: [2, 2] }))

  // Third conv layer
  model.add(
    tf.layers.conv2d({
      kernelSize: 3,
      filters: 128,
      strides: 1,
      activation: 'relu',
      kernelInitializer: 'varianceScaling'
    })
  )
  model.add(tf.layers.maxPooling2d({ poolSize: [2, 2], strides: [2, 2] }))

  // Flatten output and feed it into dense layer
  model.add(tf.layers.flatten())
  model.add(
    tf.layers.dense({
      units: 64,
      kernelInitializer: 'varianceScaling',
      activation: 'relu'
    })
  )

  // Dropout to prevent overfitting
  model.add(tf.layers.dropout({ rate: 0.5 }))

  // Output layer
  model.add(
    tf.layers.dense({
      units: NUM_CLASSES,
      kernelInitializer: 'varianceScaling',
      activation: 'softmax'
    })
  )

  const optimizer = tf.train.adam(0.0001)
  model.compile({
    optimizer: optimizer,
    loss: 'categoricalCrossentropy',
    metrics: ['accuracy']
  })

  return model
}`

// Constants
const DETECTION_PERIOD = 2000
const WEBCAM_WIDTH = 640
const WEBCAM_HEIGHT = 480

function App() {
  // State management
  const [currentModel, setCurrentModel] = useState(null)
  const [webcamActive, setWebcamActive] = useState(false)
  const [camMessage, setCamMessage] = useState('')
  const [advancedDemo, setAdvancedDemo] = useState(false)
  const [loadDataMessage, setLoadDataMessage] = useState('Load and Show Examples')
  const [isVisorOpen, setIsVisorOpen] = useState(false)
  const [prediction, setPrediction] = useState(null)
  const [isDetecting, setIsDetecting] = useState(false)
  const detectionInterval = useRef(null)

  // Refs
  const webcamRef = useRef(null)
  const modelRef = useRef(null)
  const dataRef = useRef(null)

  // Effect to set tf on window for debugging
  useEffect(() => {
    window.tf = tf
  }, [])

  useEffect(() => {
    // Initialize tfvis in a closed state
    const visor = tfvis.visor()
    visor.close()
  }, [])

  // Add useEffect to handle backend initialization
  useEffect(() => {
    const initTF = async () => {
      await tf.setBackend('webgl')
      console.log('Using backend:', tf.getBackend())
    }
    initTF()
  }, [])

  // Handlers
  const handleCreateSimpleModel = useCallback(async () => {
    setCurrentModel('Simple')
    const model = getSimpleModel()
    // Make sure the visor is open when showing model summary
    tfvis.visor().open()
    // Show the model summary
    tfvis.show.modelSummary(
      { name: 'Simple Model Architecture', tab: 'Model' },
      model
    )
    modelRef.current = model
  }, [])

  const handleCreateAdvancedModel = useCallback(() => {
    // Create model
    modelRef.current = getAdvancedModel()
    setCurrentModel('Advanced')

    // Show model summary in visor
    const visor = tfvis.visor()
    visor.open()
    setIsVisorOpen(true)
    
    tfvis.show.modelSummary(
      { name: 'Advanced Model Architecture', tab: 'Model' },
      modelRef.current
    )
  }, [])

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

  const handleTrainModel = useCallback(async () => {
    if (!dataRef.current || !modelRef.current) {
      alert('Please load data and create a model first')
      return
    }

    // Open the visor to show training progress
    tfvis.visor().open()

    // Use 10 epochs for both models
    const numEpochs = 10

    try {
      await train(modelRef.current, dataRef.current, numEpochs)

      // After training, show the updated metrics
      await showAccuracy(modelRef.current, dataRef.current, 'Trained Accuracy')
      await showConfusion(modelRef.current, dataRef.current, 'Trained Confusion Matrix')
    } catch (error) {
      console.error('Training failed:', error)
      alert('Training failed. Check console for details.')
    }
  }, [currentModel])

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

  const handleToggleWebcam = useCallback(() => {
    if (webcamActive) {
      // Stop detection if running
      if (detectionInterval.current) {
        clearInterval(detectionInterval.current)
        setIsDetecting(false)
      }
      setWebcamActive(false)
      setCamMessage('')
    } else {
      setWebcamActive(true)
      setCamMessage('Loading webcam...')
    }
  }, [webcamActive])

  const handleStartDetection = useCallback(() => {
    if (!modelRef.current) {
      alert('Please train a model first!')
      return
    }

    if (isDetecting) {
      clearInterval(detectionInterval.current)
      setIsDetecting(false)
      return
    }

    setIsDetecting(true)
    detectionInterval.current = setInterval(async () => {
      if (webcamRef.current) {
        // Wrap prediction in tf.tidy to clean up tensors
        const predictions = await tf.tidy(() => 
          doSinglePrediction(modelRef.current, webcamRef.current.video)
        )
        // Find the prediction with highest probability
        const topPrediction = predictions.reduce((prev, current) => 
          (current.probability > prev.probability) ? current : prev
        )
        setPrediction(topPrediction)
      }
    }, DETECTION_PERIOD)
  }, [isDetecting])

  useEffect(() => {
    return () => {
      if (detectionInterval.current) {
        clearInterval(detectionInterval.current)
      }
    }
  }, [])

  return (
    <div className="App">
      <header className="App-header">
        <h2>Rock Paper Scissors</h2>
        <h3>Machine Learning in the browser with TensorFlow.js</h3>
        <img src="./rps_circle.png" className="App-logo" alt="logo" id="logo" />
        <a className="App-link" href="https://infinite.red" target="_blank" rel="noopener noreferrer">
          Infinite Red
        </a>
        <a className="App-link" href="http://gantlaborde.com/" target="_blank" rel="noopener noreferrer">
          Gant Laborde
        </a>
      </header>

      <div className="Main">
        <section className="flex flex-col items-center justify-center">
          <p>
            We'll be working with a fun dataset for the classic game, "Rock Paper Scissors",
            provided here:{' '}
            <a
              href="http://www.laurencemoroney.com/rock-paper-scissors-dataset/"
              target="_blank"
              rel="noopener noreferrer"
            >
              Rock Paper Scissors Dataset
            </a>
          </p>
          <img src="./rps.jpg" alt="Rock Paper Scissors dataset" />

          <p>
            We'll show progress in the TensorFlow.js Vis panel. You'll see it
            when you click the load and show button below. Press{" "}
            <span className="cod">`</span> or <span className="cod">~</span> key
            to hide this menu.
          </p>

          <div className="GroupUp">
            <button className="btn-3d blue" onClick={handleLoadData}>
              {loadDataMessage}
            </button>
          </div>
        </section>

        <section className="flex flex-col items-center justify-center gap-4 mb-6">
        <p>
            Each of the examples have been loaded now. Due to this being a
            browser, the data is loaded with one{" "}
            <a href="./data.png" target="_blank" rel="noopener noreferrer">
              sprite-sheet
            </a>{" "}
            to get around sandboxing. My code to create sprite-sheets is
            available with{" "}
            <a
              href="https://github.com/GantMan/rps_tfjs_demo"
              target="_blank"
              rel="noopener noreferrer"
            >
              this repo on GitHub
            </a>
            .
          </p>
          <p>
            You now create the structure for the data, that hopefully works
            best.{" "}
            <strong>
              In this situation, an advanced model is a bad choice.
            </strong>{" "}
            An advanced model will train slower while overfitting this small and
            simple training data.
          </p>

          <div className="model-selection">
            <div className="button-group">
              <Disclosure as="div" className="code-disclosure">
                <div className="mb-2">
                  <button
                    className={`btn-3d blue ${currentModel === 'Simple' ? 'activeModel' : ''}`}
                    onClick={handleCreateSimpleModel}
                  >
                    Create Simple Model
                  </button>
                  <DisclosureButton className="btn-3d green">
                    <strong>&lt;/&gt;</strong>
                  </DisclosureButton>
                </div>
                <DisclosurePanel className="code-panel">
                  <Editor
                    height="600px"
                    defaultLanguage="javascript"
                    theme="vs-dark"
                    value={simpleModelCode}
                    options={{
                      readOnly: true,
                      minimap: { enabled: false },
                      scrollBeyondLastLine: false,
                      fontSize: 14
                    }}
                  />
                </DisclosurePanel>
              </Disclosure>
            </div>

            <p className="my-4 text-center">OR</p>

            <div className="button-group">
              <Disclosure as="div" className="code-disclosure">
                <div className="mb-2">
                  <button
                    className={`btn-3d blue ${currentModel === 'Advanced' ? 'activeModel' : ''}`}
                    onClick={handleCreateAdvancedModel}
                  >
                    Create Advanced Model
                  </button>
                  <DisclosureButton className="btn-3d green">
                    <strong>&lt;/&gt;</strong>
                  </DisclosureButton>
                </div>
                <DisclosurePanel className="code-panel">
                  <Editor
                    height="600px"
                    defaultLanguage="javascript"
                    theme="vs-dark"
                    value={advancedModelCode}
                    options={{
                      readOnly: true,
                      minimap: { enabled: false },
                      scrollBeyondLastLine: false,
                      fontSize: 14
                    }}
                  />
                </DisclosurePanel>
              </Disclosure>
            </div>
          </div>

          <div className="GroupUp">
            <button
              className="btn-3d blue"
              onClick={handleCheckUntrainedModel}
            >
              Check Untrained Model Results
            </button>
          </div>

          <p className="my-4 text-center">
            Train your Model with your training data. In this case 2100 labeled images,
            over and over... but not <em>toooooo much.</em>
          </p>

          <div className="GroupUp">
            <button
              className="btn-3d blue"
              onClick={handleTrainModel}
            >
              Train Your {currentModel} Model
            </button>
          </div>
        </section>

        <section className="flex flex-col items-center justify-center gap-4 mb-6">
        <div>
            Now that our model has seen some stuff{" "}
            <span role="img" aria-label="woah">
              😳
            </span>
            <hr />
            It should be smarter at identifying RPS! We can now test it with 420
            RPS images it's never seen before.
          </div>
          {/* We don't need this button for now, as tfjs vis already shows the accuracy and confusion matrix */}
          {/* <button
            className="btn-3d blue"
            onClick={async () => {
              // stop errors
              if (!this.data) return;
              if (!this.model) return;
              await showAccuracy(this.model, this.data, "Trained Accuracy");
              await showConfusion(
                this.model,
                this.data,
                "Trained Confusion Matrix"
              );
            }}
          >
            Check Model After Training
          </button> */}
          <p>
            We can now save our trained model! We can store it via downloading
            it, uploading it, or place the results in localstorage for access of
            the browser.
          </p>
          <p>
            The simple model size comes out to about 48Kb, but some models can
            be as large as 20+MBs! It depends how simple you keep the model. If
            you want the model trained above, you get two files by{" "}
            <a
              className="pointy"
              onClick={async () => {
                if (!this.model) return;
                await this.model.save("downloads://rps-model");
              }}
            >
              clicking here
            </a>
            . The <span className="cod">model.json</span> file demonstrates the
            structure of the model, and the weights are our non-random trained
            values that make the model accurate.
          </p>
          <h3>Now let's see if we can test our model with the real world!</h3>
          <img src="./rps_webcam_big.jpg" className="demo" alt="webcam demo" />
          <p>
            Keep in mind, the training data for this model had no background,
            and the model itself isn't practiced in dealing with noise and
            rotation. A more advanced model would do better, but for this demo
            you shouldn't have any problems getting consistent and accurate
            results. When testing on a webcam, you'll need to make the images as
            clean as you can. Every few seconds your webcam image will be
            converted to a 64x64 grayscale image for your model to classify.
          </p>
        </section>

        <section className="flex flex-col items-center justify-center">
          <h3 className="my-4 text-center">Test with Webcam</h3>
          <p className="my-4 text-center">Try out your trained model with your webcam!</p>
          
          <div className="GroupUp">
            <button 
              className="btn-3d blue" 
              onClick={handleToggleWebcam}
            >
              {webcamActive ? 'Disable Webcam' : 'Enable Webcam'}
            </button>
            
            {webcamActive && (
              <button 
                className="btn-3d blue"
                onClick={handleStartDetection}
              >
                {isDetecting ? 'Stop Detection' : 'Start Detection'}
              </button>
            )}
          </div>

          <div className="webcam-container">
            {webcamActive && (
              <>
                <Webcam
                  ref={webcamRef}
                  audio={false}
                  screenshotFormat="image/jpeg"
                  videoConstraints={{
                    width: WEBCAM_WIDTH,
                    height: WEBCAM_HEIGHT,
                    facingMode: 'user'
                  }}
                  width={WEBCAM_WIDTH}
                  height={WEBCAM_HEIGHT}
                  onUserMedia={() => setCamMessage('')}
                />
                {prediction && (
                  <div className="prediction">
                    {prediction.className}: {(prediction.probability * 100).toFixed(2)}%
                  </div>
                )}
              </>
            )}
            {camMessage && <div className="cam-message">{camMessage}</div>}
          </div>
        </section>
      </div>
      <div className="GroupUp">
        <p className="outro">
          Follow me (Gant Laborde) and Infinite Red for cool new experiments,
          and let us know what cool things you've come up with.{" "}
          <em>
            We can help, we're available for AI consulting and{" "}
            <a
              href="https://academy.infinite.red/"
              target="_blank"
              rel="noopener noreferrer"
            >
              training
            </a>
            .
          </em>
        </p>
      </div>
      <div className="GroupUp">
        <img src={gant} className="wiggle me" alt="Gant Laborde" />
        <ul id="footer">
          <li>
            Website:{" "}
            <a
              href="http://gantlaborde.com"
              target="_blank"
              rel="noopener noreferrer"
            >
              GantLaborde.com
            </a>
          </li>
          <li>
            Twitter:{" "}
            <a
              href="https://twitter.com/gantlaborde"
              target="_blank"
              rel="noopener noreferrer"
            >
              @GantLaborde
            </a>
          </li>
          <li>
            Medium:{" "}
            <a
              href="https://medium.freecodecamp.org/@gantlaborde"
              target="_blank"
              rel="noopener noreferrer"
            >
              GantLaborde
            </a>
          </li>
          <li>
            ML Twitter:{" "}
            <a
              href="https://twitter.com/FunMachineLearn"
              target="_blank"
              rel="noopener noreferrer"
            >
              FunMachineLearn
            </a>
          </li>
          <li>
            GitHub:{" "}
            <a
              href="https://github.com/GantMan/rps_tfjs_demo"
              target="_blank"
              rel="noopener noreferrer"
            >
              RPS TFJS Demo
            </a>
          </li>
          <li>
            Newsletter:{" "}
            <a
              href="https://ai-fyi.com"
              target="_blank"
              rel="noopener noreferrer"
            >
              AI-FYI.com
            </a>
          </li>
          <li>
            <a
              href="https://infinite.red"
              target="_blank"
              rel="noopener noreferrer"
            >
              <img src="./ir.svg" id="InfiniteRed" alt="Infinite Red" />
            </a>
          </li>
        </ul>
      </div>
      <div className="GroupUp">
        <img src="./ml.png" id="closer" alt="RPS" />
        <h4>powered by</h4>
        <img
          src="./TF_FullColor_Horizontal.png"
          id="closer"
          alt="Tensorflow logo"
          style={{ paddingLeft: "-40px" }}
        />
      </div>
    </div>
  )
}

export default App
