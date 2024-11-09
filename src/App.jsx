import { useState, useRef, useCallback, useEffect } from 'react'
import Webcam from 'react-webcam'
import * as tfvis from '@tensorflow/tfjs-vis'
import * as tf from '@tensorflow/tfjs'
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

const simpleModelCode = `
export const getSimpleModel = () => {
  const model = tf.sequential()

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

  model.add(tf.layers.maxPooling2d({ poolSize: [2, 2], strides: [2, 2] }))

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

  model.add(tf.layers.flatten())

  const NUM_OUTPUT_CLASSES = 3
  model.add(
    tf.layers.dense({
      units: NUM_OUTPUT_CLASSES,
      kernelInitializer: 'varianceScaling',
      activation: 'softmax'
    })
  )

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
  const model = tf.sequential()

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

  model.add(tf.layers.maxPooling2d({ poolSize: [2, 2], strides: [2, 2] }))

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

  model.add(tf.layers.flatten())

  const NUM_OUTPUT_CLASSES = 3
  model.add(
    tf.layers.dense({
      units: NUM_OUTPUT_CLASSES,
      kernelInitializer: 'varianceScaling',
      activation: 'softmax'
    })
  )

  const optimizer = tf.train.adam()
  model.compile({
    optimizer: optimizer,
    loss: 'categoricalCrossentropy',
    metrics: ['accuracy']
  })

  return model
}`

// Constants
const DETECTION_PERIOD = 2000

function App() {
  // State management
  const [currentModel, setCurrentModel] = useState(null)
  const [webcamActive, setWebcamActive] = useState(false)
  const [camMessage, setCamMessage] = useState('')
  const [advancedDemo, setAdvancedDemo] = useState(false)
  const [loadDataMessage, setLoadDataMessage] = useState('Load and Show Examples')

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

  const handleCreateAdvancedModel = useCallback(async () => {
    setCurrentModel('Advanced')
    const model = getAdvancedModel()
    tfvis.show.modelSummary(
      { name: 'Advanced Model Architecture' },
      model
    )
    modelRef.current = model
  }, [])

  const handleCheckUntrained = useCallback(async () => {
    if (!dataRef.current || !modelRef.current) {
      alert('Please load data and create a model first')
      return
    }

    // Make sure the visor is open
    tfvis.visor().open()
    
    // Show accuracy metrics
    await showAccuracy(modelRef.current, dataRef.current)
    
    // Show confusion matrix
    await showConfusion(modelRef.current, dataRef.current, 'Untrained Matrix')
  }, [])

  const handleTrainModel = useCallback(async () => {
    if (!dataRef.current || !modelRef.current) {
      alert('Please load data and create a model first')
      return
    }

    // Open the visor to show training progress
    tfvis.visor().open()

    // Different number of epochs based on model type
    const numEpochs = currentModel === 'Simple' ? 12 : 20
    
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
    const data = new RPSDataset()
    dataRef.current = data
    await data.load()
    // Make sure the visor is open before showing examples
    tfvis.visor().open()
    await showExamples(data)
    setLoadDataMessage('Data Loaded!')
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
        <section>
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

          <div className="GroupUp">
            <button className="btn-3d blue" onClick={handleLoadData}>
              {loadDataMessage}
            </button>
          </div>
        </section>

        <section>
          <p>
            You now create the structure for the data, that hopefully works best.{' '}
            <strong>In this situation, an advanced model is a bad choice.</strong>{' '}
            An advanced model will train slower while overfitting this small and simple training data.
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
                    height="400px"
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

            <p>OR</p>

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
                    height="400px"
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
              onClick={handleCheckUntrained}
            >
              Check Untrained Model Results
            </button>
          </div>

          <p>
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
      </div>
    </div>
  )
}

export default App
