import React from 'react'
import { TFWrapper } from './tfjs/evaluationHelpers'
import * as tf from '@tensorflow/tfjs'

// Update the URLs to use direct paths
const ADV_RPS_MODEL_URL = '/adv_rps/'
const ADV_LABELS_URL = `${ADV_RPS_MODEL_URL}labels.json`
const ADV_MODEL_JSON = `${ADV_RPS_MODEL_URL}model.json`

// Convert to functional component
function AdvancedModel() {
  const videoRef = React.useRef(null)
  const canvasRef = React.useRef(null)
  const [loading, setLoading] = React.useState(true)

  const detectFrame = React.useCallback((video, model, labels) => {
    TFWrapper(model)
      .detect(video)
      .then(predictions => {
        renderPredictions(predictions, labels)
        requestAnimationFrame(() => {
          // calm down when hidden!
          if (canvasRef.current) {
            detectFrame(video, model, labels)
          }
        })
      })
  }, [])

  const renderPredictions = React.useCallback((predictions, labels) => {
    if (canvasRef.current) {
      const ctx = canvasRef.current.getContext('2d')
      ctx.clearRect(0, 0, ctx.canvas.width, ctx.canvas.height)
      // Font options.
      const font = '16px sans-serif'
      ctx.font = font
      ctx.textBaseline = 'top'
      predictions.forEach(prediction => {
        const x = prediction.bbox[0]
        const y = prediction.bbox[1]
        const width = prediction.bbox[2]
        const height = prediction.bbox[3]
        const label = labels[parseInt(prediction.class)]
        // Draw the bounding box.
        ctx.strokeStyle = '#FF0000'
        ctx.lineWidth = 4
        ctx.strokeRect(x, y, width, height)
        // Draw the label background.
        ctx.fillStyle = '#FF0000'
        const textWidth = ctx.measureText(label).width
        const textHeight = parseInt(font, 10) // base 10
        ctx.fillRect(x, y, textWidth + 4, textHeight + 4)
      })

      predictions.forEach(prediction => {
        const x = prediction.bbox[0]
        const y = prediction.bbox[1]
        const label = labels[parseInt(prediction.class)]
        // Draw the text last to ensure it's on top.
        ctx.fillStyle = '#000000'
        ctx.fillText(label, x, y)
      })
    }
  }, [])

  React.useEffect(() => {
    if (navigator.mediaDevices && navigator.mediaDevices.getUserMedia) {
      const webCamPromise = navigator.mediaDevices
        .getUserMedia({
          audio: false,
          video: {
            facingMode: 'user'
          }
        })
        .then(stream => {
          window.stream = stream
          if (videoRef.current) {
            videoRef.current.srcObject = stream
            return new Promise((resolve) => {
              videoRef.current.onloadedmetadata = () => {
                resolve()
              }
            })
          }
        })

      const modelPromise = tf.loadGraphModel(ADV_MODEL_JSON)
      const labelsPromise = fetch(ADV_LABELS_URL).then(data => data.json())
      
      Promise.all([modelPromise, labelsPromise, webCamPromise])
        .then(values => {
          const [model, labels] = values
          setLoading(false)
          detectFrame(videoRef.current, model, labels)
        })
        .catch(error => {
          console.error(error)
        })
    }

    // Cleanup function
    return () => {
      if (window.stream) {
        const tracks = window.stream.getTracks()
        tracks.forEach(track => {
          track.stop()
        })
        window.stream = null
      }
    }
  }, [detectFrame])

  return (
    <div className="advancedContainer">
      {loading && (
        <p id="advancedLoadText">Please wait, loading advanced model</p>
      )}
      <video
        className="advancedCam"
        autoPlay
        playsInline
        muted
        ref={videoRef}
        width="600"
        height="500"
      />
      <canvas
        className="advancedBox"
        ref={canvasRef}
        width="600"
        height="500"
      />
    </div>
  )
}

export default AdvancedModel
