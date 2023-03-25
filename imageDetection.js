const STATUS = document.getElementById('status');
const VIDEO = document.getElementById('webcam');
const VIDEO_1 = document.getElementById('webcam1');
const VIDEO_2 = document.getElementById('webcam2');
const ENABLE_CAM_BUTTON = document.getElementById('enableCam');
const ENABLE_CAM_BUTTON_1 = document.getElementById('enableCam1');
const INPUT_1 = document.getElementById('class1_input');
const INPUT_2 = document.getElementById('class2_input');

const TRAIN_BUTTON = document.getElementById('train');
const MOBILE_NET_INPUT_WIDTH = 224;
const MOBILE_NET_INPUT_HEIGHT = 224;
const DATA_COLLECTOR_BUTTON_1 = document.getElementById('data-collector-1');
const DATA_COLLECTOR_BUTTON_2 = document.getElementById('data-collector-2');
const STOP_DATA_GATHER = -1;
const CLASS_NAMES = [];
let videoPlaying_1 = false;
let videoPlaying_2 = false;
let videoPlaying_3 = false;
let videoPlaying = false;
let snapshots = [];

function changeVisibility(elRef, displayProp) {
  elRef.style.visibility = displayProp;
}

INPUT_1.addEventListener('change', (e) => {
  const text = document.getElementById('data-collector-1');
  document.getElementById('data-collector-1').innerText += ' (' + e.target.value + ')';
  text.setAttribute('data-name', e.target.value);
  CLASS_NAMES[0] = e.target.value;
});

INPUT_2.addEventListener('change', (e) => {
  const text = document.getElementById('data-collector-2');
  document.getElementById('data-collector-2').innerText += ' (' + e.target.value + ')';
  text.setAttribute('data-name', e.target.value);
  CLASS_NAMES[1] = e.target.value;
});

ENABLE_CAM_BUTTON.addEventListener('click', () => enableCam(VIDEO));
ENABLE_CAM_BUTTON_1.addEventListener('click', () => enableCam(VIDEO_1));

webcamStop.addEventListener('click', () => stopGather1(VIDEO));
webcam1Stop.addEventListener('click', () => stopGather2(VIDEO_1));

TRAIN_BUTTON.addEventListener('click', () => {
  enableCam(VIDEO_2, trainAndPredict);
});

let dataCollectorButtons = document.querySelectorAll('button.dataCollector');
function stopGather1() {
  gatherDataState = STOP_DATA_GATHER;
  videoPlaying_1 = false;
}
function stopGather2() {
  gatherDataState = STOP_DATA_GATHER;

  // videoPlaying_2 = false;
}

// Just add more buttons in HTML to allow classification of more classes of data!
for (let i = 0; i < dataCollectorButtons.length; i++) {
  dataCollectorButtons[i].addEventListener('mousedown', gatherDataForClass);
  dataCollectorButtons[i].addEventListener('mouseup', gatherDataForClass);
  // For mobile.
  dataCollectorButtons[i].addEventListener('touchend', gatherDataForClass);

  // Populate the human readable names for classes.
  CLASS_NAMES.push(dataCollectorButtons[i].getAttribute('data-name'));
}

let mobilenet = undefined;
let gatherDataState = STOP_DATA_GATHER;
let trainingDataInputs = [];
let trainingDataOutputs = [];
let examplesCount = [];
let predict = false;

/**
 * Loads the MobileNet model and warms it up so ready for use.
 **/
async function loadMobileNetFeatureModel() {
  const URL = 'https://tfhub.dev/google/tfjs-model/imagenet/mobilenet_v3_small_100_224/feature_vector/5/default/1';
  mobilenet = await tf.loadGraphModel(URL, { fromTFHub: true });
  STATUS.innerText = 'AI ready for training and predict';

  // Warm up the model by passing zeros through it once.
  tf.tidy(function () {
    let answer = mobilenet.predict(tf.zeros([1, MOBILE_NET_INPUT_HEIGHT, MOBILE_NET_INPUT_WIDTH, 3]));
    console.log(answer.shape);
  });
}

loadMobileNetFeatureModel();

let model = tf.sequential();
model.add(tf.layers.dense({ inputShape: [1024], units: 128, activation: 'relu' }));
model.add(tf.layers.dense({ units: CLASS_NAMES.length, activation: 'softmax' }));

model.summary();

// Compile the model with the defined optimizer and specify a loss function to use.
model.compile({
  // Adam changes the learning rate over time which is useful.
  optimizer: 'adam',
  // Use the correct loss function. If 2 classes of data, must use binaryCrossentropy.
  // Else categoricalCrossentropy is used if more than 2 classes.
  loss: CLASS_NAMES.length === 2 ? 'binaryCrossentropy' : 'categoricalCrossentropy',
  // As this is a classification problem you can record accuracy in the logs too!
  metrics: ['accuracy'],
});

/**
 * Check if getUserMedia is supported for webcam access.
 **/
function hasGetUserMedia() {
  return !!(navigator.mediaDevices && navigator.mediaDevices.getUserMedia);
}

/**
 * Enable the webcam with video constraints applied.
 **/
function enableCam(VIDEO_ELEMENT, fn) {
  if (hasGetUserMedia()) {
    // getUsermedia parameters.
    const constraints = {
      video: true,
      width: 280,
      height: 280,
    };
    // console.log("Vide", VIDEO);
    if (VIDEO_ELEMENT === VIDEO) {
      videoPlaying_1 = true;
      videoPlaying_2 = false;

      VIDEO_1.pause();
      snapshots = [];
    } else if (VIDEO_ELEMENT === VIDEO_1) {
      videoPlaying_2 = true;
      videoPlaying_1 = false;
      VIDEO.pause();
      snapshots = [];
    } else {
      VIDEO.pause();
      VIDEO_1.pause();
      videoPlaying_3 = true;
    }

    // Activate the webcam stream.
    navigator.mediaDevices.getUserMedia(constraints).then(function (stream) {
      VIDEO_ELEMENT.srcObject = stream;
      VIDEO_ELEMENT.addEventListener('loadeddata', function () {
        videoPlaying = true;
        if (fn) {
          fn();
        }
      });
    });
  } else {
    console.warn('getUserMedia() is not supported by your browser');
  }
}

/**
 * Handle Data Gather for button mouseup/mousedown.
 **/
function gatherDataForClass() {
  let classNumber = parseInt(this.getAttribute('data-1hot'));
  if (classNumber == 0) {
    videoPlaying_1 = true;
  }
  gatherDataState = gatherDataState === STOP_DATA_GATHER ? classNumber : STOP_DATA_GATHER;
  dataGatherLoop();
}

function calculateFeaturesOnCurrentFrame(video) {
  return tf.tidy(function () {
    // Grab pixels from current VIDEO frame.
    let videoFrameAsTensor = tf.browser.fromPixels(video);
    // Resize video frame tensor to be 224 x 224 pixels which is needed by MobileNet for input.
    let resizedTensorFrame = tf.image.resizeBilinear(
      videoFrameAsTensor,
      [MOBILE_NET_INPUT_HEIGHT, MOBILE_NET_INPUT_WIDTH],
      true
    );

    let normalizedTensorFrame = resizedTensorFrame.div(255);

    try {
      shoot(video);
    } catch (e) {}
    return mobilenet.predict(normalizedTensorFrame.expandDims()).squeeze();
  });
}

/**
 * When a button used to gather data is pressed, record feature vectors along with class type to arrays.
 **/
function dataGatherLoop() {
  // Only gather data if webcam is on and a relevent button is pressed.
  if (gatherDataState !== STOP_DATA_GATHER) {
    // Ensure tensors are cleaned up.
    let imageFeatures;
    if (videoPlaying_1) imageFeatures = calculateFeaturesOnCurrentFrame(VIDEO);
    else imageFeatures = calculateFeaturesOnCurrentFrame(VIDEO_1);
    // let imageFeatures_1 = calculateFeaturesOnCurrentFrame(VIDEO_1);
    // console.log(CLASS_NAMES)
    trainingDataInputs.push(imageFeatures);
    // trainingDataInputs.push(imageFeatures_1);
    trainingDataOutputs.push(gatherDataState);

    // Intialize array index element if currently undefined.
    if (examplesCount[gatherDataState] === undefined) {
      examplesCount[gatherDataState] = 0;
    }
    // Increment counts of examples for user interface to show.
    examplesCount[gatherDataState]++;
    if (TRAIN_BUTTON.disabled && examplesCount[0] > 0 && examplesCount[1] > 0) TRAIN_BUTTON.disabled = false;

    STATUS.innerText = '';
    for (let n = 0; n < CLASS_NAMES.length; n++) {
      STATUS.innerText += CLASS_NAMES[n] + ' data count: ' + examplesCount[n] + '. ';
      STATUS.innerHTML += '<br/>';
    }

    window.requestAnimationFrame(dataGatherLoop);
  }
}

/**
 * Once data collected actually perform the transfer learning.
 **/
async function trainAndPredict() {
  predict = false;
  tf.util.shuffleCombo(trainingDataInputs, trainingDataOutputs);

  let outputsAsTensor = tf.tensor1d(trainingDataOutputs, 'int32');
  let oneHotOutputs = tf.oneHot(outputsAsTensor, CLASS_NAMES.length);
  let inputsAsTensor = tf.stack(trainingDataInputs);

  let results = await model.fit(inputsAsTensor, oneHotOutputs, {
    shuffle: true,
    batchSize: 5,
    epochs: 10,
    callbacks: { onEpochEnd: logProgress },
  });

  outputsAsTensor.dispose();
  oneHotOutputs.dispose();
  inputsAsTensor.dispose();

  predict = true;
  predictLoop();
}

/**
 * Log training progress.
 **/
function logProgress(epoch, logs) {
  console.log('Data for epoch ' + epoch, logs);
}

/**
 *  Make live predictions from webcam once trained.
 **/
function predictLoop() {
  if (predict) {
    tf.tidy(function () {
      let imageFeatures = calculateFeaturesOnCurrentFrame(VIDEO_2);
      let prediction = model.predict(imageFeatures.expandDims()).squeeze();
      let highestIndex = prediction.argMax().arraySync();
      let predictionArray = prediction.arraySync();
      predictionText.innerText = 'Prediction: ' + CLASS_NAMES[highestIndex];
    });

    window.requestAnimationFrame(predictLoop);
  }
}

var scaleFactor = 0.15;
function capture(video, scaleFactor) {
  if (scaleFactor == null) {
    scaleFactor = 1;
  }
  var w = video.videoWidth * scaleFactor;
  var h = video.videoHeight * scaleFactor;
  var canvas = document.createElement('canvas');
  canvas.width = w;
  canvas.height = h;
  var ctx = canvas.getContext('2d');
  ctx.drawImage(video, 0, 0, w, h);
  return canvas;
}

/**
 * Invokes the <code>capture</code> function and attaches the canvas element to the DOM.
 */
function shoot(VIDEO_ELEMENT) {
  var output = document.getElementById(VIDEO_ELEMENT.id + '-out');
  var canvas = capture(VIDEO_ELEMENT, scaleFactor);
  canvas.onclick = function () {
    window.open(this.toDataURL(image / jpg));
  };
  snapshots.unshift(canvas);
  output.innerHTML = '';
  for (var i = 0; i < 200; i++) {
    output.appendChild(snapshots[i]);
  }
}
