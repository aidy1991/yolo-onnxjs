import ndaarray from 'ndarray';
import { Tensor, InferenceSession } from 'onnxjs'
import "babel-polyfill";

const main = () => {
  const video = document.getElementById("video");

  const media = navigator.mediaDevices.getUserMedia({
      video: true,
      audio: false,
  });

  media.then((stream) => {
      video.srcObject = stream;
  });

  video.addEventListener("timeupdate", () => {
    const canvas = document.getElementById("canvas");
    const ctx = canvas.getContext("2d");
    ctx.drawImage(video, 0, 0, 640, 480);
    runModel(ctx);
  }, true);
}

const runModel = async (ctx) => {
  const session = new InferenceSession({ backendHint: 'webgl' })
  const modelFile = './yolo3-tiny.onnx';

  await session.loadModel(modelFile);

  const imageData = ctx.getImageData(0, 0, ctx.canvas.width, ctx.canvas.height);
  const { data, width, height } = imageData;
  console.log(data, width, height);

  // data processing
  const dataTensor = ndarray(new Float32Array(data), [width, height, 4]);
  const dataProcessedTensor = ndarray(new Float32Array(width * height * 3), [1, 3, width, height]);
  ops.assign(dataProcessedTensor.pick(0, 0, null, null), dataTensor.pick(null, null, 0));
  ops.assign(dataProcessedTensor.pick(0, 1, null, null), dataTensor.pick(null, null, 1));
  ops.assign(dataProcessedTensor.pick(0, 2, null, null), dataTensor.pick(null, null, 2));
  const tensor = new Tensor(new Float32Array(width* height* 3), 'float32', [1, 3, width, height]);
  tensor.data.set(dataProcessedTensor.data);

  const outputData = await session.run([inputTensor])
  console.log(outputData);
}

window.onload = function() {
  main();
};
