const { Interpreter } = require("node-tflite");
const fs = require("fs");
const softmax = require("softmax-fn");
const Jimp = require("jimp");
const decode = require("image-decode");

class ClassificationClass {
  interpreter;

  constructor(pathToModel) {
    this.interpreter = new Interpreter(fs.readFileSync(pathToModel));
  }

  /**
   * Returns the result of the Microsoft Azure TFLite Model as number[]
   * @param {Buffer} imgBuffer - Image as buffer
   */
  async executeAsync(imgBuffer) {
    if (!this.interpreter)
      return [NaN, NaN, NaN, NaN, NaN, NaN, NaN, NaN, NaN, NaN];
    const size = 300;
    const rgbFloat = await getRGB(await resizeImage(size, imgBuffer));
    this.interpreter.allocateTensors();
    this.interpreter.inputs[0].copyFrom(rgbFloat);
    this.interpreter.invoke();
    const scoreData = new Float32Array(11);
    this.interpreter.outputs[0].copyTo(scoreData);
    return Array.prototype.slice.call(scoreData, 0);
  }
}

class ObjectDetectionClass {
  interpreter;

  constructor(pathToModel) {
    this.interpreter = new Interpreter(fs.readFileSync(pathToModel));
  }

  /**
   * Returns the result of the Microsoft Azure TFLite Model as [boxes: [number, number, number, number], scores: number[]]
   * @param {Buffer} imgBuffer - Image as buffer
   */
  async executeAsync(imgBuffer) {
    const size = 416;
    const rgbFloat = await getRGB(await resizeImage(size, imgBuffer));
    this.interpreter.allocateTensors();
    this.interpreter.inputs[0].copyFrom(rgbFloat);
    this.interpreter.invoke();

    const scoreData = new Float32Array(13 * 13 * 30);
    this.interpreter.outputs[0].copyTo(scoreData);

    let outputs = [];
    let count = 0;
    for (let x = 0; x < 13; x += 1) {
      outputs.push([]);
      for (let y = 0; y < 13; y += 1) {
        outputs[x].push([]);
        for (let z = 0; z < 30; z += 1) {
          outputs[x][y].push(scoreData[count]);
          count += 1;
        }
      }
    }
    outputs = [outputs];

    const anchors = [
      0.573, 0.677, 1.87, 2.06, 3.34, 5.47, 7.88, 3.53, 9.77, 9.17,
    ];

    const num_anchor = anchors.length / 2;
    const channels = outputs[0][0][0].length;
    const height = outputs[0].length;
    const width = outputs[0][0].length;

    const num_class = channels / num_anchor - 5;

    let boxes = [];
    let scores = [];

    for (var grid_y = 0; grid_y < height; grid_y++) {
      for (var grid_x = 0; grid_x < width; grid_x++) {
        let offset = 0;

        for (var i = 0; i < num_anchor; i++) {
          let x =
            (this._logistic(outputs[0][grid_y][grid_x][offset++]) + grid_x) /
            width;
          let y =
            (this._logistic(outputs[0][grid_y][grid_x][offset++]) + grid_y) /
            height;
          let w =
            (Math.exp(outputs[0][grid_y][grid_x][offset++]) * anchors[i * 2]) /
            width;
          let h =
            (Math.exp(outputs[0][grid_y][grid_x][offset++]) *
              anchors[i * 2 + 1]) /
            height;
          let oneD = this._logistic(outputs[0][grid_y][grid_x][offset++]);
          let class_probabilities = softmax(
            outputs[0][grid_y][grid_x].slice(offset, offset + num_class)
          );
          offset += num_class;
          const multipled = this.multiplyToOne(class_probabilities, oneD);
          const answer = Math.max(...multipled);
          if (answer > 0.05) {
            const xpos = x - w / 2 < 0 ? 0 : x - w / 2;
            const ypos = y - h / 2 < 0 ? 0 : y - h / 2;
            const xwidth = x + w / 2 < 0 ? 0 : x + w / 2;
            const yheight = y + h / 2 < 0 ? 0 : y + h / 2;
            boxes.push([xpos, ypos, xwidth, yheight]);
            scores.push(answer);
          }
        }
      }
    }
    return [boxes, scores];
  }

  multiplyToOne(one, two) {
    let returnedArr = [];
    for (const i in one) {
      returnedArr.push(one[i] * two);
    }
    return returnedArr;
  }

  _logistic(x) {
    if (x > 0) {
      return 1 / (1 + Math.exp(-x));
    } else {
      const e = Math.exp(x);
      return e / (1 + e);
    }
  }
}

async function resizeImage(size, imageBuf) {
  let imgBuf;
  const img = await Jimp.read(imageBuf);
  img.resize(size, size);
  img.getBuffer(Jimp.MIME_PNG, (err, buffer) => {
    imgBuf = buffer;
  });
  return decode(imgBuf);
}

async function getRGB(img2) {
  const imgDecoded = Array.prototype.slice.call(img2.data, 0);
  let i = 0;
  const newArr = [];
  while (i != imgDecoded.length) {
    i += 1;
    if (!((i + 1) % 4 === 0)) {
      newArr.push(imgDecoded[i]);
    }
  }
  const rgbFloat = new Float32Array(newArr);
  return rgbFloat;
}
