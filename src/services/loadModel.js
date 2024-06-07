const tf = require('@tensorflow/tfjs-node');

// Define a custom L2 regularizer class
class L2Regularizer {
  constructor(config) {
    this.l2 = config.l2 != null ? config.l2 : 0.01;
  }

  apply(x) {
    if (this.l2 === 0) {
      return tf.zeros([1]);
    }
    return tf.mul(this.l2, tf.sum(tf.square(x)));
  }

  getConfig() {
    return {
      l2: this.l2,
    };
  }

  static get className() {
    return 'L2';
  }
}

// Register the L2 regularizer
tf.serialization.registerClass(L2Regularizer);

async function loadModel() {
  try {
    const model = await tf.loadLayersModel(process.env.MODEL_URL);
    return model;
  } catch (error) {
    throw new Error(`Failed to load the model: ${error.message}`);
  }
}

module.exports = loadModel;
