const tf = require('@tensorflow/tfjs-node');
const InputError = require('../exceptions/InputError');

// A simple vocabulary mapping for demonstration purposes
const vocabulary = {
    'this': 1, 'is': 2, 'a': 3, 'test': 4, 'message': 5, // add more words as needed
};

async function predictClassification(model, text) {
    try {
        // Preprocess the text input
        const processedText = preprocessText(text);

        // Convert the preprocessed text to a tensor with shape [1, 85]
        const inputTensor = tf.tensor2d([processedText], [1, 85]);

        // Define the classes for hate speech detection
        const classes = ['netral', 'cyberbullying', 'hatespeech', 'judol'];

        // Perform the prediction
        const prediction = model.predict(inputTensor);
        const scores = await prediction.data();
        const confidenceScore = Math.max(...scores) * 100;

        // Get the predicted class
        const classResult = tf.argMax(prediction, 1).dataSync()[0];
        const label = classes[classResult];

        // Define explanations and suggestions for each class
        let explanation, suggestion;

        switch (label) {
            case 'netral':
                explanation = "The text appears to be neutral and does not contain any offensive content.";
                suggestion = "No action needed.";
                break;
            case 'cyberbullying':
                explanation = "The text contains content that can be categorized as cyberbullying.";
                suggestion = "Please review and remove or revise the content to prevent harm.";
                break;
            case 'hatespeech':
                explanation = "The text contains hate speech which is abusive or threatening language expressing prejudice against a particular group.";
                suggestion = "Please review and remove or revise the content to prevent harm.";
                break;
            case 'judol':
                explanation = "The text contains spam or irrelevant content.";
                suggestion = "Avoid sending unsolicited messages.";
                break;
            default:
                explanation = "Unknown category.";
                suggestion = "No suggestion available.";
                break;
        }

        return { confidenceScore, label, explanation, suggestion };
    } catch (error) {
        throw new InputError(`Terjadi kesalahan input: ${error.message}`);
    }
}

function preprocessText(text) {
    // Tokenize the text, map tokens to integers using the vocabulary, and pad/truncate to length 85
    const maxLength = 85;
    const tokens = text.toLowerCase().replace(/[^\w\s]/gi, '').split(' ');
    const tokenIds = tokens.map(token => vocabulary[token] || 0); // Map to vocabulary, use 0 if not found
    const paddedTokenIds = tokenIds.concat(Array(maxLength - tokenIds.length).fill(0)).slice(0, maxLength);
    return paddedTokenIds;
}

module.exports = predictClassification;
