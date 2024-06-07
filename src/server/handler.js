const predictClassification = require('../services/inferenceService');
const crypto = require('crypto');
const storeData = require('../services/storeData');

async function postPredictHandler(request, h) {
    console.log('Request received:', request.payload); // Add this line
    const { text } = request.payload;
    const { model } = request.server.app;

    const { confidenceScore, label, explanation, suggestion } = await predictClassification(model, text);
    const id = crypto.randomUUID();
    const createdAt = new Date().toISOString();

    const data = {
        id: id,
        result: label,
        explanation: explanation,
        suggestion: suggestion,
        confidenceScore: confidenceScore,
        createdAt: createdAt
    };
    await storeData(id, data);

    const response = h.response({
        status: 'success',
        message: confidenceScore > 90 ? 'Model is predicted successfully.' : 'Model is predicted successfully but under threshold. Please use the correct input.',
        data
    });
    response.code(201);
    return response;
}

module.exports = postPredictHandler;
