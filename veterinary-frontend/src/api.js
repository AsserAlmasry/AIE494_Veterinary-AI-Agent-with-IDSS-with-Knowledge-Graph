const BASE_URL = 'http://localhost:8000/api/v1';

/**
 * Submits cattle clinical data to the Veterinary AI inference pipeline.
 * @param {File} image - Cattle photograph (JPEG/PNG)
 * @param {number} weight - Animal weight in kg
 * @param {number} age - Animal age in years
 * @param {number} cowId - Manual Cow ID override
 * @param {string} sensorJson - Raw JSON string for sensor data
 * @param {boolean} generateReport - Whether to run the final LLM report stage
 * @returns {Promise<Object>} Final inference JSON result
 */
export const submitPrediction = async (image, weight, age, cowId, sensorJson = "{}", generateReport = true) => {
  const formData = new FormData();
  if (image) formData.append('image', image);
  if (weight) formData.append('animal_weight_kg', weight);
  if (age) formData.append('animal_age_years', age);
  if (cowId) formData.append('cow_id_override', cowId);
  formData.append('generate_report', generateReport);
  
  if (sensorJson) {
      formData.append('sensor_json', sensorJson);
  }

  try {
    const response = await fetch(`${BASE_URL}/predict`, {
      method: 'POST',
      body: formData,
    });

    if (!response.ok) {
      const errorData = await response.json().catch(() => ({}));
      throw new Error(errorData.detail || `Server error: ${response.status}`);
    }

    return await response.json();
  } catch (error) {
    console.error('API Error:', error);
    throw error;
  }
};

/**
 * Fetch health status of the backend API.
 */
export const checkHealth = async () => {
    try {
        const response = await fetch(`${BASE_URL}/health`);
        return response.ok;
    } catch {
        return false;
    }
};

/**
 * Sends a clinical question to the AI Agent for grounded answers.
 * @param {string} question - Query for the agent
 * @param {string} context - Optional context (e.g. current findings)
 */
export const clinicalQA = async (question, context = "") => {
    try {
        const response = await fetch(`${BASE_URL}/report/qa`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ question, disease_context: context }),
        });
        
        if (!response.ok) throw new Error("Agent Q&A failed");
        return await response.json();
    } catch (error) {
        console.error('QA Error:', error);
        throw error;
    }
};
