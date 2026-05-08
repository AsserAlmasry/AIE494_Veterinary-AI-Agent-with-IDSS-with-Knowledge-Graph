const BASE_URL = 'http://localhost:8000/api/v1';

/**
 * Submit full prediction pipeline.
 */
export const submitPrediction = async (image, weight, age, cowId, sensorJson = "{}", generateReport = true, dayIndex = 0) => {
  const formData = new FormData();
  if (image) formData.append('image', image);
  if (weight) formData.append('animal_weight_kg', weight);
  if (age) formData.append('animal_age_years', age);
  if (cowId) formData.append('cow_id_override', cowId);
  formData.append('generate_report', generateReport);
  formData.append('day_index', dayIndex);
  if (sensorJson) formData.append('sensor_json', sensorJson);

  const response = await fetch(`${BASE_URL}/predict`, { method: 'POST', body: formData });
  if (!response.ok) {
    const err = await response.json().catch(() => ({}));
    throw new Error(err.detail || `Server error: ${response.status}`);
  }
  return await response.json();
};

/**
 * Analyze a cropped region of an image.
 */
export const analyzeCrop = async (image, x, y, width, height, description = "") => {
  const formData = new FormData();
  formData.append('image', image);
  formData.append('x', x);
  formData.append('y', y);
  formData.append('width', width);
  formData.append('height', height);
  formData.append('description', description);

  const response = await fetch(`${BASE_URL}/predict/crop`, { method: 'POST', body: formData });
  if (!response.ok) throw new Error("Crop analysis failed");
  return await response.json();
};

/**
 * Chat with the AI agent.
 */
export const agentChat = async (message, cowId = null, diseaseContext = "", imageB64 = null) => {
  const response = await fetch(`${BASE_URL}/agent/chat`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ 
      message, 
      cow_id: cowId, 
      disease_context: diseaseContext,
      image_b64: imageB64 
    }),
  });
  if (!response.ok) throw new Error("Agent chat failed");
  return await response.json();
};

/**
 * Get cow profile and history.
 */
export const getCowProfile = async (cowId) => {
  const response = await fetch(`${BASE_URL}/agent/cow/${cowId}`);
  if (!response.ok) throw new Error("Failed to get cow profile");
  return await response.json();
};

/**
 * Get daily farm report.
 */
export const getDailyReport = async (dayIndex) => {
  const response = await fetch(`${BASE_URL}/agent/daily-report/${dayIndex}`);
  if (!response.ok) throw new Error("Failed to get daily report");
  return await response.json();
};

/**
 * List all known cows.
 */
export const listCows = async () => {
  const response = await fetch(`${BASE_URL}/agent/cows`);
  if (!response.ok) throw new Error("Failed to list cows");
  return await response.json();
};

/**
 * Clinical Q&A with RAG.
 */
export const clinicalQA = async (question, context = "") => {
  const response = await fetch(`${BASE_URL}/report/qa`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ question, disease_context: context }),
  });
  if (!response.ok) throw new Error("Agent Q&A failed");
  return await response.json();
};

/**
 * BovineIQ Chat
 */
export const bovineIqChat = async (message, history = []) => {
  const response = await fetch(`${BASE_URL}/bovine_iq/chat`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ message, history }),
  });
  if (!response.ok) throw new Error("BovineIQ chat failed");
  return await response.json();
};

/**
 * Health check.
 */
export const checkHealth = async () => {
  try {
    const response = await fetch(`${BASE_URL}/health`);
    return response.ok;
  } catch { return false; }
};
