const API_BASE_URL = 'http://localhost:5000';

export const predictScore = async (matchData) => {
  try {
    const response = await fetch(`${API_BASE_URL}/predict/score`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(matchData),
    });
    
    if (!response.ok) {
      const errorData = await response.json();
      throw new Error(errorData.error || `HTTP error! status: ${response.status}`);
    }
    
    const data = await response.json();
    
    // Only return the predicted score if successful
    if (data.success) {
      return data.predicted_score;
    } else {
      throw new Error(data.error || 'Prediction failed');
    }
  } catch (error) {
    console.error('Prediction API Error:', error);
    throw error;
  }
};