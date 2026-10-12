export const predict = async (file: File) => {
  const formData = new FormData();
  formData.append('image', file);
  try {
    const res = await fetch('http://localhost:8000/predict', {
      method: 'POST',
      body: formData,
    });
    return res.json();
  } catch (e) {
    console.error(e);
    return null;
  }
};

export const getHealth = async () => {
  try {
    const res = await fetch('http://localhost:8000/health');
    return res.json();
  } catch (e) {
    return { status: 'offline' };
  }
};

export const getModelInfo = async () => {
  try {
    const res = await fetch('http://localhost:8000/model-info');
    return res.json();
  } catch (e) {
    return null;
  }
};
