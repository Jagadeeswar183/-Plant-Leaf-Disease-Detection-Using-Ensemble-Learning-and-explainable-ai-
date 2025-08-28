🌿 Plant Leaf Disease Detection
Using Ensemble Learning & Explainable AI


📌 Overview

This project focuses on automated detection of plant leaf diseases using Ensemble Learning techniques and Explainable AI (XAI).
The model analyzes leaf images, predicts the disease type with high accuracy, and uses LIME/SHAP to explain why the prediction was made, helping farmers and researchers make informed decisions.

✨ Key Features

✅ Detects multiple plant leaf diseases with high accuracy
✅ Uses ensemble learning to combine CNN + XGBoost + Random Forest
✅ Integrates Explainable AI (XAI) for transparent predictions
✅ Supports visual insights using Grad-CAM / LIME / SHAP
✅ Scalable for real-time disease detection via webcam or mobile app

🛠️ Tech Stack

Programming Language: Python 🐍

ML/DL Frameworks: TensorFlow, PyTorch, Scikit-learn, XGBoost

Explainable AI: LIME, SHAP, Grad-CAM

Image Processing: OpenCV, PIL

Visualization: Matplotlib, Seaborn, Plotly

Deployment (optional): Streamlit / Flask

📂 Project Structure
Plant-Leaf-Disease-Detection/
├── data/               # Dataset files
├── models/             # Trained models
├── notebooks/          # Jupyter notebooks for training & testing
├── src/                # Source code (data processing, training, prediction)
├── explainability/     # LIME & SHAP implementations
├── app/                # Streamlit / Flask app (optional)
├── requirements.txt    # Required dependencies
└── README.md           # Project documentation

⚙️ Installation
# Clone the repository
git clone https://github.com/Jagadeeswar183/Plant-Leaf-Disease-Detection.git

# Navigate to the project folder
cd Plant-Leaf-Disease-Detection

# Install dependencies
pip install -r requirements.txt

🚀 Usage
For Model Training
python src/train_model.py

For Prediction
python src/predict.py --image path_to_leaf_image.jpg

For Web App (if using Streamlit)
streamlit run app/app.py

📊 Model Performance
Model	Accuracy	Precision	Recall	F1-Score
CNN	93%	91%	92%	91%
Random Forest	88%	87%	86%	86%
XGBoost	90%	89%	88%	89%
Ensemble Model	96%	95%	96%	95%
🔍 Explainable AI

To improve trust and interpretability, we integrated:

LIME → Explains predictions by highlighting influential image regions

SHAP → Shows feature contribution to model outputs

Grad-CAM → Visualizes heatmaps of CNN attention areas

Example Visualization: (Sample Placeholder)


📌 Future Enhancements

🚀 Add real-time detection using smartphone camera
🌐 Deploy model on cloud / IoT devices for farmers
📈 Improve dataset with more plant species & diseases
🧠 Use Vision Transformers (ViT) for better accuracy
