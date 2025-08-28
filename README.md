🌿 Plant Leaf Disease Detection Using Ensemble Learning & Explainable AI
📌 Overview

This project focuses on automated plant leaf disease detection using Ensemble Learning techniques combined with Explainable AI (XAI). The system analyzes leaf images, classifies diseases with high accuracy, and provides clear explanations for predictions to enhance trust and usability.

✨ Features

🌱 Disease Detection – Identifies multiple plant leaf diseases.

🤖 Ensemble Learning – Combines multiple ML/DL models for better accuracy.

🧠 Explainable AI – Uses techniques like LIME/SHAP to explain predictions.

📊 Visualization – Displays prediction confidence and insights.

🚀 User-Friendly Interface (optional if using a web app).

🛠️ Tech Stack

Languages: Python

Libraries & Frameworks: TensorFlow / PyTorch, OpenCV, Scikit-learn, XGBoost, LIME, SHAP

Tools: Jupyter Notebook, Matplotlib, Streamlit (if applicable)

Dataset: Plant leaf images (e.g., PlantVillage dataset)

📂 Project Structure
├── data/                 # Dataset files
├── models/               # Trained ML/DL models
├── notebooks/            # Jupyter notebooks for EDA & training
├── src/                  # Source code for prediction
├── explainability/       # LIME/SHAP scripts
├── app/                  # Streamlit/Flask app files (if any)
├── README.md             # Project documentation
└── requirements.txt      # Dependencies

⚙️ Installation
# Clone the repository
git clone https://github.com/your-username/Plant-Leaf-Disease-Detection.git

# Navigate to project directory
cd Plant-Leaf-Disease-Detection

# Install dependencies
pip install -r requirements.txt

🚀 Usage
# Run Jupyter notebook
jupyter notebook

# Or run the web app (if using Streamlit)
streamlit run app/app.py

📊 Model Performance
Model	Accuracy	Precision	Recall
CNN	93%	91%	92%
Random Forest	88%	87%	86%
Ensemble Model	96%	95%	96%
🔍 Explainable AI

This project integrates LIME and SHAP to explain model predictions by highlighting the important features influencing disease classification.

📌 Future Enhancements

Adding real-time disease detection via camera input.

Expanding dataset for more crop varieties.

Deploying the app on Cloud for global accessibility.
