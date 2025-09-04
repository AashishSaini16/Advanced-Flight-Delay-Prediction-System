# Advanced Flight Delay Prediction System

This repository contains a Jupyter notebook for developing a neural network model to predict flight delays using historical aviation data. The project uses embeddings for categorical features, SHAP for interpretability, and an interactive widget for real-time predictions, demonstrating preprocessing, model training, evaluation, and deployment-ready insights.

### Dataset Source

The dataset is the "2022 US domestic flight data for delay prediction" from Kaggle (https://www.kaggle.com/datasets/ahmedzoarob/2022-us-domestic-flight-data-for-delay-prediction), sourced from the U.S. Bureau of Transportation Statistics' On-Time Performance data. It features over 7 million records of U.S. domestic flights in 2022. Download link: [flights.zip](https://www.dropbox.com/scl/fi/dw8adpyocge43xnczeobj/flights.zip?rlkey=jnzty4kcip6ka2z9ujly1yerv&dl=1). It includes categorical (e.g., airline, origin/destination airports, routes) and numerical features (e.g., departure times, distances, taxi outs). Preprocessed for class imbalance and split into train/validation/test sets.

### Key Features

- Data Preprocessing: Embed categorical variables, normalize numericals, engineer cyclical features (sin/cos for time/month).
- Model Development: Feedforward neural network with Keras/TensorFlow, combining embeddings and dense layers for binary classification (delay >15 mins).
- Training & Evaluation: Adam optimizer, BCE loss, metrics like AUC (0.97), precision-recall curves, confusion matrix.
- Interactive Prediction: Jupyter widget for inputting flight details and getting delay probability/label.
- Interpretability: SHAP values to explain predictions, highlighting top features like departure delays and seasonal effects.

### Technologies Used

- Python: Core scripting and data handling.
- TensorFlow & Keras: Model building and training.
- Pandas & NumPy: Data manipulation and preprocessing.
- Scikit-learn: Scaling, splitting, and metrics.
- SHAP: Model explainability.
- Matplotlib & Seaborn: Visualizations (curves, matrices, SHAP plots).
- IPyWidgets: Interactive prediction interface.

### Project Workflow

- Data Loading: Import and preprocess flight dataset.
- Feature Engineering: Handle categoricals with embeddings, scale numericals.
- Model Building: Define architecture with concatenation, batch norm, dropout.

![Model Architecture](https://raw.githubusercontent.com/AashishSaini16/Advanced-Flight-Delay-Prediction-System/main/Model%20Architecture.png)

- Training: Fit model with early stopping, plot loss/accuracy/AUC curves.

![Training Curves](https://raw.githubusercontent.com/AashishSaini16/Advanced-Flight-Delay-Prediction-System/main/Training%20Curves.PNG)

- Evaluation: Compute test metrics, generate confusion matrix, PR/ROC curves.

![Confusion Matrix](https://raw.githubusercontent.com/AashishSaini16/Advanced-Flight-Delay-Prediction-System/main/Confusion%20Matrix.PNG)  
![Precision-Recall Curve](https://raw.githubusercontent.com/AashishSaini16/Advanced-Flight-Delay-Prediction-System/main/Precision-Recall%20Curve.PNG)  
![ROC Curve](https://raw.githubusercontent.com/AashishSaini16/Advanced-Flight-Delay-Prediction-System/main/ROC%20Curve.PNG)

- Prediction: Use widget for user inputs and output delay forecasts.
- Interpretability: Apply SHAP to visualize feature impacts.

![SHAP Values](https://raw.githubusercontent.com/AashishSaini16/Advanced-Flight-Delay-Prediction-System/main/SHAP%20Values.PNG)
