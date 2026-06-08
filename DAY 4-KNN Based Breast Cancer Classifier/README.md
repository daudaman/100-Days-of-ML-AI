# Day 4 - Breast Cancer Classification using KNN

Predicting whether a breast tumor is **Malignant** or **Benign** using the K-Nearest Neighbors (KNN) algorithm. This is a binary classification problem based on tumor measurement features from the Wisconsin Breast Cancer Dataset.

## Dataset

569 breast cancer samples with 8 selected features.

Features used:

- mean radius
- mean texture
- mean perimeter
- mean area
- mean smoothness
- mean compactness
- mean concavity
- mean concave points

Target classes:

- Malignant (0)
- Benign (1)

The dataset is included directly in scikit-learn through:

```python
from sklearn.datasets import load_breast_cancer
```

## Model

- K-Nearest Neighbors (KNN)

The value of K is automatically selected by testing values from 1 to 20 and choosing the value with the highest accuracy.

KNN is a distance-based algorithm, so feature scaling is performed using StandardScaler before training.

## Results

| Metric | Value |
|----------|----------|
| Accuracy | ~96–98% |
| Best K | Auto Selected |
| Classes | Malignant / Benign |

The model performs well because the selected tumor features provide strong separation between malignant and benign cases.

## Visualizations

The training pipeline automatically generates:

- Target Distribution
- Feature Distributions
- Correlation Heatmap
- K vs Accuracy Plot
- Confusion Matrix

## Flask Web App

The project includes an interactive Flask application where users can enter tumor measurements and receive:

- Predicted diagnosis
- Confidence score
- Benign probability
- Malignant probability

## How to run

```bash
pip install -r requirements.txt
python train.py
python app.py
```

Then open:

```text
http://localhost:5000
```

## Project Structure

```text
Day-004-Breast-Cancer-Classification/
│
├── train.py
├── app.py
├── requirements.txt
├── model.pkl
├── scaler.pkl
├── feature_cols.pkl
│
├── dataset/
│   └── cancer.csv
│
└── screenshots/
    ├── target_distribution.png
    ├── feature_distributions.png
    ├── correlation.png
    ├── accuracy_plot.png
    └── confusion_matrix.png
```

## Stack

- Python
- scikit-learn
- pandas
- numpy
- matplotlib
- seaborn
- Flask

## Concepts Covered

- K-Nearest Neighbors (KNN)
- Euclidean Distance
- Feature Scaling
- Train/Test Split
- Hyperparameter Tuning
- Classification Metrics
- Confusion Matrix
- Model Serialization with Pickle
- Flask Deployment
