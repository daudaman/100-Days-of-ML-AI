# Day 8 - SVM Classifier

Classifying tumors as benign or malignant using Support Vector Machine with an RBF kernel. SVM works by finding the optimal hyperplane that separates classes with the maximum margin.

## Dataset

2000 tumor samples with 9 cell nucleus measurements: radius, texture, perimeter, area, smoothness, compactness, concavity, symmetry, and fractal dimension.

## How SVM works

SVM finds the hyperplane that separates classes with the largest possible margin. The RBF kernel maps data into higher dimensions to handle non-linear separation. The key parameters are C (controls margin width vs misclassification penalty) and gamma (controls how far the influence of a single training sample reaches).

## Kernel comparison

| Kernel | CV Accuracy |
|--------|-------------|
| Linear | 89.1% |
| RBF    | 87.6% |
| Poly   | 86.3% |

Linear kernel actually performed best here — useful reminder that linear models win when the data is roughly linearly separable.

## Results

Test Accuracy: 88% (RBF kernel)

| Class     | Precision | Recall | F1   |
|-----------|-----------|--------|------|
| Benign    | 0.90      | 0.91   | 0.90 |
| Malignant | 0.84      | 0.82   | 0.83 |

## How to run

```bash
pip install -r requirements.txt
python train.py
python app.py
```

Then open http://localhost:5000

## Project structure

```
Day-008-SVM-Classifier/
├── train.py
├── app.py
├── notebook.ipynb
├── requirements.txt
├── model.pkl
├── scaler.pkl
├── feature_cols.pkl
├── dataset/
│   └── cancer.csv
└── screenshots/
    ├── distribution.png
    ├── confusion_matrix.png
    ├── correlation.png
    └── feature_distributions.png
```

## Stack

Python, scikit-learn, pandas, matplotlib, seaborn, Flask
