from flask import Flask, request, render_template
import pickle
import numpy as np

app = Flask(__name__)

model         = pickle.load(open('model.pkl',        'rb'))
feature_names = pickle.load(open('feature_names.pkl', 'rb'))

FIELD_META = {
    'age':      {'label': 'Age',                     'type': 'number', 'min': 1,   'max': 120, 'step': 1,   'placeholder': 'e.g. 52'},
    'sex':      {'label': 'Sex (1=Male, 0=Female)',  'type': 'number', 'min': 0,   'max': 1,   'step': 1,   'placeholder': '0 or 1'},
    'cp':       {'label': 'Chest Pain Type (0-3)',   'type': 'number', 'min': 0,   'max': 3,   'step': 1,   'placeholder': '0-3'},
    'trestbps': {'label': 'Resting Blood Pressure',  'type': 'number', 'min': 50,  'max': 250, 'step': 1,   'placeholder': 'e.g. 130'},
    'chol':     {'label': 'Serum Cholesterol (mg/dl)','type': 'number','min': 100, 'max': 600, 'step': 1,   'placeholder': 'e.g. 250'},
    'fbs':      {'label': 'Fasting Blood Sugar >120 (1=True)', 'type': 'number', 'min': 0, 'max': 1, 'step': 1, 'placeholder': '0 or 1'},
    'restecg':  {'label': 'Resting ECG Results (0-2)','type': 'number','min': 0,  'max': 2,   'step': 1,   'placeholder': '0-2'},
    'thalach':  {'label': 'Max Heart Rate Achieved', 'type': 'number', 'min': 50,  'max': 250, 'step': 1,   'placeholder': 'e.g. 150'},
    'exang':    {'label': 'Exercise Induced Angina (1=Yes)', 'type': 'number', 'min': 0, 'max': 1, 'step': 1, 'placeholder': '0 or 1'},
    'oldpeak':  {'label': 'ST Depression (Oldpeak)', 'type': 'number', 'min': 0,   'max': 10,  'step': 0.1, 'placeholder': 'e.g. 1.5'},
    'slope':    {'label': 'Slope of Peak ST (0-2)', 'type': 'number',  'min': 0,   'max': 2,   'step': 1,   'placeholder': '0-2'},
    'ca':       {'label': 'Major Vessels Colored (0-4)', 'type': 'number', 'min': 0, 'max': 4, 'step': 1,  'placeholder': '0-4'},
    'thal':     {'label': 'Thalassemia (0-3)',       'type': 'number', 'min': 0,   'max': 3,   'step': 1,   'placeholder': '0-3'},
}


@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    error      = None
    values     = {}

    if request.method == 'POST':
        try:
            values = {f: request.form.get(f, '') for f in feature_names}
            features = [float(values[f]) for f in feature_names]
            arr    = np.array(features).reshape(1, -1)
            result = model.predict(arr)[0]
            proba  = model.predict_proba(arr)[0]
            prediction = {
                'label':       'Heart Disease Detected ⚠️' if result == 1 else 'No Heart Disease ✅',
                'class':       'danger' if result == 1 else 'success',
                'confidence':  f"{max(proba)*100:.1f}%",
            }
        except Exception as e:
            error = f"Error: {e}"

    return render_template('index.html',
                           fields=feature_names,
                           meta=FIELD_META,
                           prediction=prediction,
                           error=error,
                           values=values)


if __name__ == '__main__':
    app.run(debug=True)
