from flask import Flask, render_template, request, url_for
import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import os
from sklearn.datasets import load_breast_cancer


# Load model
model = pickle.load(open('model.pkl', 'rb'))

# Flask app
app = Flask(__name__)

# Define a folder for storing images
VISUALS_FOLDER = 'static/visuals'
os.makedirs(VISUALS_FOLDER, exist_ok=True)


@app.route('/')
def home():
    # Visualize the data and save the plot
    visualize_data()

    # Provide the path to the saved visualization
    img_url = url_for('static', filename='visuals/data_plot.png')
    return render_template('index.html', img_url=img_url)


@app.route('/predict', methods=['POST'])
def predict():
    features = request.form['feature']
    features = features.split(',')
    np_features = np.asarray(features, dtype=np.float32)

    # Prediction
    pred = model.predict(np_features.reshape(1, -1))
    message = ['Cancerous' if pred[0] == 1 else 'Not Cancerous']
    return render_template('index.html', message=message)


def visualize_data():
    # Load the dataset
    data = pd.DataFrame(load_breast_cancer().data, columns=load_breast_cancer().feature_names)
    target = load_breast_cancer().target
    data['target'] = target

    # Create a simple plot (e.g., histogram of a feature)
    plt.figure(figsize=(8, 6))
    data['mean radius'].hist(bins=30, alpha=0.7, color='blue')
    plt.title('Distribution of Mean Radius')
    plt.xlabel('Mean Radius')
    plt.ylabel('Frequency')

    # Save the plot
    plt.tight_layout()
    plt.savefig(os.path.join(VISUALS_FOLDER, 'data_plot.png'))
    plt.close()


if __name__ == '__main__':
    app.run(debug=True)
