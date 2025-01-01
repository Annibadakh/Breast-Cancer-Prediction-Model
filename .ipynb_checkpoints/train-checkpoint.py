from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import pickle
import pandas as pd


# During training
data = pd.read_csv('breast cancer.csv')

# Drop irrelevant columns
data = data.drop(['id', 'Unnamed: 32'], axis=1)  # Drop 'id' and unnecessary columns


# Load dataset
data = load_breast_cancer()
X = data['data']
y = data['target']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Save model
with open('model.pkl', 'wb') as file:
    pickle.dump(model, file)

print("Model trained and saved as model.pkl")
