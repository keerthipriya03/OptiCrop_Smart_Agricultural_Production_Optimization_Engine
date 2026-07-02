import os
import joblib
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report



# Import classifiers

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans


def train_and_evaluate():
    # 1. Load dataset
    dataset_path = 'data/Crop_recommendation.csv'
    if not os.path.exists(dataset_path):
        print(f"Dataset not found at {dataset_path}! Please download the dataset first.")
        return
    
    df = pd.read_csv(dataset_path)




# 2. Preprocess Data: check and handle missing values
    for col in df.columns[:-1]:
        df[col] = df[col].fillna(df[col].median())




# 3. Split features (X) and labels (y)
    X = df.drop(columns=['label'])
    y = df['label']



    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)




# 4. Standardize Features to support algorithm convergence (Logistic/KNN)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)




# 5. Define classification models
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1500, random_state=42),
        "K-Nearest Neighbors": KNeighborsClassifier(n_neighbors=5),
        "Decision Tree": DecisionTreeClassifier(random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42)
    }




    best_model_name = ""
    best_accuracy = 0.0
    best_model = None




    print("\n=== MODEL COMPARISON ===")
    for name, model in models.items():
        # Train model using scaled data
        model.fit(X_train_scaled, y_train)
        predictions = model.predict(X_test_scaled)
        accuracy = accuracy_score(y_test, predictions)
        print(f"{name} Test Accuracy: {accuracy * 100:.2f}%")
        
        # Save best performing
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_model_name = name
            best_model = model




    # Unsupervised K-Means clustering check
        kmeans = KMeans(n_clusters=5, random_state=42, n_init=10)
        kmeans.fit(X_train_scaled)
        print("\nUnsupervised K-Means clustering completed on scaled features.")




    print(f"\n>> Selected Best Model: {best_model_name} with {best_accuracy * 100:.2f}% accuracy.")




    # Output detailed evaluation report
    best_predictions = best_model.predict(X_test_scaled)
    print("\nDetailed Evaluation Report:")
    print(classification_report(y_test, best_predictions))




    # 6. Save the best model and scaler using joblib
    os.makedirs('models', exist_ok=True)
    joblib.dump(best_model, 'models/crop_model.joblib')
    joblib.dump(scaler, 'models/scaler.joblib')
    print("Model and Scaler successfully saved inside models/ folder!")


if __name__ == '__main__':
    train_and_evaluate()