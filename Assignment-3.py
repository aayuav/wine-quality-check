import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import joblib

def train_model(csv_file):
    df = pd.read_csv(csv_file)

    X = df.drop(columns='quality')
    y = df['quality']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestClassifier()
    model.fit(X_train, y_train)

    joblib.dump(model, 'wine_quality_model.pkl')
    print("Model trained and saved as wine_quality_model.pkl")

    y_pred = model.predict(X_test)
    print("\nEvaluation:")
    print(classification_report(y_test, y_pred))

    # Generate and plot confusion matrix heatmap
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=sorted(y.unique()), yticklabels=sorted(y.unique()))
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Matrix Heatmap')
    plt.tight_layout()
    plt.show()

# Run the function with your CSV
if __name__ == "__main__":
    train_model("wine_data.csv")
