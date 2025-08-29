import pandas as pd
import joblib
import tkinter as tk
from tkinter import messagebox

MODEL_FILE = "wine_quality_model.pkl"

def load_model():
    try:
        return joblib.load(MODEL_FILE)
    except FileNotFoundError:
        messagebox.showerror("Error")
        return None

def clean_float(entry):
    try:
        return float(entry.get().strip().replace(',', '.'))
    except Exception as e:
        print(f"[ERROR] Invalid input: {entry.get()} -> {e}")
        raise ValueError("Please enter valid numeric values.")

def predict_quality():
    try:
        input_data = {
            'fixed acidity': clean_float(entry_fixed_acidity),
            'volatile acidity': clean_float(entry_va),
            'citric acid': clean_float(entry_ca),
            'residual sugar': clean_float(entry_sugar),
            'chlorides': clean_float(entry_chlorides),
            'free sulfur dioxide': clean_float(entry_fsd),
            'total sulfur dioxide': clean_float(entry_tsd),
            'density': clean_float(entry_density),
            'pH': clean_float(entry_ph),
            'sulphates': clean_float(entry_sulphates),
            'alcohol': clean_float(entry_alcohol)
        }

        model = load_model()
        if model:
            df = pd.DataFrame([input_data])[model.feature_names_in_]
            prediction = model.predict(df)[0]
            messagebox.showinfo("Prediction Result", f"Predicted Wine Quality: {prediction}")
    except ValueError as e:
        messagebox.showerror("Input Error", str(e))

root = tk.Tk()
root.title("Wine Quality Predictor")

fields = [
    ("Fixed Acidity", "entry_fixed_acidity"),
    ("Volatile Acidity", "entry_va"),
    ("Citric Acid", "entry_ca"),
    ("Residual Sugar", "entry_sugar"),
    ("Chlorides", "entry_chlorides"),
    ("Free SO₂", "entry_fsd"),
    ("Total SO₂", "entry_tsd"),
    ("Density", "entry_density"),
    ("pH", "entry_ph"),
    ("Sulphates", "entry_sulphates"),
    ("Alcohol", "entry_alcohol"),
]

defaults = {
    'entry_fixed_acidity': 7.4,
    'entry_va': 0.4,
    'entry_ca': 0.3,
    'entry_sugar': 2.1,
    'entry_chlorides': 0.05,
    'entry_fsd': 28,
    'entry_tsd': 105,
    'entry_density': 0.9940,
    'entry_ph': 3.3,
    'entry_sulphates': 0.65,
    'entry_alcohol': 10.5
}

entries = {}

for i, (label, varname) in enumerate(fields):
    tk.Label(root, text=label).grid(row=i, column=0, padx=10, pady=5, sticky="w")
    entry = tk.Entry(root, width=20)
    entry.grid(row=i, column=1, padx=10, pady=5)
    entry.insert(0, str(defaults.get(varname, "")))
    entries[varname] = entry


entry_fixed_acidity = entries["entry_fixed_acidity"]
entry_va = entries["entry_va"]
entry_ca = entries["entry_ca"]
entry_sugar = entries["entry_sugar"]
entry_chlorides = entries["entry_chlorides"]
entry_fsd = entries["entry_fsd"]
entry_tsd = entries["entry_tsd"]
entry_density = entries["entry_density"]
entry_ph = entries["entry_ph"]
entry_sulphates = entries["entry_sulphates"]
entry_alcohol = entries["entry_alcohol"]

tk.Button(root, text="Predict Quality", command=predict_quality,
          bg="#4CAF50", fg="white").grid(row=len(fields), column=0, columnspan=2, pady=10)

root.mainloop()
