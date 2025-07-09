import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import GridSearchCV, KFold
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler

# ==== ML Logic ====
def train_and_evaluate(file_path, output_text):
    try:
        df = pd.read_csv(file_path)
        df.dropna(subset=['Binary Sent', 'Analog Value'], inplace=True)
        X = df[['Analog Value']].values
        y = df['Binary Sent'].values

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        kf = KFold(n_splits=5, shuffle=True, random_state=42)

        models = {
            'SVM': {
                'model': SVC(),
                'params': {
                    'C': [0.1, 1, 10],
                    'kernel': ['linear', 'rbf'],
                    'gamma': ['scale', 'auto']
                }
            },
            'KNN': {
                'model': KNeighborsClassifier(),
                'params': {
                    'n_neighbors': [3, 5, 7],
                    'weights': ['uniform', 'distance'],
                    'metric': ['euclidean', 'manhattan']
                }
            },
            'Naive Bayes': {
                'model': GaussianNB(),
                'params': {}
            },
            'Decision Tree': {
                'model': DecisionTreeClassifier(),
                'params': {
                    'max_depth': [3, 5, 10, None],
                    'criterion': ['gini', 'entropy'],
                    'min_samples_split': [2, 5, 10]
                }
            }
        }

        best_models = {}

        output_text.delete(1.0, tk.END)
        output_text.insert(tk.END, "Training models...\n")

        for name, mp in models.items():
            grid = GridSearchCV(mp['model'], mp['params'], cv=kf, scoring='accuracy')
            grid.fit(X_scaled, y)
            best_models[name] = grid.best_estimator_
            output_text.insert(tk.END, f"\n{name}:\n")
            output_text.insert(tk.END, f"  Best Accuracy: {grid.best_score_:.4f}\n")
            output_text.insert(tk.END, f"  Best Params: {grid.best_params_}\n")

        output_text.insert(tk.END, "\nFinal Evaluation:\n")

        for name, model in best_models.items():
            y_pred = model.predict(X_scaled)
            acc = accuracy_score(y, y_pred)
            report = classification_report(y, y_pred)
            cm = confusion_matrix(y, y_pred)

            output_text.insert(tk.END, f"\n{name} Accuracy: {acc:.4f}\n")
            output_text.insert(tk.END, f"{report}\n")

            plt.figure(figsize=(4, 3))
            ax = sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
            for spine in ax.spines.values():
                spine.set_visible(False)
            plt.title(f'{name} - Confusion Matrix')
            plt.xlabel('Predicted')
            plt.ylabel('Actual')
            plt.tight_layout()
            plt.show()

    except Exception as e:
        messagebox.showerror("Error", f"An error occurred:\n{e}")

# ==== GUI Setup ====
def upload_and_run():
    file_path = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv")])
    if file_path:
        train_and_evaluate(file_path, output_text)

root = tk.Tk()
root.title("ML Model Trainer")
root.geometry("700x600")

frame = tk.Frame(root)
frame.pack(pady=10)

label = tk.Label(frame, text="Upload CSV with 'Analog Value' and 'Binary Sent' columns", font=("Arial", 12))
label.pack(pady=10)

upload_btn = tk.Button(frame, text="Upload and Train", command=upload_and_run, bg="#4CAF50", fg="white", font=("Arial", 12))
upload_btn.pack(pady=5)

output_text = tk.Text(root, wrap=tk.WORD, width=80, height=25, font=("Courier", 10))
output_text.pack(padx=10, pady=10)

root.mainloop()