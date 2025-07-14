import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import f1_score
from tkinter import filedialog, messagebox, Tk, Button, Label

def process_file(filepath):
    try:
        df = pd.read_csv(filepath)
        labels = df['labels'].values.astype(int)
        features = df['feature'].values.astype(float)
        X = features.reshape(-1, 1)

        model_results = []

        # 1️⃣ Threshold-Based Classifier
        best_f1_thresh = 0
        best_thresh = None
        thresholds = np.linspace(min(features), max(features), num=100000)

        for t in thresholds:
            preds = (features >= t).astype(int)
            f1 = f1_score(labels, preds)
            if f1 > best_f1_thresh:
                best_f1_thresh = f1
                best_thresh = t
            if f1 == 1.0:
                break

        df_thresh = df.copy()
        df_thresh['bit'] = (features >= best_thresh).astype(int)
        df_thresh.to_csv('threshold_output.csv', index=False)
        model_results.append(("Threshold", best_f1_thresh))

        # 2️⃣ Logistic Regression
        log_model = LogisticRegression()
        log_model.fit(X, labels)
        preds_log = log_model.predict(X)
        f1_log = f1_score(labels, preds_log)

        df_log = df.copy()
        df_log['bit'] = preds_log
        df_log.to_csv('logistic_output.csv', index=False)
        model_results.append(("Logistic Regression", f1_log))

        # 3️⃣ Support Vector Machine
        svm_model = SVC(kernel='linear')
        svm_model.fit(X, labels)
        preds_svm = svm_model.predict(X)
        f1_svm = f1_score(labels, preds_svm)

        df_svm = df.copy()
        df_svm['bit'] = preds_svm
        df_svm.to_csv('svm_output.csv', index=False)
        model_results.append(("SVM (Linear)", f1_svm))

        # 4️⃣ Decision Tree
        tree_model = DecisionTreeClassifier(max_depth=None)
        tree_model.fit(X, labels)
        preds_tree = tree_model.predict(X)
        f1_tree = f1_score(labels, preds_tree)

        df_tree = df.copy()
        df_tree['bit'] = preds_tree
        df_tree.to_csv('tree_output.csv', index=False)
        model_results.append(("Decision Tree", f1_tree))

        # 5️⃣ K-Nearest Neighbors
        knn_model = KNeighborsClassifier(n_neighbors=1)
        knn_model.fit(X, labels)
        preds_knn = knn_model.predict(X)
        f1_knn = f1_score(labels, preds_knn)

        df_knn = df.copy()
        df_knn['bit'] = preds_knn
        df_knn.to_csv('knn_output.csv', index=False)
        model_results.append(("KNN (n=1)", f1_knn))

        # 📊 Summary Message
        summary = "✅ Models Processed:\n"
        for name, score in model_results:
            summary += f"🔹 {name}: F1 Score = {score:.4f}\n"
            if score == 1.0:
                summary += f"🎯 Perfect classification achieved with {name}!\n"

        summary += "\n📁 All files saved with respective model outputs."

        messagebox.showinfo("Results Summary", summary)

    except Exception as e:
        messagebox.showerror("Error", f"❌ Failed to process file:\n{str(e)}")

def browse_file():
    filepath = filedialog.askopenfilename(filetypes=[("CSV Files", "*.csv")])
    if filepath:
        process_file(filepath)

# 🖼️ GUI Setup
root = Tk()
root.title("F1-Score Maximizer")
Label(root, text="Upload CSV to compare models for bit classification", font=("Arial", 12)).pack(pady=10)
Button(root, text="📂 Upload & Analyze", command=browse_file, font=("Arial", 12), bg="#673AB7", fg="white").pack(pady=10)
root.geometry("460x170")
root.mainloop()