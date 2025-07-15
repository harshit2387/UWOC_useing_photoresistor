import pandas as pd
import numpy as np
import os
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import f1_score, make_scorer
from tkinter import filedialog, messagebox, Tk, Button, Label

MAX_ROWS = 7000

def process_file(filepath):
    try:
        df = pd.read_csv(filepath)

        # Limit to top rows
        df = df.iloc[:min(len(df), MAX_ROWS)].copy()

        # Auto-fix and clean column names
        df.columns = [col.strip().lower().replace("lables", "labels") for col in df.columns]

        # Select label and feature columns
        label_cols = [col for col in df.columns if col.startswith("labels")]
        feature_cols = [col for col in df.columns if col.startswith("feature")]

        if len(feature_cols) == 0:
            raise ValueError("No feature columns found. Check CSV column names.")

        labels = df[label_cols].values
        features = df[feature_cols].values

        # Normalize features
        scaler = StandardScaler()
        X = scaler.fit_transform(features)

        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        scorer = make_scorer(f1_score, average='macro')

        model_results = []
        prediction_dfs = {}

        def evaluate_model(model, name):
            scores = []
            for train_idx, test_idx in cv.split(X, labels[:, 0]):
                model.fit(X[train_idx], labels[train_idx])
                preds = model.predict(X[test_idx])
                score = f1_score(labels[test_idx], preds, average='macro')
                scores.append(score)
            avg = np.mean(scores)
            model_results.append((name, avg))

            preds_full = model.predict(X)
            df_out = df.copy()
            df_out['bit'] = preds_full[:, 0]
            prediction_dfs[name] = df_out

        # Logistic Regression
        log_model = MultiOutputClassifier(LogisticRegression(max_iter=1000))
        evaluate_model(log_model, "Logistic Regression")

        # SVM
        svm_model = MultiOutputClassifier(SVC(kernel='linear'))
        evaluate_model(svm_model, "SVM (Linear)")

        # Decision Tree
        tree_params = {'max_depth': range(1, 10)}
        tree_search = GridSearchCV(DecisionTreeClassifier(), tree_params, scoring=scorer, cv=cv)
        tree_model = MultiOutputClassifier(tree_search)
        tree_model.fit(X, labels)
        preds_tree = tree_model.predict(X)
        score_tree = f1_score(labels, preds_tree, average='macro')
        model_results.append(("Decision Tree (Grid Search)", score_tree))
        df_tree = df.copy()
        df_tree['bit'] = preds_tree[:, 0]
        prediction_dfs["Decision Tree (Grid Search)"] = df_tree

        # KNN
        knn_params = {'n_neighbors': [1, 3, 5], 'weights': ['uniform', 'distance']}
        knn_search = GridSearchCV(KNeighborsClassifier(), knn_params, scoring=scorer, cv=cv)
        knn_model = MultiOutputClassifier(knn_search)
        knn_model.fit(X, labels)
        preds_knn = knn_model.predict(X)
        score_knn = f1_score(labels, preds_knn, average='macro')
        model_results.append(("KNN (Grid Search)", score_knn))
        df_knn = df.copy()
        df_knn['bit'] = preds_knn[:, 0]
        prediction_dfs["KNN (Grid Search)"] = df_knn

        # Save results
        out_dir = os.path.join(os.path.dirname(filepath), "classified_outputs")
        os.makedirs(out_dir, exist_ok=True)
        for name, df_out in prediction_dfs.items():
            safe_name = name.lower().replace(" ", "_").replace("(", "").replace(")", "")
            out_path = os.path.join(out_dir, f"{os.path.basename(filepath).split('.')[0]}_{safe_name}.csv")
            df_out.to_csv(out_path, index=False)

        # Return results summary
        summary = f"✅ Results for: {os.path.basename(filepath)}\n"
        for name, score in model_results:
            summary += f"🔹 {name}: F1 = {score:.4f}\n"
            if score == 1.0:
                summary += f"🎯 Perfect score achieved!\n"
        return summary

    except Exception as e:
        return f"❌ Error in {os.path.basename(filepath)}:\n{str(e)}"

def browse_files():
    filepaths = filedialog.askopenfilenames(filetypes=[("CSV Files", "*.csv")])
    if not filepaths:
        return

    full_report = ""
    for path in filepaths:
        full_report += process_file(path) + "\n"

    messagebox.showinfo("Batch Classification Summary", full_report)

# GUI Setup
root = Tk()
root.title("Batch Bit Classifier 🧠📂")
Label(root, text="Select one or more CSV files for scoring and export", font=("Arial", 12)).pack(pady=10)
Button(root, text="📁 Upload Files", command=browse_files, font=("Arial", 12), bg="#3f51b5", fg="white").pack(pady=10)
root.geometry("580x160")
root.mainloop()