import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import f1_score, make_scorer
from tkinter import filedialog, messagebox, Tk, Button, Label, Text, Scrollbar, RIGHT, Y, END, BOTH, Frame

MAX_ROWS = 7000

def process_file(filepath, log_widget):
    try:
        df = pd.read_csv(filepath)
        df = df.iloc[:min(len(df), MAX_ROWS)].copy()
        df.columns = [col.strip().lower().replace("lables", "labels") for col in df.columns]

        label_cols = [col for col in df.columns if col.startswith("labels")]
        feature_cols = [col for col in df.columns if col.startswith("feature")]

        if len(feature_cols) == 0:
            raise ValueError("No feature columns found.")

        labels = df[label_cols].values
        features = df[feature_cols].values
        X = StandardScaler().fit_transform(features)

        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        scorer = make_scorer(f1_score, average='macro')

        model_results = []
        prediction_dfs = {}

        def log(text): log_widget.insert(END, f"{text}\n"); log_widget.update_idletasks()

        def evaluate_model(model, name):
            log(f"⏳ Evaluating: {name}")
            scores = []
            for train_idx, test_idx in cv.split(X, labels[:, 0]):
                model.fit(X[train_idx], labels[train_idx])
                preds = model.predict(X[test_idx])
                scores.append(f1_score(labels[test_idx], preds, average='macro'))
            avg = np.mean(scores)
            model_results.append((name, avg))

            preds_full = model.predict(X)
            df_out = df.copy()
            df_out['bit'] = preds_full[:, 0]
            prediction_dfs[name] = df_out
            log(f"✅ {name}: F1 = {avg:.4f}")

        # Models
        evaluate_model(MultiOutputClassifier(LogisticRegression(max_iter=1000)), "Logistic Regression")
        evaluate_model(MultiOutputClassifier(SVC(kernel='linear')), "SVM (Linear)")

        tree_params = {'max_depth': range(1, 10)}
        tree_search = GridSearchCV(DecisionTreeClassifier(), tree_params, scoring=scorer, cv=cv)
        tree_model = MultiOutputClassifier(tree_search)
        tree_model.fit(X, labels)
        score_tree = f1_score(labels, tree_model.predict(X), average='macro')
        model_results.append(("Decision Tree (Grid Search)", score_tree))
        df_tree = df.copy()
        df_tree['bit'] = tree_model.predict(X)[:, 0]
        prediction_dfs["Decision Tree (Grid Search)"] = df_tree
        log(f"✅ Decision Tree: F1 = {score_tree:.4f}")

        knn_params = {'n_neighbors': [1, 3, 5], 'weights': ['uniform', 'distance']}
        knn_search = GridSearchCV(KNeighborsClassifier(), knn_params, scoring=scorer, cv=cv)
        knn_model = MultiOutputClassifier(knn_search)
        knn_model.fit(X, labels)
        score_knn = f1_score(labels, knn_model.predict(X), average='macro')
        model_results.append(("KNN (Grid Search)", score_knn))
        df_knn = df.copy()
        df_knn['bit'] = knn_model.predict(X)[:, 0]
        prediction_dfs["KNN (Grid Search)"] = df_knn
        log(f"✅ KNN: F1 = {score_knn:.4f}")

        out_dir = os.path.join(os.path.dirname(filepath), "classified_outputs")
        os.makedirs(out_dir, exist_ok=True)
        for name, df_out in prediction_dfs.items():
            safe_name = name.lower().replace(" ", "_").replace("(", "").replace(")", "")
            df_out.to_csv(os.path.join(out_dir, f"{os.path.basename(filepath).split('.')[0]}_{safe_name}.csv"), index=False)

        return os.path.basename(filepath), model_results

    except Exception as e:
        log_widget.insert(END, f"❌ Error in {os.path.basename(filepath)}: {str(e)}\n")
        log_widget.update_idletasks()
        return None, []

def show_bar_chart(results):
    models = [name for name, score in results]
    scores = [score for name, score in results]
    plt.figure(figsize=(8, 5))
    bars = plt.barh(models, scores, color="#4CAF50")
    plt.xlabel("F1 Score")
    plt.title("Model Performance")
    plt.xlim(0, 1.05)
    for bar, score in zip(bars, scores):
        plt.text(score + 0.01, bar.get_y() + bar.get_height() / 2, f"{score:.4f}", va='center')
    plt.tight_layout()
    plt.show()

def browse_files(log_widget):
    filepaths = filedialog.askopenfilenames(filetypes=[("CSV Files", "*.csv")])
    if not filepaths:
        return

    all_scores = []
    log_widget.delete(1.0, END)
    for path in filepaths:
        fname, results = process_file(path, log_widget)
        if fname:
            all_scores.extend(results)
            log_widget.insert(END, f"📁 Completed: {fname}\n\n")

    if all_scores:
        show_bar_chart(all_scores)

# GUI Setup
root = Tk()
root.title("📊 Batch Bit Classifier with Visualization")
Label(root, text="Upload CSV files for multi-label F1 scoring", font=("Arial", 12)).pack(pady=5)

Button(root, text="📂 Upload & Process", command=lambda: browse_files(log_box), font=("Arial", 11), bg="#2196F3", fg="white").pack(pady=5)

frame = Frame(root)
frame.pack(fill=BOTH, expand=True, padx=10, pady=10)

scrollbar = Scrollbar(frame)
scrollbar.pack(side=RIGHT, fill=Y)

log_box = Text(frame, wrap='word', yscrollcommand=scrollbar.set, font=("Consolas", 10))
log_box.pack(fill=BOTH, expand=True)
scrollbar.config(command=log_box.yview)

root.geometry("720x480")
root.mainloop()