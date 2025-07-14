import pandas as pd
import numpy as np
from sklearn.metrics import f1_score
from tkinter import filedialog, messagebox, Tk, Button, Label

def process_file(filepath):
    try:
        # Load data
        df = pd.read_csv(filepath)

        labels = df['labels'].values
        features = df['feature'].values

        # Find best threshold
        best_f1 = 0
        best_threshold = None

        for threshold in np.linspace(min(features), max(features), num=1000):
            preds = (features >= threshold).astype(int)
            f1 = f1_score(labels, preds)
            if f1 > best_f1:
                best_f1 = f1
                best_threshold = threshold

        # Add bit column
        df['bit'] = (df['feature'] >= best_threshold).astype(int)
        df.to_csv('threshold_output.csv', index=False)

        # Notify result
        messagebox.showinfo("Done", f"✅ File processed!\nBest Threshold: {best_threshold:.6f}\nF1 Score: {best_f1:.4f}\nSaved as threshold_output.csv")
    except Exception as e:
        messagebox.showerror("Error", f"Failed to process file:\n{str(e)}")

def browse_file():
    filepath = filedialog.askopenfilename(filetypes=[("CSV Files", "*.csv")])
    if filepath:
        process_file(filepath)

# GUI setup
root = Tk()
root.title("Bit Threshold Detector")

Label(root, text="Upload labeled CSV file to optimize bit threshold", font=("Arial", 12)).pack(pady=10)
Button(root, text="📂 Upload & Process", command=browse_file, font=("Arial", 12), bg="#4CAF50", fg="white").pack(pady=10)

root.geometry("400x150")
root.mainloop()