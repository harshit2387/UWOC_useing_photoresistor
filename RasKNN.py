import smbus
import time
import numpy as np
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import ttk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import csv
import os

# === Constants ===
ADS1115_ADDRESS = 0x48
ADS1115_REG_CONVERSION = 0x00
ADS1115_REG_CONFIG = 0x01
CSV_FILENAME = "adc_ml_results.csv"
NUM_SAMPLES = 200

bus = smbus.SMBus(1)

# === ADC Sampling ===
def read_ads1115(channel=0):
    mux = {0: 0x4000, 1: 0x5000, 2: 0x6000, 3: 0x7000}
    config = 0x8000 | mux[channel] | 0x0200 | 0x0100 | 0x0080 | 0x0003
    bus.write_i2c_block_data(ADS1115_ADDRESS, ADS1115_REG_CONFIG,
                             [(config >> 8) & 0xFF, config & 0xFF])
    time.sleep(0.01)
    result = bus.read_i2c_block_data(ADS1115_ADDRESS, ADS1115_REG_CONVERSION, 2)
    raw = (result[0] << 8) | result[1]
    if raw > 0x7FFF:
        raw -= 0x10000
    return raw

# === ML Metrics ===
def calculate_metrics(actual, predicted):
    actual = np.array(actual)
    predicted = np.array(predicted)
    TP = np.sum((actual == 1) & (predicted == 1))
    TN = np.sum((actual == 0) & (predicted == 0))
    FP = np.sum((actual == 0) & (predicted == 1))
    FN = np.sum((actual == 1) & (predicted == 0))

    precision = TP / (TP + FP) if (TP + FP) else 0
    recall = TP / (TP + FN) if (TP + FN) else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0

    return {
        "TP": TP, "TN": TN, "FP": FP, "FN": FN,
        "Precision": round(precision, 3),
        "Recall": round(recall, 3),
        "F1 Score": round(f1, 3)
    }

# === GUI Class ===
class BitstreamApp:
    def __init__(self, root):
        self.root = root
        self.root.title("ADC ML Bitstream Analyzer (No sklearn)")
        self.root.geometry("1000x740")

        self.adc_data = []
        self.bit_data = []
        self.predictions = {}
        self.metrics = {}

        # === GUI Layout ===
        ttk.Button(root, text="Sample ADC", command=self.sample_adc).pack(pady=5)
        ttk.Button(root, text="Threshold Classifier", command=self.threshold_model).pack(pady=5)
        ttk.Button(root, text="KNN Classifier", command=self.knn_model).pack(pady=5)
        ttk.Button(root, text="SVM Classifier", command=self.svm_model).pack(pady=5)
        ttk.Button(root, text="Decision Tree", command=self.decision_tree_model).pack(pady=5)
        ttk.Button(root, text="Export CSV", command=self.export_csv).pack(pady=5)

        self.status = ttk.Label(root, text="Status: Ready")
        self.status.pack(pady=5)

        self.plot_frame = ttk.Frame(root)
        self.plot_frame.pack(fill=tk.BOTH, expand=True)

    def sample_adc(self):
        self.adc_data.clear()
        self.bit_data.clear()
        self.predictions.clear()
        self.metrics.clear()
        self.status.config(text="Sampling ADC...")

        for i in range(NUM_SAMPLES):
            val = read_ads1115()
            self.adc_data.append(val)
            self.bit_data.append(i % 2)  # Simulated 'U' → 010101...
            time.sleep(0.01)

        self.status.config(text="✅ ADC Sampling Complete")

    def threshold_model(self):
        X = np.array(self.adc_data)
        y = np.array(self.bit_data)
        best_thresh = int(np.median(X))
        pred = (X > best_thresh).astype(int).tolist()
        self.predictions["Threshold"] = pred
        self.metrics["Threshold"] = calculate_metrics(y, pred)
        self.status.config(text=f"Threshold → F1: {self.metrics['Threshold']['F1 Score']}")
        self.plot_results(pred, "Threshold")

    def knn_model(self, k=3):
        X = np.array(self.adc_data)
        y = np.array(self.bit_data)
        pred = []
        for i in range(len(X)):
            dists = np.abs(X - X[i])
            nn_idx = np.argsort(dists)[1:k+1]
            vote = int(np.round(np.mean(y[nn_idx])))
            pred.append(vote)
        self.predictions["KNN"] = pred
        self.metrics["KNN"] = calculate_metrics(y, pred)
        self.status.config(text=f"KNN → F1: {self.metrics['KNN']['F1 Score']}")
        self.plot_results(pred, "KNN")

    def svm_model(self):
        X = np.array(self.adc_data)
        y = np.array(self.bit_data)
        X0 = X[y == 0]
        X1 = X[y == 1]
        mean0 = np.mean(X0)
        mean1 = np.mean(X1)
        boundary = (mean0 + mean1) / 2
        pred = (X > boundary).astype(int).tolist()
        self.predictions["SVM"] = pred
        self.metrics["SVM"] = calculate_metrics(y, pred)
        self.status.config(text=f"SVM → F1: {self.metrics['SVM']['F1 Score']}")
        self.plot_results(pred, "SVM")

    def decision_tree_model(self):
        X = np.array(self.adc_data)
        y = np.array(self.bit_data)
        mean_val = np.mean(X)
        left_label = int(np.round(np.mean(y[X <= mean_val])))
        right_label = int(np.round(np.mean(y[X > mean_val])))
        pred = [left_label if x <= mean_val else right_label for x in X]
        self.predictions["Decision Tree"] = pred
        self.metrics["Decision Tree"] = calculate_metrics(y, pred)
        self.status.config(text=f"Tree → F1: {self.metrics['Decision Tree']['F1 Score']}")
        self.plot_results(pred, "Decision Tree")

    def plot_results(self, prediction, label):
        for widget in self.plot_frame.winfo_children():
            widget.destroy()

        actual = np.array(self.bit_data)
        pred = np.array(prediction)
        x = np.arange(len(actual))
        error_idx = np.where(actual != pred)[0]

        fig, ax = plt.subplots(figsize=(10, 5))
        ax.step(x, actual, where='mid', label="Actual", color='green')
        ax.step(x, pred, where='mid', label=f"{label} Prediction", linestyle='--', color='blue')
        ax.plot(error_idx, actual[error_idx], 'rx', label="Errors")

        ax.set_title(f"Bit Stream Comparison - {label}")
        ax.set_xlabel("Index")
        ax.set_ylabel("Bit (0/1)")
        ax.legend()
        ax.grid(True)

        canvas = FigureCanvasTkAgg(fig, master=self.plot_frame)
        canvas.draw()
        canvas.get_tk_widget().pack()

    def export_csv(self):
        if not self.adc_data:
            self.status.config(text="⚠️ Sample ADC first")
            return

        headers = ["Index", "ADC", "Actual Bit"] + [f"{k} Pred" for k in self.predictions]
        rows = [headers]
        for i in range(NUM_SAMPLES):
            row = [i, self.adc_data[i], self.bit_data[i]]
            for k in self.predictions:
                row.append(self.predictions[k][i])
            rows.append(row)

        with open(CSV_FILENAME, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerows(rows)

            # Write metrics
            for k in self.metrics:
                writer.writerow([])
                writer.writerow([f"{k} Metrics"])
                for key, val in self.metrics[k].items():
                    writer.writerow([key, val])

        self.status.config(text=f"📄 Exported to {os.path.abspath(CSV_FILENAME)}")

# === Launch GUI ===
if __name__ == "__main__":
    root = tk.Tk()
    app = BitstreamApp(root)
    root.mainloop()
