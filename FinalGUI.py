import smbus
import time
import numpy as np
import tkinter as tk
from tkinter import ttk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import csv
import os

# === Constants ===
ADS1115_ADDRESS = 0x48
ADS1115_REG_CONVERSION = 0x00
ADS1115_REG_CONFIG = 0x01
NUM_SAMPLES = 200
CSV_FILENAME = "bitstream_analysis.csv"

bus = smbus.SMBus(1)

def read_ads1115(channel=0):
    mux = {0: 0x4000, 1: 0x5000, 2: 0x6000, 3: 0x7000}
    config = 0x8000 | mux[channel] | 0x0200 | 0x0100 | 0x0080 | 0x0003
    bus.write_i2c_block_data(ADS1115_ADDRESS, ADS1115_REG_CONFIG,
                             [(config >> 8) & 0xFF, config & 0xFF])
    time.sleep(0.005)
    result = bus.read_i2c_block_data(ADS1115_ADDRESS, ADS1115_REG_CONVERSION, 2)
    raw_adc = (result[0] << 8) | result[1]
    if raw_adc > 0x7FFF:
        raw_adc -= 0x10000
    return raw_adc

def calculate_metrics(actual, predicted):
    actual = np.array(actual)
    predicted = np.array(predicted)
    TP = np.sum((actual == 1) & (predicted == 1))
    TN = np.sum((actual == 0) & (predicted == 0))
    FP = np.sum((actual == 0) & (predicted == 1))
    FN = np.sum((actual == 1) & (predicted == 0))
    total = len(actual)
    error_rate = (FP + FN) / total if total else 0
    precision = TP / (TP + FP) if (TP + FP) else 0
    recall = TP / (TP + FN) if (TP + FN) else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0
    return {
        "TP": TP, "TN": TN, "FP": FP, "FN": FN,
        "Precision": round(precision, 3),
        "Recall": round(recall, 3),
        "F1 Score": round(f1, 3),
        "Bit Error Rate": round(error_rate, 3)
    }

class BitstreamApp:
    def __init__(self, root):
        self.root = root
        self.root.title("ADC Bitstream Analyzer")
        self.root.geometry("1000x800")

        self.adc_data = []
        self.tx_bits = []
        self.predictions = {}
        self.metrics = {}

        # === UI Controls ===
        ttk.Button(root, text="📡 Sample ADC", command=self.sample_adc).pack(pady=5)
        ttk.Button(root, text="Threshold Classifier", command=self.threshold_model).pack(pady=5)

        manual_frame = ttk.Frame(root)
        manual_frame.pack(pady=5)
        ttk.Label(manual_frame, text="Manual Threshold:").pack(side=tk.LEFT)
        self.manual_entry = ttk.Entry(manual_frame, width=10)
        self.manual_entry.pack(side=tk.LEFT)
        ttk.Button(manual_frame, text="Apply Manual", command=self.manual_threshold_model).pack(side=tk.LEFT, padx=5)

        ttk.Button(root, text="KNN Classifier", command=self.knn_model).pack(pady=5)
        ttk.Button(root, text="SVM Classifier", command=self.svm_model).pack(pady=5)
        ttk.Button(root, text="Decision Tree", command=self.decision_tree_model).pack(pady=5)
        ttk.Button(root, text="📁 Export CSV", command=self.export_csv).pack(pady=5)

        self.status = ttk.Label(root, text="Status: Ready")
        self.status.pack(pady=10)

        self.plot_frame = ttk.Frame(root)
        self.plot_frame.pack(fill=tk.BOTH, expand=True)

    def sample_adc(self):
        self.adc_data.clear()
        self.tx_bits.clear()
        self.predictions.clear()
        self.metrics.clear()
        self.status.config(text="⏳ Sampling ADC...")
        for i in range(NUM_SAMPLES):
            val = read_ads1115()
            self.adc_data.append(val)
            self.tx_bits.append(i % 2)  # Simulated bit pattern
        self.status.config(text="✅ Sampling Complete")

    def threshold_model(self):
        X = np.array(self.adc_data)
        y = np.array(self.tx_bits)
        threshold = int(np.median(X))
        pred = (X > threshold).astype(int).tolist()
        label = f"Threshold ({threshold})"
        self.predictions[label] = pred
        self.metrics[label] = calculate_metrics(y, pred)
        self.plot_bits(label)

    def manual_threshold_model(self):
        try:
            threshold = int(self.manual_entry.get())
        except ValueError:
            self.status.config(text="⚠️ Invalid threshold")
            return
        X = np.array(self.adc_data)
        y = np.array(self.tx_bits)
        pred = (X > threshold).astype(int).tolist()
        label = f"Manual ({threshold})"
        self.predictions[label] = pred
        self.metrics[label] = calculate_metrics(y, pred)
        self.plot_bits(label)

    def knn_model(self, k=3):
        X = np.array(self.adc_data)
        y = np.array(self.tx_bits)
        pred = []
        for i in range(len(X)):
            dist = np.abs(X - X[i])
            neighbors = np.argsort(dist)[1:k+1]
            vote = int(np.round(np.mean(y[neighbors])))
            pred.append(vote)
        label = f"KNN (k={k})"
        self.predictions[label] = pred
        self.metrics[label] = calculate_metrics(y, pred)
        self.plot_bits(label)

    def svm_model(self):
        X = np.array(self.adc_data)
        y = np.array(self.tx_bits)
        boundary = (np.mean(X[y == 0]) + np.mean(X[y == 1])) / 2
        pred = (X > boundary).astype(int).tolist()
        label = f"SVM (boundary={int(boundary)})"
        self.predictions[label] = pred
        self.metrics[label] = calculate_metrics(y, pred)
        self.plot_bits(label)

    def decision_tree_model(self):
        X = np.array(self.adc_data)
        y = np.array(self.tx_bits)
        split = np.mean(X)
        left = int(np.round(np.mean(y[X <= split])))
        right = int(np.round(np.mean(y[X > split])))
        pred = [left if x <= split else right for x in X]
        label = f"Tree (split={int(split)})"
        self.predictions[label] = pred
        self.metrics[label] = calculate_metrics(y, pred)
        self.plot_bits(label)

    def plot_bits(self, label):
        for widget in self.plot_frame.winfo_children():
            widget.destroy()
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.step(range(NUM_SAMPLES), self.tx_bits, where='mid', label="Transmitted", color='black')
        ax.step(range(NUM_SAMPLES), self.predictions[label], where='mid', label="Received", linestyle='--', color='blue')
        ax.set_title(f"Bitstream Comparison – {label}")
        ax.set_xlabel("Sample Index")
        ax.set_ylabel("Bit (0/1)")
        ax.legend()
        ax.grid(True)
        canvas = FigureCanvasTkAgg(fig, master=self.plot_frame)
        canvas.draw()
        canvas.get_tk_widget().pack()
        ber = self.metrics[label]["Bit Error Rate"]
        self.status.config(text=f"{label} BER: {ber}")

    def export_csv(self):
        if not self.adc_data:
            self.status.config(text="⚠️ No data to export")
            return
        headers = ["Index", "ADC", "Transmitted Bit"] + list(self.predictions.keys())
        rows = [headers]
        for i in range(NUM_SAMPLES):
            row = [i, self.adc_data[i], self.tx_bits[i]]
            for label in self.predictions:
                row.append(self.predictions[label][i])
            rows.append(row)
        with open(CSV_FILENAME, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerows(rows)
            for label in self.metrics:
                writer.writerow([])
                writer.writerow([f"{label} Metrics"])
                for key, val in self.metrics[label].items():
                    writer.writerow([key, val])
        self.status.config(text=f"📄 Exported to {os.path.abspath(CSV_FILENAME)}")

# === Run App ===
# === Run App ===
if __name__ == "__main__":
    root = tk.Tk()
    app = BitstreamApp(root)
    root.mainloop()
