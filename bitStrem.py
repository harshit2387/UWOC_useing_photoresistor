import smbus
import time
import numpy as np
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import ttk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

# === ADC Configuration ===
ADS1115_ADDRESS = 0x48
ADS1115_REG_CONVERSION = 0x00
ADS1115_REG_CONFIG = 0x01
V_REF = 3.3
RESOLUTION = 32768
DEFAULT_THRESHOLD = 4410

bus = smbus.SMBus(1)

def read_ads1115(channel=0):
    mux = {0: 0x4000, 1: 0x5000, 2: 0x6000, 3: 0x7000}
    config = 0x8000 | mux[channel] | 0x0200 | 0x0100 | 0x0080 | 0x0003
    bus.write_i2c_block_data(ADS1115_ADDRESS, ADS1115_REG_CONFIG, [(config >> 8) & 0xFF, config & 0xFF])
    time.sleep(0.01)
    result = bus.read_i2c_block_data(ADS1115_ADDRESS, ADS1115_REG_CONVERSION, 2)
    raw_adc = (result[0] << 8) | result[1]
    if raw_adc > 0x7FFF:
        raw_adc -= 0x10000
    return raw_adc

# === GUI App ===
class MultiClassifierApp:
    def __init__(self, root):
        self.root = root
        self.root.title("ADC Multi-Classifier BER & Bit Stream Analyzer")
        self.root.geometry("1200x800")

        self.adc_data = []
        self.label_data = []
        self.frame = ttk.Frame(root)
        self.frame.pack(fill=tk.BOTH, expand=True)

        ttk.Label(root, text="Manual Threshold:").pack()
        self.threshold_entry = ttk.Entry(root)
        self.threshold_entry.insert(0, str(DEFAULT_THRESHOLD))
        self.threshold_entry.pack(pady=5)

        ttk.Button(root, text="Start Sampling", command=self.start_sampling).pack(pady=5)
        ttk.Button(root, text="Run All ML Models", command=self.run_all_ml).pack(pady=5)
        ttk.Button(root, text="Run Manual Threshold", command=self.run_manual).pack(pady=5)

        self.status = ttk.Label(root, text="Status: Waiting")
        self.status.pack(pady=5)

    def start_sampling(self):
        self.adc_data.clear()
        self.label_data.clear()
        self.status.config(text="🔄 Sampling 100 values...")
        for _ in range(100):
            adc = read_ads1115()
            bit = 1 if adc > DEFAULT_THRESHOLD else 0
            self.adc_data.append(adc)
            self.label_data.append(bit)
            time.sleep(0.05)
        self.status.config(text="✅ Sampling Complete")

    def run_all_ml(self):
        adc_array = np.array(self.adc_data).reshape(-1, 1)
        label_array = np.array(self.label_data)

        models = {
            "KNN": KNeighborsClassifier(n_neighbors=3),
            "SVM": SVC(),
            "LogReg": LogisticRegression(),
            "DecisionTree": DecisionTreeClassifier(),
            "RandomForest": RandomForestClassifier()
        }

        prediction_dict = {}
        for name, model in models.items():
            model.fit(adc_array, label_array)
            prediction_dict[name] = model.predict(adc_array)

        self.status.config(text="🧠 ML Bit Streams Generated")
        self.plot_bit_streams(prediction_dict)

    def run_manual(self):
        try:
            threshold = int(self.threshold_entry.get())
        except:
            self.status.config(text="❗ Invalid threshold")
            return
        pred = [1 if adc > threshold else 0 for adc in self.adc_data]
        self.status.config(text="📏 Manual Bit Stream Generated")
        self.plot_bit_streams({"Manual Threshold": pred})

    def plot_bit_streams(self, predictions_dict):
        for widget in self.frame.winfo_children():
            widget.destroy()

        fig, axs = plt.subplots(len(predictions_dict), 1, figsize=(10, 4 * len(predictions_dict)))
        if len(predictions_dict) == 1:
            axs = [axs]

        x = np.arange(len(self.label_data))
        actual = np.array(self.label_data)

        for ax, (name, pred) in zip(axs, predictions_dict.items()):
            pred = np.array(pred)
            ax.step(x, actual, where='mid', label="Actual Bit Stream", color='green')
            ax.step(x, pred, where='mid', label=f"Predicted ({name})", linestyle='--', color='blue')
            error_idx = np.where(pred != actual)[0]
            ax.plot(error_idx, actual[error_idx], 'rx', label="Error")
            ax.set_title(f"Bit Stream Comparison: {name}")
            ax.set_ylabel("Bit (0/1)")
            ax.set_xlabel("Sample Index")
            ax.legend()
            ax.grid(True)

        plt.tight_layout()
        canvas = FigureCanvasTkAgg(fig, master=self.frame)
        canvas.draw()
        canvas.get_tk_widget().pack()

# === Start GUI ===
if __name__ == "__main__":
    root = tk.Tk()
    app = MultiClassifierApp(root)
    root.mainloop()
