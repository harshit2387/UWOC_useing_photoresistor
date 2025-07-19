import smbus
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
import tkinter as tk
from tkinter import ttk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

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

# === GUI Builder ===
class ADCApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Real-Time ADC Classifier")
        self.root.geometry("1000x700")
        
        self.adc_data = []
        self.label_data = []
        self.frame = ttk.Frame(self.root)
        self.frame.pack(fill=tk.BOTH, expand=True)

        self.threshold_label = ttk.Label(root, text=f"Manual Threshold: {DEFAULT_THRESHOLD}")
        self.threshold_label.pack(pady=5)

        self.threshold_entry = ttk.Entry(root)
        self.threshold_entry.insert(0, str(DEFAULT_THRESHOLD))
        self.threshold_entry.pack(pady=5)

        ttk.Button(root, text="Run Manual Threshold", command=self.run_manual).pack(pady=5)
        ttk.Button(root, text="Run ML Model (KNN)", command=self.run_ml).pack(pady=5)
        ttk.Button(root, text="Start Sampling", command=self.start_sampling).pack(pady=5)

        self.status = ttk.Label(root, text="Status: Waiting")
        self.status.pack(pady=5)

    def start_sampling(self):
        self.adc_data.clear()
        self.label_data.clear()
        try:
            for _ in range(100):  # sample 100 points
                adc = read_ads1115()
                bit = 1 if adc > DEFAULT_THRESHOLD else 0
                self.adc_data.append(adc)
                self.label_data.append(bit)
                time.sleep(0.05)
            self.status.config(text="✅ Data Sampling Complete")
        except Exception as e:
            self.status.config(text=f"⚠️ Error during sampling: {str(e)}")

    def run_manual(self):
        try:
            threshold = int(self.threshold_entry.get())
        except:
            self.status.config(text="❗ Invalid threshold")
            return
        pred = [1 if adc > threshold else 0 for adc in self.adc_data]
        ber = np.mean(np.array(pred) != np.array(self.label_data))
        self.status.config(text=f"🔧 Manual BER: {ber:.3f}")
        self.plot_results(pred, "Manual Threshold")

    def run_ml(self):
        adc_array = np.array(self.adc_data).reshape(-1, 1)
        labels = np.array(self.label_data)
        model = KNeighborsClassifier(n_neighbors=3)
        model.fit(adc_array, labels)
        pred = model.predict(adc_array)
        ber = np.mean(pred != labels)
        self.status.config(text=f"🧠 ML BER (KNN): {ber:.3f}")
        self.plot_results(pred, "ML Model (KNN)")

    def plot_results(self, predictions, title):
        for widget in self.frame.winfo_children():
            widget.destroy()
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.step(range(len(self.label_data)), self.label_data, label="Actual Bits", where='mid', color='green')
        ax.step(range(len(predictions)), predictions, label=title, where='mid', linestyle='--', color='blue')
        errors = np.array(predictions) != np.array(self.label_data)
        ax.plot(np.where(errors)[0], np.array(self.label_data)[errors], 'rx', label="Errors")
        ax.set_title(f"{title} - Bit Stream Analysis")
        ax.set_ylabel("Bit (0/1)")
        ax.set_xlabel("Sample Index")
        ax.legend()
        ax.grid(True)
        canvas = FigureCanvasTkAgg(fig, master=self.frame)
        canvas.draw()
        canvas.get_tk_widget().pack()

# === Main Program ===
if __name__ == "__main__":
    root = tk.Tk()
    app = ADCApp(root)
    root.mainloop()
