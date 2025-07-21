import smbus
import time
import numpy as np
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import ttk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# === ADC Setup ===
ADS1115_ADDRESS = 0x48
ADS1115_REG_CONVERSION = 0x00
ADS1115_REG_CONFIG = 0x01

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
class ThresholdPredictorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("ML Threshold Predictor (No scikit-learn)")
        self.root.geometry("1000x700")

        self.adc_data = []
        self.bit_data = []

        ttk.Button(root, text="Start Sampling", command=self.start_sampling).pack(pady=5)
        ttk.Button(root, text="Predict Threshold", command=self.predict_threshold).pack(pady=5)

        self.status = ttk.Label(root, text="Status: Ready")
        self.status.pack(pady=5)

        self.plot_frame = ttk.Frame(root)
        self.plot_frame.pack(fill=tk.BOTH, expand=True)

    def start_sampling(self):
        self.adc_data.clear()
        self.bit_data.clear()
        self.status.config(text="⏳ Sampling ADC values...")

        # Simulate labeled bit stream (0/1) for training
        for _ in range(100):
            val = read_ads1115()
            self.adc_data.append(val)
            # Simulated label: let's assume midpoint of signal range
            self.bit_data.append(1 if val > 4400 else 0)
            time.sleep(0.05)

        self.status.config(text="✅ Sampling Complete")

    def predict_threshold(self):
        self.status.config(text="🧠 Predicting threshold...")

        adc_array = np.array(self.adc_data)
        label_array = np.array(self.bit_data)

        min_val = int(np.min(adc_array))
        max_val = int(np.max(adc_array))
        candidate_thresholds = np.arange(min_val, max_val, 10)

        best_threshold = min_val
        lowest_error = float('inf')

        for threshold in candidate_thresholds:
            pred = (adc_array > threshold).astype(int)
            error = np.sum(pred != label_array)
            if error < lowest_error:
                lowest_error = error
                best_threshold = threshold

        self.status.config(text=f"✅ Best Threshold: {best_threshold} (Errors: {lowest_error})")

        self.plot_bit_stream(best_threshold)

    def plot_bit_stream(self, threshold):
        for widget in self.plot_frame.winfo_children():
            widget.destroy()

        fig, ax = plt.subplots(figsize=(10, 5))
        x = np.arange(len(self.adc_data))
        actual_bits = np.array(self.bit_data)
        predicted_bits = (np.array(self.adc_data) > threshold).astype(int)

        ax.step(x, actual_bits, where='mid', label="Actual Bit Stream", color='green')
        ax.step(x, predicted_bits, where='mid', label=f"Predicted (Threshold={threshold})", linestyle='--', color='blue')
        error_idx = np.where(predicted_bits != actual_bits)[0]
        ax.plot(error_idx, actual_bits[error_idx], 'rx', label="Error Points")

        ax.set_title("Bit Stream Comparison (Manual ML Threshold)")
        ax.set_xlabel("Sample Index")
        ax.set_ylabel("Bit Value")
        ax.legend()
        ax.grid(True)

        canvas = FigureCanvasTkAgg(fig, master=self.plot_frame)
        canvas.draw()
        canvas.get_tk_widget().pack()

# === Run GUI ===
if __name__ == "__main__":
    root = tk.Tk()
    app = ThresholdPredictorApp(root)
    root.mainloop()
