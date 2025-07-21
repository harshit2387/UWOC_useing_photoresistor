import smbus
import time
import numpy as np
import tkinter as tk
from tkinter import ttk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# === ADC Configuration ===
ADS1115_ADDRESS = 0x48
ADS1115_REG_CONVERSION = 0x00
ADS1115_REG_CONFIG = 0x01
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
class BasicThresholdApp:
    def __init__(self, root):
        self.root = root
        self.root.title("ADC Bit Stream Analyzer")
        self.root.geometry("1000x700")

        self.adc_data = []
        self.bit_data = []

        self.plot_frame = ttk.Frame(root)
        self.plot_frame.pack(fill=tk.BOTH, expand=True)

        ttk.Label(root, text="Threshold:").pack()
        self.threshold_entry = ttk.Entry(root)
        self.threshold_entry.insert(0, str(DEFAULT_THRESHOLD))
        self.threshold_entry.pack(pady=5)

        ttk.Button(root, text="Start Sampling", command=self.start_sampling).pack(pady=5)
        ttk.Button(root, text="Generate Bit Stream", command=self.show_plot).pack(pady=5)

        self.status = ttk.Label(root, text="Status: Ready")
        self.status.pack(pady=5)

    def start_sampling(self):
        self.adc_data.clear()
        self.bit_data.clear()
        self.status.config(text="⏳ Sampling ADC values...")
        try:
            threshold = int(self.threshold_entry.get())
        except:
            self.status.config(text="❗ Invalid threshold")
            return

        for _ in range(100):
            val = read_ads1115()
            self.adc_data.append(val)
            self.bit_data.append(1 if val > threshold else 0)
            time.sleep(0.05)

        self.status.config(text="✅ Sampling Complete")

    def show_plot(self):
        for widget in self.plot_frame.winfo_children():
            widget.destroy()

        fig, ax = plt.subplots(figsize=(10, 5))
        x = np.arange(len(self.adc_data))

        actual_bits = np.array(self.bit_data)
        ax.step(x, actual_bits, where='mid', label="Bit Stream", color='green')
        ax.set_title("Bit Stream Visualization")
        ax.set_xlabel("Sample Index")
        ax.set_ylabel("Bit Value (0/1)")
        ax.grid(True)
        ax.legend()

        canvas = FigureCanvasTkAgg(fig, master=self.plot_frame)
        canvas.draw()
        canvas.get_tk_widget().pack()

# === Launch App ===
if __name__ == "__main__":
    root = tk.Tk()
    app = BasicThresholdApp(root)
    root.mainloop()
