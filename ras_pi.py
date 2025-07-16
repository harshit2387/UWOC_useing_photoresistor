import pandas as pd
import numpy as np
import time
import os
import spidev
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tkinter import Tk, Button, Label, Text, Scrollbar, Frame, RIGHT, Y, END, BOTH

# Constants
CHANNEL = 0
DELAY = 0.1
SAMPLE_COUNT = 1000
DATA_PATH = "/home/pi/photodiode_data.csv"

# SPI setup for MCP3008
spi = spidev.SpiDev()
spi.open(0, 0)
spi.max_speed_hz = 1350000

def read_channel(channel):
    adc = spi.xfer2([1, (8 + channel) << 4, 0])
    value = ((adc[1] & 3) << 8) + adc[2]
    return value

def log(widget, msg):
    widget.insert(END, msg + '\n')
    widget.update_idletasks()

def collect_data(log_box):
    data = []
    log(log_box, "📡 Collecting real-time photodiode data...")
    for i in range(SAMPLE_COUNT):
        analog_value = read_channel(CHANNEL)
        label = 1 if analog_value > 500 else 0  # Basic label (can be changed)
        data.append([analog_value, label])
        log(log_box, f"Sample {i+1}/{SAMPLE_COUNT}: {analog_value} => Label {label}")
        time.sleep(DELAY)
    
    df = pd.DataFrame(data, columns=["analog_value", "label"])
    df.to_csv(DATA_PATH, index=False)
    log(log_box, f"✅ Data saved to {DATA_PATH}")

def train_model(log_box):
    try:
        log(log_box, "🧠 Training model on collected data...")
        df = pd.read_csv(DATA_PATH)
        X = df[["analog_value"]].values
        y = df["label"].values

        X = StandardScaler().fit_transform(X)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        model = LogisticRegression()
        model.fit(X_train, y_train)

        preds = model.predict(X_test)
        score = f1_score(y_test, preds, average="macro")

        log(log_box, f"✅ Model trained! F1 Score: {score:.4f}")
        plot_result(X_test, y_test, preds)

    except Exception as e:
        log(log_box, f"❌ Error: {str(e)}")

def plot_result(X, y_true, y_pred):
    plt.figure(figsize=(8, 5))
    plt.scatter(X, y_true, color="green", label="True", alpha=0.6)
    plt.scatter(X, y_pred, color="red", label="Predicted", alpha=0.6)
    plt.title("Prediction vs Ground Truth")
    plt.xlabel("Standardized Analog Value")
    plt.ylabel("Label")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# GUI Setup
root = Tk()
root.title("📡 Photodiode Real-Time Classifier (Raspberry Pi)")

Label(root, text="Photodiode Classifier via MCP3008 SPI", font=("Arial", 12)).pack(pady=5)

Button(root, text="🟢 Collect Data", command=lambda: collect_data(log_box), font=("Arial", 11), bg="#4CAF50", fg="white").pack(pady=5)
Button(root, text="🧠 Train Model", command=lambda: train_model(log_box), font=("Arial", 11), bg="#2196F3", fg="white").pack(pady=5)

frame = Frame(root)
frame.pack(fill=BOTH, expand=True, padx=10, pady=10)

scrollbar = Scrollbar(frame)
scrollbar.pack(side=RIGHT, fill=Y)

log_box = Text(frame, wrap='word', yscrollcommand=scrollbar.set, font=("Consolas", 10))
log_box.pack(fill=BOTH, expand=True)
scrollbar.config(command=log_box.yview)

root.geometry("720x480")
root.mainloop()
