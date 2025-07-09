import pandas as pd
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import filedialog, messagebox

def process_and_plot(file_path):
    try:
        # Load and clean data
        df = pd.read_csv(file_path)
        df.columns = df.columns.str.strip()
        df = df.dropna(subset=['S No', 'Analog'])
        df = df.head(400) # Limit to first 400 rows
        df['Binary Output'] = (df['Analog'] > 0.49).astype(int)

        # Plotting
        plt.figure(figsize=(12, 6))
        plt.scatter(df['S No'], df['Analog'],
                    c=df['Binary Output'].map({0: 'darkblue', 1: 'lightskyblue'}),
                    label='Analog Voltage (Thresholded)')
        plt.axhline(y=0.49, color='gray', linestyle='--', linewidth=1, label='Threshold = 0.49')
        plt.title('Analog Voltage with Binary Classification (Threshold = 0.49)')
        plt.xlabel('Serial Number')
        plt.ylabel('Analog Voltage (V)')
        plt.ylim(0, 1)
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.show()
    except Exception as e:
        messagebox.showerror("Error", f"Failed to process file:\n{e}")

def upload_file():
    file_path = filedialog.askopenfilename(
        filetypes=[("CSV files", "*.csv")],
        title="Select a CSV file"
    )
    if file_path:
        process_and_plot(file_path)

# GUI setup
root = tk.Tk()
root.title("CSV Analog Voltage Plotter")
root.geometry("400x200")

label = tk.Label(root, text="Upload a CSV file to visualize analog voltage", font=("Arial", 12))
label.pack(pady=20)

upload_btn = tk.Button(root, text="Upload CSV", command=upload_file, font=("Arial", 12), bg="#4CAF50", fg="white")
upload_btn.pack(pady=10)

root.mainloop()