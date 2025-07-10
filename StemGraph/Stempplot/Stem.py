import tkinter as tk
from tkinter import filedialog, messagebox
import pandas as pd
import matplotlib.pyplot as plt

def plot_stem():
    # Open file dialog to select CSV
    file_path = filedialog.askopenfilename(filetypes=[("CSV Files", "*.csv")])
    if not file_path:
        return

    try:
        # Read CSV file
        df = pd.read_csv(file_path)

        # Check required columns
        required_cols = ['S No', 'Analog']
        if not all(col in df.columns for col in required_cols):
            messagebox.showerror("Missing Columns", "CSV must contain 'S No' and 'Analog Val'")
            return

        # Extract data
        s_no = df['S No'].values
        analog_val = df['Analog'].values

        # Create stem plot
        plt.figure(figsize=(10, 5))
        markerline, stemlines, baseline = plt.stem(s_no, analog_val)
        plt.setp(markerline, marker='o', markersize=5, color='blue')
        plt.setp(stemlines, linestyle='-', color='gray')
        plt.setp(baseline, linestyle='--', color='black')

        # Customize plot
        plt.title("Stem Plot: Analog vs S No")
        plt.xlabel("S No")
        plt.ylabel("Analog")
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.tight_layout()
        plt.show()

    except Exception as e:
        messagebox.showerror("Error", f"An error occurred:\n{str(e)}")

# === GUI setup ===
window = tk.Tk()
window.title("Analog Stem Plot Viewer")
window.geometry("400x200")

label = tk.Label(window, text="Upload CSV with 'S No' & 'Analog'", font=("Arial", 14))
label.pack(pady=20)

btn = tk.Button(window, text="Upload & Plot", command=plot_stem,
                font=("Arial", 12), bg="navy", fg="white")
btn.pack(pady=10)

window.mainloop()