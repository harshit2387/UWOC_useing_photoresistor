import pandas as pd
import matplotlib.pyplot as plt

# Load the CSV file
file_path = 'C:\\Users\\sharm\\OneDrive\\Desktop\\val\\5-6NTU\\5-6NTU.csv'
df = pd.read_csv(file_path)

# Clean column names
df.columns = df.columns.str.strip()

# Drop rows with missing values in required columns
df = df.dropna(subset=['S No', 'Analog'])

# Take only the first 80 rows
df = df.head(80)

# Apply threshold: 1 if Analog > 0.49, else 0
df['Binary Output'] = (df['Analog'] > 0.49).astype(int)

# Plotting: show analog voltage on Y-axis, color-coded by binary classification
plt.figure(figsize=(12, 6))
plt.scatter(df['S No'], df['Analog'],
            c=df['Binary Output'].map({0: 'darkblue', 1: 'lightskyblue'}),
            label='Analog Voltage (Thresholded)')

# Add threshold line
plt.axhline(y=0.49, color='gray', linestyle='--', linewidth=1, label='Threshold = 0.49')

# Customize the plot
plt.title('Analog Voltage with Binary Classification (Threshold = 0.49)')
plt.xlabel('Serial Number')
plt.ylabel('Analog Voltage (V)')
plt.ylim(0, 1)  # Fix Y-axis range from 0 to 1
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()