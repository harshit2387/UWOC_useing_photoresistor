import serial
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from collections import deque
import time

# === CONFIGURATION ===
PORT = 'COM12'       # Replace with your Arduino port
BAUD_RATE = 9600
MAX_POINTS = 100     # Number of points to display

# === SETUP ===
ser = serial.Serial(PORT, BAUD_RATE)
analog_values = deque([0]*MAX_POINTS, maxlen=MAX_POINTS)
timestamps = deque([0]*MAX_POINTS, maxlen=MAX_POINTS)
start_time = time.time()

# === PLOT SETUP ===
fig, ax = plt.subplots()
line, = ax.plot([], [], lw=2)
ax.set_ylim(0, 200)
ax.set_title("Analog Value vs Time")
ax.set_xlabel("Time (s)")
ax.set_ylabel("Analog Value")

# === UPDATE FUNCTION ===
def update(frame):
    try:
        raw = ser.readline().decode('utf-8', errors='ignore').strip()
        if raw and raw.count(',') >= 4:
            parts = raw.split(',')
            analog_value = int(parts[2])
            current_time = time.time() - start_time

            analog_values.append(analog_value)
            timestamps.append(current_time)

            line.set_data(timestamps, analog_values)
            ax.set_xlim(timestamps[0], timestamps[-1])
    except:
        pass
    return line,

# === ANIMATE ===
ani = animation.FuncAnimation(fig, update, interval=50)
plt.tight_layout()
plt.show()
