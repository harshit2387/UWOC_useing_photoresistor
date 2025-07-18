import smbus
import time
import numpy as np
import matplotlib.pyplot as plt

# ADS1115 I2C setup
ADS1115_ADDRESS = 0x48
ADS1115_REG_CONVERSION = 0x00
ADS1115_REG_CONFIG = 0x01

# Reference voltage and resolution
V_REF = 3.3
RESOLUTION = 32768  # Signed 16-bit range: -32768 to +32767
THRESHOLD = 4410    # ADC threshold for bit detection

# Initialize I2C bus
bus = smbus.SMBus(1)

def read_ads1115(channel):
    mux = {0: 0x4000, 1: 0x5000, 2: 0x6000, 3: 0x7000}
    config = 0x8000 | mux[channel] | 0x0200 | 0x0100 | 0x0080 | 0x0003
    config_high = (config >> 8) & 0xFF
    config_low = config & 0xFF

    bus.write_i2c_block_data(ADS1115_ADDRESS, ADS1115_REG_CONFIG, [config_high, config_low])
    time.sleep(0.01)

    result = bus.read_i2c_block_data(ADS1115_ADDRESS, ADS1115_REG_CONVERSION, 2)
    raw_adc = (result[0] << 8) | result[1]

    if raw_adc > 0x7FFF:
        raw_adc -= 0x10000

    return raw_adc

# Initialize empty lists
adc_values = []
voltage_values = []
bit_values = []
scaled_voltage_values = []
normalized_adc_values = []

try:
    print(f"{'ADC':<6} {'Voltage (V)':<10} {'Bit':<3} {'Scaled V':<10} {'Norm ADC':<10}")
    while True:
        adc = read_ads1115(0)
        voltage = adc * V_REF / RESOLUTION
        bit = 1 if adc > THRESHOLD else 0
        scaled_voltage = voltage * (5.0 / V_REF)
        normalized_adc = (adc + RESOLUTION) / (2 * RESOLUTION)

        adc_values.append(adc)
        voltage_values.append(voltage)
        bit_values.append(bit)
        scaled_voltage_values.append(scaled_voltage)
        normalized_adc_values.append(normalized_adc)

        print(f"{adc:<6} {voltage:<10.4f} {bit:<3} {scaled_voltage:<10.4f} {normalized_adc:<10.4f}")
        time.sleep(0.5)

except KeyboardInterrupt:
    adc_array = np.array(adc_values)
    voltage_array = np.array(voltage_values)
    bit_array = np.array(bit_values).flatten()
    scaled_voltage_array = np.array(scaled_voltage_values)
    normalized_adc_array = np.array(normalized_adc_values)

    print("\n🛑 Logging stopped.")
    print(f"\nTotal samples: {len(adc_array)}")
    print(f"Scaled Voltage - Min: {np.min(scaled_voltage_array):.4f} V, Max: {np.max(scaled_voltage_array):.4f} V")
    print(f"Normalized ADC - Min: {np.min(normalized_adc_array):.4f}, Max: {np.max(normalized_adc_array):.4f}")

    # Save arrays to CSV
    np.savetxt("adc_array.csv", adc_array, delimiter=",", fmt="%d")
    np.savetxt("voltage_array.csv", voltage_array, delimiter=",", fmt="%.4f")
    np.savetxt("bit_array.csv", bit_array, delimiter=",", fmt="%d")
    np.savetxt("scaled_voltage.csv", scaled_voltage_array, delimiter=",", fmt="%.4f")
    np.savetxt("normalized_adc.csv", normalized_adc_array, delimiter=",", fmt="%.4f")

    # Plot all data
    plt.figure(figsize=(12, 8))

    plt.subplot(4, 1, 1)
    plt.plot(voltage_array, label="Raw Voltage (3.3V ref)", color="blue")
    plt.ylabel("Voltage (V)")
    plt.grid(True)
    plt.legend()

    plt.subplot(4, 1, 2)
    plt.plot(scaled_voltage_array, label="Scaled Voltage (0–5V)", color="green")
    plt.ylabel("Scaled Voltage (V)")
    plt.grid(True)
    plt.legend()

    plt.subplot(4, 1, 3)
    plt.plot(normalized_adc_array, label="Normalized ADC (0–1)", color="purple")
    plt.ylabel("Normalized ADC")
    plt.grid(True)
    plt.legend()

    plt.subplot(4, 1, 4)
    plt.step(range(len(bit_array)), bit_array, label="Bit Stream", color="red", where="mid")
    plt.ylabel("Bit")
    plt.xlabel("Sample Index")
    plt.grid(True)
    plt.legend()

    plt.tight_layout()
    plt.show()
