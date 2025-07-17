import smbus
import time
import csv

# ADS1115 I2C address
ADS1115_ADDRESS = 0x48

# Register addresses
ADS1115_REG_CONVERSION = 0x00
ADS1115_REG_CONFIG     = 0x01

# Initialize I2C bus
bus = smbus.SMBus(1)

def read_ads1115(channel):
    if channel < 0 or channel > 3:
        raise ValueError("Channel must be 0-3")

    mux = {
        0: 0x4000,
        1: 0x5000,
        2: 0x6000,
        3: 0x7000
    }

    config = 0x8000 | mux[channel] | 0x0200 | 0x0100 | 0x0080 | 0x0003
    config_high = (config >> 8) & 0xFF
    config_low  = config & 0xFF

    bus.write_i2c_block_data(ADS1115_ADDRESS, ADS1115_REG_CONFIG, [config_high, config_low])
    time.sleep(0.01)

    result = bus.read_i2c_block_data(ADS1115_ADDRESS, ADS1115_REG_CONVERSION, 2)
    raw_adc = (result[0] << 8) | result[1]
    if raw_adc > 0x7FFF:
        raw_adc -= 0x10000

    voltage = raw_adc * 4.096 / 32768.0
    return raw_adc, voltage

# CSV setup
adc_file = open("adc_values.csv", mode="w", newline="")
voltage_file = open("voltage_values.csv", mode="w", newline="")
adc_writer = csv.writer(adc_file)
voltage_writer = csv.writer(voltage_file)

adc_writer.writerow(["Timestamp", "ADC Value"])
voltage_writer.writerow(["Timestamp", "Voltage (V)"])

# Main loop
try:
    print(f"{'Timestamp':<20} {'ADC Value':<10} {'Voltage (V)':<10}")
    while True:
        adc_value, voltage = read_ads1115(0)
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")

        # Print to terminal
        print(f"{timestamp:<20} {adc_value:<10} {voltage:<10.4f}")

        # Write to CSV files
        adc_writer.writerow([timestamp, adc_value])
        voltage_writer.writerow([timestamp, f"{voltage:.4f}"])

        time.sleep(0.5)

except KeyboardInterrupt:
    adc_file.close()
    voltage_file.close()
    print("\n🛑 Data logging stopped. Files saved.")
