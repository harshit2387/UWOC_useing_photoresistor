import RPi.GPIO as GPIO
import time

# Set up GPIO pin
tx_pin = 25
GPIO.setmode(GPIO.BCM)
GPIO.setup(tx_pin, GPIO.OUT)

# Define bit timing for 115200 bps
baud_rate = 115200
bit_time = 1 / baud_rate

# Build message manually: start + data + checksum
def calc_checksum(data_byte):
    return 0xFF - data_byte

def build_message(data_byte):
    start_byte = 0xAA
    checksum = calc_checksum(data_byte)
    return [start_byte, data_byte, checksum]

def send_byte(byte_val):
    for i in range(8):
        bit = (byte_val >> i) & 0x01
        GPIO.output(tx_pin, bit)
        time.sleep(bit_time)

try:
    while True:
        message = build_message(0x55)  # 'U'
        for byte in message:
            send_byte(byte)
        time.sleep(0.001)
except KeyboardInterrupt:
    GPIO.cleanup()
