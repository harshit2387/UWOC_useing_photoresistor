const int laserPin = 9;                   // Digital pin to control laser
const unsigned int bitDuration = 10;    // Milliseconds per bit = 10 bps
const char* message = "HELLO\n";         // Message to send

void setup() {
  pinMode(laserPin, OUTPUT);
  digitalWrite(laserPin, LOW);
  delay(1);  // Give receiver time to get ready
  Serial.begin(9600);
}

void loop() {
  for (int i = 0; message[i] != '\0'; i++) {
    sendCharWithParity(message[i]);
    delay(bitDuration * 2);  // Short pause between characters
  }

  delay(2);  // Wait before repeating the message
}

void sendCharWithParity(char c) {
  Serial.print("Sending: ");
  Serial.println(c);

  sendBit(0);  // Start bit

  int parityCount = 0;

  // Send 8 data bits, LSB first
  for (int i = 0; i < 8; i++) {
    bool bit = (c >> i) & 0x01;
    sendBit(bit);
    parityCount += bit;
  }

  // Send even parity bit
  bool parityBit = (parityCount % 2 == 0) ? 0 : 1;
  sendBit(parityBit);

  // Send stop bit
  sendBit(1);
}

void sendBit(bool bit) {
  digitalWrite(laserPin, bit ? HIGH : LOW);
  delay(bitDuration);
}
