const int ldrPin = A0;                   // Analog pin connected to LDR
const unsigned int bitDuration = 10;   // Milliseconds per bit = 10 bps
const int threshold = 100;              // Adjust based on your setup

unsigned long serialNumber = 0;         // Serial number for each character

void setup() {
  Serial.begin(9600);
  delay(1);
  Serial.println("Serial Number,Bit Number,Analog Value,Binary Bit,Character");
}

void loop() {
  if (waitForStartBit()) {
    char receivedChar;
    bool valid = receiveCharWithCSV(receivedChar);

    if (!valid) {
      Serial.print(serialNumber++);
      Serial.println(",,,,[ERROR]");
    }
  }
}

bool waitForStartBit() {
  while (analogRead(ldrPin) > threshold);  // Wait for laser OFF (LOW)
  delay(bitDuration / 2);

  if (analogRead(ldrPin) <= threshold) {
    delay(bitDuration / 2);
    return true;
  }

  return false;
}

bool receiveCharWithCSV(char &character) {
  character = 0;
  int parityCount = 0;

  // Read 8 data bits
  for (int i = 0; i < 8; i++) {
    int ldrValue = analogRead(ldrPin);
    bool bit = ldrValue > threshold;
    character |= (bit << i);
    parityCount += bit;

    Serial.print(serialNumber);     // Serial number
    Serial.print(",");
    Serial.print(i);                // Bit number
    Serial.print(",");
    Serial.print(ldrValue);         // Analog value
    Serial.print(",");
    Serial.print(bit);              // Binary bit
    Serial.print(",");

    if (i == 7) {
      Serial.println(character);    // Print character on last bit
      serialNumber++;               // Increment after full character
    } else {
      Serial.println();             // Empty character column for other bits
    }

    delay(bitDuration);
  }

  // Parity bit
  int parityLdr = analogRead(ldrPin);
  bool parityBit = parityLdr > threshold;
  parityCount += parityBit;
  delay(bitDuration);

  // Stop bit
  int stopLdr = analogRead(ldrPin);
  bool stopBit = stopLdr > threshold;
  delay(bitDuration);

  return (parityCount % 2 == 0) && stopBit;
}
