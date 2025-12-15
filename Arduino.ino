/*
 * Arduino UNO TRIAC Control with Zero-Crossing Detection
 *
 * This code has been written by Hadrien Guillaud on the 10th of December 2025 with the help of generative IA. 
 * This work was part of the project course EI2525 Electric Power Engineering Project.
 * 
 * Group 3:
 *  - Dio Randa Damara
 *  - Iraklis Bournazos
 *  - Hadrien Guillaud
 *  - Nathaniel Taylor (supervisor)
 *  - Lina Bertling Tjernberg (examiner)
 * 
 * This program controls an AC load using a TRIAC and zero-crossing detection.
 * It reads a duty cycle value (0.0 to 1.0) from the serial USB port (JSON format: {"duty":x})
 * and adjusts the firing angle of the TRIAC to regulate the power delivered to the load.
 * The zero-crossing signal is used to synchronize the TRIAC triggering with the AC waveform,
 * ensuring smooth and efficient power control.
 *
 * Connections:
 * - Zero-crossing detector output to pin 2 (interrupt-capable)
 * - Optotriac input to pin 3
 *
 * Serial communication:
 * - Send JSON commands like {"duty":0.5} to set the duty cycle.
 * - The board responds with status or error messages.
 *
 * Safety:
 * - Ensure proper isolation between the AC mains and the Arduino.
 * - Use appropriate components for safety and avoid touching unprotected areas.
 * - Always disconnect the AC power before handling the circuit.
*/

#include <Arduino.h>
#include <ArduinoJson.h>

// Pins
const int ZERO_CROSS_PIN = 2;      // Zero-crossing detection pin
const int TRIAC_PIN = 3;           // Optotriac control pin

// Global variables
volatile bool zero_cross_detected = false;
volatile unsigned long last_zero_cross_time = 0;
volatile unsigned long current_zero_cross_time = 0;
unsigned long half_period_us = 10000; // Initial value (10 ms for 50 Hz)
float duty_cycle = 0.0; // Duty cycle (0.0 to 1.0)

void setup() {
  Serial.begin(9600);
  pinMode(ZERO_CROSS_PIN, INPUT_PULLUP);
  pinMode(TRIAC_PIN, OUTPUT);
  attachInterrupt(digitalPinToInterrupt(ZERO_CROSS_PIN), zeroCrossDetect, CHANGE);
}

void loop() {
  // Read serial commands in format {"duty":x}
  if (Serial.available() > 0) {
    String input = Serial.readStringUntil('\n');
    input.trim();

    StaticJsonDocument<64> doc;
    DeserializationError error = deserializeJson(doc, input);

    if (!error && doc.containsKey("duty")) {
      duty_cycle = doc["duty"];
      duty_cycle = constrain(duty_cycle, 0.0, 1.0);
      Serial.print("{\"status\":\"ok\",\"duty\":");
      Serial.print(duty_cycle);
      Serial.println("}");
    } else {
      Serial.println("{\"error\":\"Invalid JSON or missing 'duty' key\"}");
    }
  }
}

// Zero-crossing detection
void zeroCrossDetect() {
  static unsigned long last_interrupt_time = 0;
  unsigned long interrupt_time = micros();

  // Debounce: ignore interrupts that are too close
  if (interrupt_time - last_interrupt_time < 100) {
    return;
  }
  last_interrupt_time = interrupt_time;

  current_zero_cross_time = interrupt_time;
  if (last_zero_cross_time != 0) {
    half_period_us = current_zero_cross_time - last_zero_cross_time;
  }
  last_zero_cross_time = current_zero_cross_time;

  int delay_us = calculateDelayFromDutyCycle(duty_cycle);
  fireTriac(delay_us);
}

// Convert duty_cycle to delay in microseconds
int calculateDelayFromDutyCycle(float duty) {
  if (duty <= 0.0) {
    return half_period_us + 1;
  } else if (duty >= 1.0) {
    return 0;
  } else {
    return half_period_us - (duty * half_period_us);
  }
}

// Trigger the TRIAC after a delay
void fireTriac(int delay_us) {
  if (delay_us == 0) {
    digitalWrite(TRIAC_PIN, HIGH);
  } else if (delay_us <= half_period_us) {
    delayMicroseconds(delay_us);
    digitalWrite(TRIAC_PIN, HIGH);
    delayMicroseconds(10);
    digitalWrite(TRIAC_PIN, LOW);
  }
}
