#include <Wire.h>
#include <Adafruit_GFX.h>
#include <Adafruit_SSD1306.h>

// ---------- Pin Configuration ----------
#define TRIG_PIN 27
#define ECHO_PIN 26
#define BUZZER_PIN 25

// ---------- OLED Configuration ----------
#define SCREEN_WIDTH 128
#define SCREEN_HEIGHT 64
Adafruit_SSD1306 display(SCREEN_WIDTH, SCREEN_HEIGHT, &Wire, -1);

// ---------- Setup ----------
void setup() {
  Serial.begin(115200);

  pinMode(TRIG_PIN, OUTPUT);
  pinMode(ECHO_PIN, INPUT);
  pinMode(BUZZER_PIN, OUTPUT);

  // Initialize OLED display
  if (!display.begin(SSD1306_SWITCHCAPVCC, 0x3C)) {  // Default I2C address: 0x3C
    Serial.println(F("SSD1306 allocation failed"));
    for (;;);  // Stop here if display fails
  }

  display.clearDisplay();
  display.setTextSize(1);
  display.setTextColor(SSD1306_WHITE);
  display.setCursor(0, 10);
  display.println("System Ready...");
  display.display();
  delay(2000);
}

// ---------- Loop ----------
void loop() {
  long duration;
  float distance;

  // --- Trigger Ultrasonic Pulse ---
  digitalWrite(TRIG_PIN, LOW);
  delayMicroseconds(2);
  digitalWrite(TRIG_PIN, HIGH);
  delayMicroseconds(10);
  digitalWrite(TRIG_PIN, LOW);

  // --- Measure Echo ---
  duration = pulseIn(ECHO_PIN, HIGH, 30000); // Timeout 30ms
  if (duration == 0) {
    Serial.println("No reading (timeout)");
    return;
  }

  distance = duration * 0.034 / 2;  // Convert microseconds to cm

  Serial.print("Distance: ");
  Serial.print(distance);
  Serial.println(" cm");

  // --- Display and Buzzer Logic ---
  display.clearDisplay();
  display.setTextSize(1);
  display.setTextColor(SSD1306_WHITE);

  if (distance < 20) {
    // Too close
    digitalWrite(BUZZER_PIN, LOW);
    display.setCursor(0, 20);
    display.println("Too close to camera!");
  } 
  else if (distance >= 20 && distance <= 25) {
    // In alert range → short beep pattern
    digitalWrite(BUZZER_PIN, HIGH);
    delay(100);
    digitalWrite(BUZZER_PIN, LOW);
    delay(100);
    display.setCursor(0, 20);
    display.println("Alert! In range");
  } 
  else if (distance > 25) {
    // Too far
    digitalWrite(BUZZER_PIN, LOW);
    display.setCursor(0, 20);
    display.println("Away from camera");
  } 
  else {
    digitalWrite(BUZZER_PIN, LOW);
    display.setCursor(0, 20);
    display.println("Measuring...");
  }

  display.display();
  delay(200);  // Small delay for stability
}