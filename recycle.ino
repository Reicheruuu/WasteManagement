#include <Servo.h>
// upload the code in arduino before running the python code or python application
void setup() {
  Serial.begin(115200);
}


void biodegradable() {
  // main function of your biodegradable
}

void non_biodegradable() {
  // main function of your non-biodegradable
}

void recyclable() {
  // main function of your recyclable
}

void loop() {
  if (Serial.available() > 0) {
    char command = Serial.read(); // Read the incoming command

    if (command == '1') { // Command for biodegradable
      biodegradable();
    }
    else if (command == '2') { // Command for non-biodegradable
      non_biodegradable();
    }
    else if (command == '3') { // Command for recyclable
      recyclable();
    }
  }
}
