#include <SimpleFOC.h>

// SimpleFOC Shield v3.2
#define INH_A 27
#define INH_B 5
#define INH_C 16
#define EN_PIN 12

// Encodeur AS5048A
#define SENSOR_CS 13

#define POLE_PAIRS 7
#define MAX_RPM 20.0

MagneticSensorSPI sensor = MagneticSensorSPI(SENSOR_CS, 14, 0x3FFF);
BLDCMotor motor = BLDCMotor(POLE_PAIRS);
BLDCDriver3PWM driver = BLDCDriver3PWM(INH_A, INH_B, INH_C, EN_PIN);

// Vitesse en rad/s (calculee a partir du RPM)
float target_rpm = 0;
float ramp_velocity = 0;

float start_angle = 0;
unsigned long start_time = 0;
bool running = false;

// PID gains (tunes)
float Kp = 110.0;
float Kd = 0.8;
float Ki = 15.0;
float Kff = 0.5;

// PID state
float integral = 0;
float prev_error = 0;
unsigned long prev_time_us = 0;
float filtered_deriv = 0;
float deriv_alpha = 0.05;

Commander command = Commander(Serial);

void startRamp() {
  sensor.update();
  start_angle = sensor.getAngle();
  start_time = millis();
  integral = 0;
  prev_error = 0;
  filtered_deriv = 0;
  prev_time_us = micros();
  running = true;
}

void doRPM(char* cmd) {
  float rpm = 0;
  command.scalar(&rpm, cmd);

  if (rpm == 0) {
    // Afficher le RPM actuel
    Serial.print("RPM: ");
    Serial.println(target_rpm, 3);
    return;
  }

  if (fabs(rpm) > MAX_RPM) {
    Serial.print("SECURITE: max ");
    Serial.print(MAX_RPM, 0);
    Serial.println(" rpm");
    return;
  }

  target_rpm = rpm;
  ramp_velocity = target_rpm * 2.0 * PI / 60.0;
  Serial.print("RPM: ");
  Serial.print(target_rpm, 3);
  Serial.print(" (");
  Serial.print(ramp_velocity, 5);
  Serial.println(" rad/s)");

  startRamp();
}

void doStop(char* cmd) {
  running = false;
  target_rpm = 0;
  ramp_velocity = 0;
  integral = 0;
  Serial.println("Stop");
}

void setup() {
  Serial.begin(115200);
  delay(1000);

  sensor.init();
  motor.linkSensor(&sensor);

  driver.voltage_power_supply = 19;
  driver.voltage_limit = 19;
  driver.init();

  motor.linkDriver(&driver);
  motor.foc_modulation = FOCModulationType::SpaceVectorPWM;
  motor.controller = MotionControlType::torque;
  motor.voltage_limit = 4;
  motor.voltage_sensor_align = 1;

  motor.init();
  motor.initFOC();

  command.add('R', doRPM, "rpm");
  command.add('X', doStop, "stop");

  Serial.println("=== Motor Control ===");
  Serial.print("Max: ");
  Serial.print(MAX_RPM, 0);
  Serial.println(" rpm");
  Serial.println("R<rpm> = set speed, X = stop");
}

void loop() {
  motor.loopFOC();
  sensor.update();

  float current_angle = sensor.getAngle();
  float voltage = 0;

  if (running) {
    unsigned long now_us = micros();
    float dt = (now_us - prev_time_us) * 1e-6;
    prev_time_us = now_us;
    if (dt <= 0 || dt > 0.1) dt = 0.001;

    float elapsed = (millis() - start_time) / 1000.0;
    float target_angle = start_angle + ramp_velocity * elapsed;
    float error = target_angle - current_angle;

    integral += error * dt;
    if (integral > 3.0) integral = 3.0;
    if (integral < -3.0) integral = -3.0;

    float raw_deriv = (error - prev_error) / dt;
    filtered_deriv = filtered_deriv * (1.0 - deriv_alpha) + raw_deriv * deriv_alpha;
    prev_error = error;

    voltage = Kff * ramp_velocity + Kp * error + Ki * integral + Kd * filtered_deriv;

    if (voltage > motor.voltage_limit) voltage = motor.voltage_limit;
    if (voltage < -motor.voltage_limit) voltage = -motor.voltage_limit;
  }

  motor.move(voltage);
  command.run();

  static unsigned long last = 0;
  if (millis() - last > 500) {
    last = millis();
    float err_deg = 0;
    if (running) {
      float target = start_angle + ramp_velocity * (millis() - start_time) / 1000.0;
      err_deg = degrees(target - current_angle);
    }
    float total_deg = degrees(current_angle - start_angle);
    float total_turns = total_deg / 360.0;
    Serial.print("Tours:");
    Serial.print(total_turns, 3);
    Serial.print(" (");
    Serial.print(total_deg, 1);
    Serial.print("°) Err:");
    Serial.print(err_deg, 3);
    Serial.print(" rpm:");
    Serial.println(target_rpm, 3);
  }
}
