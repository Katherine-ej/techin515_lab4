// Edge Impulse ingestion SDK
// Copyright (c) 2022 EdgeImpulse Inc.
//
// Licensed under the Apache License, Version 2.0
// http://www.apache.org/licenses/LICENSE-2.0

/* Includes ---------------------------------------------------------------- */
#include <KatherineC-project-1_inferencing.h>
#include <Adafruit_MPU6050.h>
#include <Adafruit_Sensor.h>
#include <Wire.h>

// Initialize MPU6050 sensor
Adafruit_MPU6050 mpu;

// Sampling and capture configuration
#define SAMPLE_RATE_MS 10        // 100Hz sampling rate (10ms interval)
#define CAPTURE_DURATION_MS 1000 // Total capture duration: 1 second
#define FEATURE_SIZE EI_CLASSIFIER_DSP_INPUT_FRAME_SIZE // Number of input features

// LED and button pin definitions
#define RED_PIN 9
#define GREEN_PIN 7
#define BLUE_PIN 8
#define BUTTON_PIN 10

// Variables for capture state
bool capturing = false;
unsigned long last_sample_time = 0;
unsigned long capture_start_time = 0;
int sample_count = 0;

// Array to store accelerometer readings
float features[FEATURE_SIZE];

/**
 * @brief Copies feature data to the classifier input
 */
int raw_feature_get_data(size_t offset, size_t length, float *out_ptr) {
    memcpy(out_ptr, features + offset, length * sizeof(float));
    return 0;
}

/**
 * @brief Declares function to print inference result
 */
void print_inference_result(ei_impulse_result_t result);

/**
 * @brief Arduino setup function
 */
void setup() {
    Serial.begin(115200);

    // Initialize RGB LEDs
    pinMode(RED_PIN, OUTPUT);
    pinMode(GREEN_PIN, OUTPUT);
    pinMode(BLUE_PIN, OUTPUT);

    // Turn off all LEDs at the beginning
    digitalWrite(RED_PIN, LOW);
    digitalWrite(GREEN_PIN, LOW);
    digitalWrite(BLUE_PIN, LOW);

    // Initialize push button
    pinMode(BUTTON_PIN, INPUT_PULLUP);

    // Initialize MPU6050 sensor
    Serial.println("Initializing MPU6050...");
    if (!mpu.begin()) {
        Serial.println("Failed to find MPU6050 chip");
        while (1) {
            delay(10);
        }
    }

    // Set sensor configuration (same as training)
    mpu.setAccelerometerRange(MPU6050_RANGE_8_G);
    mpu.setGyroRange(MPU6050_RANGE_500_DEG);
    mpu.setFilterBandwidth(MPU6050_BAND_21_HZ);

    Serial.println("MPU6050 initialized successfully");
    Serial.println("Press the button to start gesture capture");
}

/**
 * @brief Capture accelerometer data and store into features[]
 */
void capture_accelerometer_data() {
    if (millis() - last_sample_time >= SAMPLE_RATE_MS) {
        last_sample_time = millis();

        // Read accelerometer and gyroscope values
        sensors_event_t a, g, temp;
        mpu.getEvent(&a, &g, &temp);

        // Store accelerometer (x, y, z) values in feature array
        if (sample_count < FEATURE_SIZE / 3) {
            int idx = sample_count * 3;
            features[idx] = a.acceleration.x;
            features[idx + 1] = a.acceleration.y;
            features[idx + 2] = a.acceleration.z;
            sample_count++;
        }

        // Check if capture is complete
        if (millis() - capture_start_time >= CAPTURE_DURATION_MS) {
            capturing = false;
            Serial.println("Capture complete");
            run_inference();
        }
    }
}

/**
 * @brief Run Edge Impulse inference on captured data
 */
void run_inference() {
    if (sample_count * 3 < EI_CLASSIFIER_DSP_INPUT_FRAME_SIZE) {
        Serial.println("ERROR: Not enough data for inference");
        return;
    }

    ei_impulse_result_t result = { 0 };

    // Prepare signal structure for inference
    signal_t features_signal;
    features_signal.total_length = EI_CLASSIFIER_DSP_INPUT_FRAME_SIZE;
    features_signal.get_data = &raw_feature_get_data;

    // Run the classifier
    EI_IMPULSE_ERROR res = run_classifier(&features_signal, &result, false);
    if (res != EI_IMPULSE_OK) {
        Serial.print("ERR: Failed to run classifier (");
        Serial.print(res);
        Serial.println(")");
        return;
    }

    print_inference_result(result);
}

/**
 * @brief Arduino main loop
 * Checks for button press to trigger data capture
 */
void loop() {
    // When button is pressed and not already capturing
    if (digitalRead(BUTTON_PIN) == LOW && !capturing) {
        Serial.println("Starting gesture capture...");
        sample_count = 0;
        capturing = true;
        capture_start_time = millis();
        last_sample_time = millis();

        // Wait until button is released (debounce)
        while (digitalRead(BUTTON_PIN) == LOW) {
            delay(10);
        }
    }

    // If capturing, read and store accelerometer data
    if (capturing) {
        capture_accelerometer_data();
    }
}

/**
 * @brief Display prediction result and control LEDs accordingly
 */
void print_inference_result(ei_impulse_result_t result) {
    float max_value = 0;
    int max_index = -1;

    for (uint16_t i = 0; i < EI_CLASSIFIER_LABEL_COUNT; i++) {
        if (result.classification[i].value > max_value) {
            max_value = result.classification[i].value;
            max_index = i;
        }
    }

    // Turn off all LEDs before showing result
    digitalWrite(RED_PIN, LOW);
    digitalWrite(GREEN_PIN, LOW);
    digitalWrite(BLUE_PIN, LOW);

    if (max_index != -1) {
        const char* prediction = ei_classifier_inferencing_categories[max_index];
        Serial.print("Prediction: ");
        Serial.print(prediction);
        Serial.print(" (");
        Serial.print(max_value * 100);
        Serial.println("%)");

        // Light corresponding LED based on prediction label
        if (strcmp(prediction, "Z") == 0) {
            digitalWrite(RED_PIN, HIGH);
        }
        else if (strcmp(prediction, "O") == 0) {
            digitalWrite(BLUE_PIN, HIGH);
        }
        else if (strcmp(prediction, "V") == 0) {
            digitalWrite(GREEN_PIN, HIGH);
        }
    }
}