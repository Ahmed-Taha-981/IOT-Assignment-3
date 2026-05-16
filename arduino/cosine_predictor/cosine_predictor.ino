#include <Arduino.h>
#include <math.h>

#include "model.h"

#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "tensorflow/lite/version.h"

constexpr int kTensorArenaSize = 4 * 1024;
alignas(16) static uint8_t tensor_arena[kTensorArenaSize];

const tflite::Model* model = nullptr;
tflite::MicroInterpreter* interpreter = nullptr;
TfLiteTensor* input = nullptr;
TfLiteTensor* output = nullptr;

float input_scale = 0.0f;
int input_zero_point = 0;
float output_scale = 0.0f;
int output_zero_point = 0;

float x_value = 0.0f;
const float kTwoPi = 6.28318530718f;
const int kSteps = 80;

void setup() {
  Serial.begin(115200);
  delay(2000);

  Serial.println("TinyML Cosine Predictor Starting...");

  model = tflite::GetModel(g_model);
  if (model->version() != TFLITE_SCHEMA_VERSION) {
    Serial.println("Model schema version mismatch!");
    while (true) {}
  }

  static tflite::MicroMutableOpResolver<2> resolver;
  resolver.AddFullyConnected();
  resolver.AddRelu();

  static tflite::MicroInterpreter static_interpreter(
    model,
    resolver,
    tensor_arena,
    kTensorArenaSize
  );

  interpreter = &static_interpreter;

  TfLiteStatus allocate_status = interpreter->AllocateTensors();
  if (allocate_status != kTfLiteOk) {
    Serial.println("AllocateTensors() failed");
    while (true) {}
  }

  input = interpreter->input(0);
  output = interpreter->output(0);

  input_scale = input->params.scale;
  input_zero_point = input->params.zero_point;
  output_scale = output->params.scale;
  output_zero_point = output->params.zero_point;

  Serial.println("Startup Diagnostics:");
  Serial.print("g_model size bytes: ");
  Serial.println(g_model_len);

  Serial.print("Tensor arena size bytes: ");
  Serial.println(kTensorArenaSize);

  Serial.print("Tensor arena used bytes: ");
  Serial.println(interpreter->arena_used_bytes());

  Serial.print("Input scale: ");
  Serial.println(input_scale, 8);

  Serial.print("Input zero_point: ");
  Serial.println(input_zero_point);

  Serial.print("Output scale: ");
  Serial.println(output_scale, 8);

  Serial.print("Output zero_point: ");
  Serial.println(output_zero_point);

  Serial.println("predicted,actual");
}

void loop() {
  for (int i = 0; i < kSteps; i++) {
    x_value = (kTwoPi * i) / kSteps;

    int8_t quantized_x = (int8_t)round((x_value / input_scale) + input_zero_point);
    input->data.int8[0] = quantized_x;

    TfLiteStatus invoke_status = interpreter->Invoke();
    if (invoke_status != kTfLiteOk) {
      Serial.println("Invoke failed");
      return;
    }

    int8_t quantized_y = output->data.int8[0];
    float predicted = (quantized_y - output_zero_point) * output_scale;
    float actual = cos(x_value);

    Serial.print(predicted, 6);
    Serial.print(",");
    Serial.println(actual, 6);

    delay(50);
  }
}
