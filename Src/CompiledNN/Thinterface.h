#pragma once

namespace NeuralNetwork {
  class CompiledNN;
}

struct Tensor {
  float* data;
  unsigned long data_size;
  const unsigned int* dimensions;
  unsigned int dimensions_size;
};

struct CompiledNN {
  CompiledNN();
  CompiledNN(const CompiledNN &model) = delete;
  CompiledNN(CompiledNN &&model) = delete;
  CompiledNN &operator=(const CompiledNN &model) = delete;
  CompiledNN &operator=(CompiledNN &&model) = delete;
  ~CompiledNN();

  void compile(const char* filename);

  Tensor input(unsigned long index);
  Tensor output(unsigned long index);

  unsigned long inputSize(unsigned long index);

  void apply();

private:
  NeuralNetwork::CompiledNN *core{nullptr};
};
