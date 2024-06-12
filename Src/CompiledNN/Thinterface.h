#pragma once

namespace NeuralNetwork {
class CompiledNN;
}

struct Tensor {
  const float *data;
  unsigned long data_size;
  const unsigned int *dimensions;
  unsigned long dimensions_size;
};

struct TensorMut {
  float *data;
  unsigned long data_size;
  const unsigned int *dimensions;
  unsigned long dimensions_size;
};

struct CompiledNN {
  CompiledNN();
  CompiledNN(const CompiledNN &model) = delete;
  CompiledNN(CompiledNN &&model) = delete;
  CompiledNN &operator=(const CompiledNN &model) = delete;
  CompiledNN &operator=(CompiledNN &&model) = delete;
  ~CompiledNN();

  void compile(const char *filename);
  void apply();

  Tensor input(unsigned long index) const;
  Tensor output(unsigned long index) const;

  TensorMut input_mut(unsigned long index);
  TensorMut output_mut(unsigned long index);

private:
  NeuralNetwork::CompiledNN *core{nullptr};
};
