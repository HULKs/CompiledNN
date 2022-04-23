#include "Thinterface.h"
#include "CompiledNN.h"
#include "Model.h"
#include <string>

CompiledNN::CompiledNN()
    : core{new NeuralNetwork::CompiledNN} {}

CompiledNN::~CompiledNN() {
  delete reinterpret_cast<NeuralNetwork::CompiledNN *>(core);
}

void CompiledNN::compile(const char *filename) {
  core->compile(filename);
}

float *CompiledNN::input(std::size_t index) {
  return core->input(index).data();
}

float *CompiledNN::output(std::size_t index) {
  return core->output(index).data();
}

unsigned long CompiledNN::inputSize(unsigned long index) {
  return core->input(index).size();
}

unsigned long CompiledNN::outputSize(unsigned long index) {
  return core->output(index).size();
}

void CompiledNN::apply() {
  return core->apply();
}
