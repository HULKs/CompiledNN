#include "Thinterface.h"
#include "CompiledNN.h"
#include "Model.h"
#include <string>

CompiledNN::CompiledNN() : core{new NeuralNetwork::CompiledNN} {}

CompiledNN::~CompiledNN() {
  delete reinterpret_cast<NeuralNetwork::CompiledNN *>(core);
}

void CompiledNN::compile(const char *filename) { core->compile(filename); }

Tensor CompiledNN::input(std::size_t index) const {
  Tensor tensor{core->input(index).data(), core->input(index).size(),
                core->input(index).dims().data(),
                core->input(index).dims().size()};
  return tensor;
}

TensorMut CompiledNN::input_mut(std::size_t index) {
  TensorMut tensor{core->input(index).data(), core->input(index).size(),
                   core->input(index).dims().data(),
                   core->input(index).dims().size()};
  return tensor;
}

Tensor CompiledNN::output(std::size_t index) const {
  Tensor tensor{core->output(index).data(), core->output(index).size(),
                core->output(index).dims().data(),
                core->output(index).dims().size()};
  return tensor;
}

TensorMut CompiledNN::output_mut(std::size_t index) {
  TensorMut tensor{core->output(index).data(), core->output(index).size(),
                   core->output(index).dims().data(),
                   core->output(index).dims().size()};
  return tensor;
}

void CompiledNN::apply() { return core->apply(); }
