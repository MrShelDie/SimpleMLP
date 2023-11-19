#include "layer.h"

#include <cassert>

Layer::Layer(size_t layer_size, bool need_bias) : has_bias_(need_bias) {
  if (need_bias) {
    neurons_.reserve(layer_size + 1);
  } else {
    neurons_.reserve(layer_size);
  }

  for (size_t i = 0; i < layer_size; ++i) {
    neurons_.emplace_back(Neuron(i));
  }

  if (need_bias) {
    neurons_.emplace_back(Neuron(Neuron::kBiasId));
  }
}

std::ostream& operator<<(std::ostream& os, const Layer& layer) {
  size_t size = layer.neurons_.size();

  if (layer.has_bias_) {
    --size;
  }

  for (size_t i = 0; i < size; ++i) {
    os << layer.neurons_[i];
  }

  return os;
}

void Layer::SetNeuronsOutput(const std::vector<double>& values) {
  assert((has_bias_ && values.size() == neurons_.size() - 1)
         || (!has_bias_ && values.size() == neurons_.size()));

  for (size_t i = 0; i < values.size(); ++i) {
    neurons_[i].SetOutput(values[i]);
  }
}

std::vector<double> Layer::GetNeuronsOutput() {
  size_t size = neurons_.size();

  if (has_bias_) {
    --size;
  }

  std::vector<double> outputs(size);

  for (size_t i = 0; i < size; ++i) {
    outputs[i] = neurons_[i].GetOutput();
  }

  return outputs;
}

size_t Layer::GetSize() const {
  return neurons_.size();
}

const std::vector<Neuron>& Layer::GetNeurons() const {
  return neurons_;
}

void Layer::LinkLayers(Layer* prev, Layer* next) {
  size_t size = neurons_.size();

  if (has_bias_) {
    --size;
  }

  for (size_t i = 0; i < size; ++i) {
    neurons_[i].SetLayers(prev, next);
  }

  if (has_bias_) {
    neurons_[size].SetLayers(nullptr, next);
  }
}

void Layer::ForwardPropogation() {
  size_t size = neurons_.size();

  if (has_bias_) {
    --size;
  }

  for (size_t i = 0; i < size; ++i) {
    neurons_[i].ForwardPropogation();
  }
}
