#include "neuron.h"

#include <cmath>
#include <cassert>

#include <random>
#include <iomanip>

Neuron::Neuron(size_t id) : id_(id), out_(1) {}

std::ostream& operator<<(std::ostream& os, const Neuron& neuron) {
  assert(neuron.id_ != Neuron::kBiasId);

  os << '[';

  for (auto weight : neuron.input_weights_) {
    os << ' ' << std::setw(7) << std::setprecision(4) << std::right << weight
       << ' ';
  }

  os << "]\t";

  return os;
}

void Neuron::SetLayers(Layer* prev, Layer* next) {
  assert((id_ != kBiasId) || (!prev));

  prev_layer_ = prev;
  next_layer_ = next;

  if (!prev) {
    return;
  }

  static std::random_device rd;
  static std::mt19937 rng(rd());
  static std::uniform_real_distribution<double> distrib(-1.0, 1.0);

  size_t prev_layer_size = prev->GetSize();
  input_weights_.reserve(prev_layer_size);

  for (size_t i = 0; i < prev_layer_size; ++i) {
    double rand_weight = distrib(rng);
    input_weights_.emplace_back(rand_weight);
  }
}

void Neuron::SetOutput(double out_value) {
  assert(id_ != kBiasId);

  out_ = out_value;
}

double Neuron::GetOutput() {
  return out_;
}

void Neuron::ForwardPropogation() {
  assert(id_ != kBiasId);
  assert(!input_weights_.empty());

  out_ = 0;
  const auto& prev_layer_neurons = prev_layer_->GetNeurons();

  for (size_t i = 0; i < input_weights_.size(); ++i) {
    out_ += input_weights_[i] * prev_layer_neurons[i].out_;
  }

  out_ = 1 / (1 + exp(-out_));
}
