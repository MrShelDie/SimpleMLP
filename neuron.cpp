#include "neuron.h"

#include <cmath>
#include <cassert>

#include <random>
#include <iomanip>

Neuron::Neuron(size_t id, bool is_bias)
    : id_(id),
      is_bias_(is_bias),
      prev_layer_(nullptr),
      next_layer_(nullptr),
      out_(1) {}

std::ostream& operator<<(std::ostream& os, const Neuron& neuron) {
  if (neuron.is_bias_) {
    return os;
  }

  os << '[';

  for (auto weight : neuron.input_weights_) {
    os << ' ' << std::setw(7) << std::setprecision(4) << std::right << weight
       << ' ';
  }

  os << "]\t";

  return os;
}

void Neuron::SetLayers(std::vector<Neuron>* prev, std::vector<Neuron>* next) {
  if (!is_bias_) {
    prev_layer_ = prev;
  }

  next_layer_ = next;

  if (!prev_layer_) {
    return;
  }

  static std::random_device rd;
  static std::mt19937 rng(rd());
  static std::uniform_real_distribution<double> distrib(-1.0, 1.0);

  input_weights_.reserve(prev_layer_->size());

  for (size_t i = 0; i < prev_layer_->size(); ++i) {
    double rand_weight = distrib(rng);
    input_weights_.emplace_back(rand_weight);
  }
}

void Neuron::ForwardPropogation() {
  if (is_bias_) {
    return;
  }

  assert(!input_weights_.empty());

  out_ = 0;

  for (size_t i = 0; i < input_weights_.size(); ++i) {
    out_ += (*prev_layer_)[i].GetOutput() * input_weights_[i];
  }

  out_ = 1 / (1 + exp(-out_));
}
