#include "mlp.h"

#include <cassert>

MLP::MLP(std::initializer_list<size_t> layer_sizes) {
  size_t layers_cnt = layer_sizes.size();

  assert(layers_cnt > 1);
  layers_.reserve(layers_cnt);

  auto it = layer_sizes.begin();

  for (; it != std::prev(layer_sizes.end()); ++it) {
    assert(*it > 0);
    layers_.emplace_back(Layer(*it));
  }

  assert(*it > 0);
  layers_.emplace_back(Layer(*it, false));

  layers_[0].LinkLayers(nullptr, &layers_[1]);

  for (size_t i = 1; i < layers_cnt - 1; ++i) {
    layers_[i].LinkLayers(&layers_[i - 1], &layers_[i + 1]);
  }

  layers_[layers_cnt - 1].LinkLayers(&layers_[layers_cnt - 2], nullptr);
}

std::ostream& operator<<(std::ostream& os, const MLP& net) {
  for (size_t i = 1; i < net.layers_.size(); ++i) {
    os << net.layers_[i] << std::endl;
  }
  return os;
}

std::vector<double> MLP::ForwardPropogation(
    const std::vector<double>& input_values) {
  layers_[0].SetNeuronsOutput(input_values);

  for (size_t i = 1; i < layers_.size(); ++i) {
    layers_[i].ForwardPropogation();
  }

  return layers_[layers_.size() - 1].GetNeuronsOutput();
}
