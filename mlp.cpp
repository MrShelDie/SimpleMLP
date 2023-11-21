#include "mlp.h"

#include <cassert>

MLP::MLP(std::vector<size_t> layer_sizes) {
  size_t layers_cnt = layer_sizes.size();

  assert(layers_cnt > 1);
  neuron_matrix_.resize(layers_cnt);

  /* Fill in the matrix of neurons.
   * Add a bias to every layer except the last one. */
  for (size_t i = 0; i < layers_cnt; ++i) {
    auto layer_size = layer_sizes[i];

    assert(layer_size > 0);

    if (i < layers_cnt - 1) {
      ++layer_size;
    }

    neuron_matrix_[i].reserve(layer_size);

    for (size_t j = 0; j < layer_size - 1; ++j) {
      neuron_matrix_[i].emplace_back(Neuron(j, false));
    }

    if (i < layers_cnt - 1) {
      neuron_matrix_[i].emplace_back(Neuron(layer_size - 1, true));
    } else {
      neuron_matrix_[i].emplace_back(Neuron(layer_size - 1, false));
    }
  }

  /* Link neighbouring layers to each other.
   * The first and the last layer have only one neighbour */
  for (size_t i = 0; i < layers_cnt; ++i) {
    for (size_t j = 0; j < neuron_matrix_[i].size(); ++j) {
      if (i == 0) {
        neuron_matrix_[i][j].SetLayers(nullptr, &neuron_matrix_[i + 1]);
      } else if (i == layers_cnt - 1) {
        neuron_matrix_[i][j].SetLayers(&neuron_matrix_[i - 1], nullptr);
      } else {
        neuron_matrix_[i][j].SetLayers(&neuron_matrix_[i - 1], &neuron_matrix_[i + 1]);
      }
    }
  }
}

std::ostream& operator<<(std::ostream& os, const MLP& net) {
  for (size_t i = 1; i < net.neuron_matrix_.size(); ++i) {
    for (size_t j = 0; j < net.neuron_matrix_[i].size(); ++j) {
      os << net.neuron_matrix_[i][j];
    }
    os << std::endl;
  }
  return os;
}

std::vector<double> MLP::ForwardPropogation(const std::vector<double>& input_values) {
  assert(input_values.size() == neuron_matrix_[0].size() - 1);

  for (size_t j = 0; j < input_values.size(); ++j) {
    neuron_matrix_[0][j].SetOutput(input_values[j]);
  }

  for (size_t i = 1; i < neuron_matrix_.size(); ++i) {
    for (size_t j = 0; j < neuron_matrix_[i].size(); ++j) {
      neuron_matrix_[i][j].ForwardPropogation();
    }
  }

  const size_t outputs_size = neuron_matrix_.back().size();
  std::vector<double> outputs(outputs_size);

  for (size_t j = 0; j < outputs_size; ++j) {
    outputs[j] = neuron_matrix_.back()[j].GetOutput();
  }

  return outputs;
}
