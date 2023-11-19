#ifndef LAYER_H
#define LAYER_H

#include "neuron.h"

#include <vector>
#include <memory>
#include <ostream>

class Neuron;

class Layer {
 public:
  Layer(size_t layer_size, bool need_bias = true);

  friend std::ostream& operator<<(std::ostream& os, const Layer& layer);

  void SetNeuronsOutput(const std::vector<double>& values);
  std::vector<double> GetNeuronsOutput();

  size_t GetSize() const;
  const std::vector<Neuron>& GetNeurons() const;

  void LinkLayers(Layer* prev, Layer* next);
  void ForwardPropogation();

 private:
  std::vector<Neuron> neurons_;
  bool has_bias_;
};

#endif  // LAYER_H
