#ifndef NEURON_H
#define NEURON_H

#include "layer.h"

#include <vector>
#include <ostream>

class Layer;

class Neuron {
 public:
  static const size_t kBiasId = static_cast<size_t>(-1);

  Neuron(size_t id);

  friend std::ostream& operator<<(std::ostream& os, const Neuron& neuron);

  std::ostream& Print(std::ostream& os) const;
  void SetLayers(Layer* prev, Layer* next);
  void SetOutput(double out_value);
  double GetOutput();
  void ForwardPropogation();

 private:
  size_t id_;
  Layer* prev_layer_;
  Layer* next_layer_;
  std::vector<double> input_weights_;
  double out_;
};

#endif  // NEURON_H
