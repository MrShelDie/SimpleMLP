#ifndef NEURON_H
#define NEURON_H

#include <vector>
#include <ostream>

class Layer;

class Neuron {
 public:
  Neuron(size_t id, bool is_bias = false);

  friend std::ostream& operator<<(std::ostream& os, const Neuron& neuron);

  void SetOutput(double out_value) { if (!is_bias_) out_ = out_value; }
  double GetOutput() const { return out_; }

  void SetLayers(std::vector<Neuron>* prev, std::vector<Neuron>* next);
  void ForwardPropogation();

 private:
  size_t id_;
  bool is_bias_;
  std::vector<Neuron>* prev_layer_;
  std::vector<Neuron>* next_layer_;
  std::vector<double> input_weights_;
  double out_;
};

#endif  // NEURON_H
