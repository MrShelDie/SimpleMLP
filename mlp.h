#ifndef MLP_H
#define MLP_H

#include "neuron.h"

#include <vector>
#include <ostream>
#include <initializer_list>

class MLP {
 public:
  MLP(std::vector<size_t> layer_sizes);

  friend std::ostream& operator<<(std::ostream& os, const MLP& net);

  std::vector<double> ForwardPropogation(
      const std::vector<double>& input_values);

 private:
  std::vector<std::vector<Neuron>> neuron_matrix_;
};

#endif // MLP_H
