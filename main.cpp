#include "mlp.h"

#include <iostream>

int main()
{
  MLP mlp = {2, 2, 2};
  std::cout << mlp << std::endl;

  auto v = mlp.ForwardPropogation({1, 0});

  for (auto val : v) {
    std::cout << val << ' ';
  }

  std::cout << std::endl;

  return 0;
}
