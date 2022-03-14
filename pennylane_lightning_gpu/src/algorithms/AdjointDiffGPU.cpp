#include "AdjointDiffGPU.hpp"

// explicit instantiation
template class Pennylane::Algorithms::AdjointJacobianGPU<float>;
template class Pennylane::Algorithms::AdjointJacobianGPU<double>;