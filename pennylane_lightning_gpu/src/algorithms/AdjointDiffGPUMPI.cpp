#include "AdjointDiffGPU.hpp"
#include "StateVectorCudaMPI.hpp"

// explicit instantiation
template class Pennylane::Algorithms::AdjointJacobianGPUMPI<float,
                                                            StateVectorCudaMPI>;
template class Pennylane::Algorithms::AdjointJacobianGPUMPI<double,
                                                            StateVectorCudaMPI>;