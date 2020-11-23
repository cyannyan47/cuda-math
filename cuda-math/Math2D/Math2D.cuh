#ifndef MATH2D_H
#define MATH2D_H
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <math.h>
#include <stdio.h>

cudaError_t Mul2DCuda(int* c, const int* a, const int* b, unsigned int sizeA, unsigned int sizeB, unsigned int widthA, unsigned int heightA);


#endif // !MATH2D_H
