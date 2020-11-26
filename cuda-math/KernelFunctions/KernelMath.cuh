#ifndef KERNELMATH_H
#define KERNELMATH_H
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

__global__ void addKernel(int* c, const int* a, const int* b, const int N);

__global__ void subKernel(int* c, const int* a, const int* b, const int N);

__global__ void mulKernel(int* c, const int* a, const int* b, const int N);

__global__ void divKernel(int* c, const int* a, const int* b, const int N);

/// <summary>
/// width is the width of the matrix
/// For a AxB, A is NxM, B is MxN
/// then width = M
/// </summary>
/// <param name="c"></param>
/// <param name="a"></param>
/// <param name="b"></param>
/// <param name="N"></param>
/// <returns></returns>
__global__ void mul2DKernel(int* c, const int* a, const int* b, int widthA, int heightA);

#endif // !KERNELMATH_H
