#pragma once
#ifndef MATH1D_H
#define MATH1D_H

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <math.h>
#include <stdio.h>

cudaError_t Add1DCuda(int* c, const int* a, const int* b, unsigned int size);

cudaError_t Sub1DCuda(int* c, const int* a, const int* b, unsigned int size);

cudaError_t Mul1DCuda(int* c, const int* a, const int* b, unsigned int size);

/// <summary>
/// Assuming there are no 0 in b array
/// </summary>
/// <param name="c"></param>
/// <param name="a"></param>
/// <param name="b"></param>
/// <param name="size"></param>
/// <returns></returns>
cudaError_t Div1DCuda(int* c, const int* a, const int* b, unsigned int size);


#endif