
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include "./Math1D/Math1D.cuh"
#include "./Math2D/Math2D.cuh"

int main()
{
    const int arraySize = 6;
    const int a[arraySize] = { 1, 2, 3, 4, 5, 6 };
    const int b[arraySize] = { 10, 20, 30, 40, 50, 0 };
    int c[arraySize] = { 0 };

    // Add vectors in parallel.
    cudaError_t cudaStatus = Add1DCuda(c, a, b, arraySize);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "addWithCuda failed!");
        return 1;
    }

    printf("{1,2,3,4,5} + {10,20,30,40,50} = {%d,%d,%d,%d,%d}\n",
        c[0], c[1], c[2], c[3], c[4]);

    cudaStatus = Sub1DCuda(c, a, b, arraySize);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "subWithCuda failed!");
        return 1;
    }
    printf("{1,2,3,4,5} - {10,20,30,40,50} = {%d,%d,%d,%d,%d}\n",
        c[0], c[1], c[2], c[3], c[4]);

    cudaStatus = Mul1DCuda(c, a, b, arraySize);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "mulWithCuda failed!");
        return 1;
    }
    printf("{1,2,3,4,5} * {10,20,30,40,50} = {%d,%d,%d,%d,%d}\n",
        c[0], c[1], c[2], c[3], c[4]);


    
    const int widthA = 3;
    const int heightA = 2;

    const int sizeMatrixA = widthA * heightA;
    const int sizeMatrixB = widthA * heightA;
    const int sizeMatrixC = heightA * heightA;

    const int matrixA[sizeMatrixA] = { 1, 2, 3, 4, 5, 6};
    const int matrixB[sizeMatrixB] = { 10, 11, 20, 21, 30, 31 };
    int matrixC[sizeMatrixC] = { 0 };

    cudaStatus = Mul2DCuda(matrixC, matrixA, matrixB, sizeMatrixA, sizeMatrixB, widthA, heightA);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "divWithCuda failed!");
        return 1;
    }
    //printf("{1,2,3,4,5,6} 2D* {10,11,20,21,30,31} = {%d,%d,%d,%d}\n",
    //    matrixC[0], matrixC[1], matrixC[2], matrixC[3]);

    // Checking every b values to see if there's any 0
    for (int i = 0; i <= arraySize; i++) {
        if (b[i] == 0) {
            fprintf(stderr, "One of the b values is 0");
            return 1;
        }
    }
    cudaStatus = Div1DCuda(c, a, b, arraySize);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "divWithCuda failed!");
        return 1;
    }
    printf("{1,2,3,4,5} / {10,20,30,40,50} = {%d,%d,%d,%d,%d}\n",
        c[0], c[1], c[2], c[3], c[4]);

    // cudaDeviceReset must be called before exiting in order for profiling and
    // tracing tools such as Nsight and Visual Profiler to show complete traces.
    cudaStatus = cudaDeviceReset();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceReset failed!");
        return 1;
    }

    return 0;
}