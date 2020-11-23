#include "KernelMath.cuh"
#include <stdio.h>

__global__ void addKernel(int* c, const int* a, const int* b)
{
    int i = threadIdx.x;
    c[i] = a[i] + b[i];
}

__global__ void subKernel(int* c, const int* a, const int* b)
{
    int i = threadIdx.x;
    c[i] = a[i] - b[i];
}

__global__ void mulKernel(int* c, const int* a, const int* b)
{
    int i = threadIdx.x;
    c[i] = a[i] * b[i];
}

__global__ void divKernel(int* c, const int* a, const int* b)
{
    int i = threadIdx.x;
    c[i] = a[i] / b[i];
}

__global__ void mul2DKernel(int* c, const int* a, const int* b, int widthA, int heightA) 
{
    /*int ROW = blockIdx.y * blockDim.y + threadIdx.y;
    int COL = blockIdx.x * blockDim.x + threadIdx.x;*/

    int aROW = threadIdx.y;
    int bCOL = threadIdx.x;

    // Checking if the ROW/COL exceed the actual number of ROW/COL
    // widthA = heightB
    // heightA = widthB
    if (aROW < widthA && bCOL < heightA) {
        int dotProduct = 0;
        for (int k = 0; k < widthA; k++) {
            printf("Row %d, Col %d, loop num %d, A index %d, B index %d\n", aROW, bCOL, k, (aROW * widthA + k), (k * heightA + bCOL));
            dotProduct += a[aROW * widthA + k] * b[k * heightA + bCOL];  // widthA = heightB     
        }
        c[aROW * heightA + bCOL] = dotProduct;
        printf("Output value at row %d, col %d is %d\n", aROW, bCOL, c[aROW * heightA + bCOL]);
    }
}