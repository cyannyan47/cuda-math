#include "Math2D.cuh"
#include "../KernelFunctions/KernelMath.cuh"
/*
    2D matrix configure int* c:
    For a 2d matrix c[x][y], x will be the row of the matrix, y will be the column of the matrix
    For a NxM matrix, we will represent this matrix with a 1D array of size (N * M). Note: N: row dimensions, M column dimensions
    To access the matrix at c[x][y] (accessing element at row x, column y), we use array[ ( (x-1) * N + (y-1) ]
    The '-1' means that the array start its index at 0 as opposed to the matrix starts its row and column at 1
    For example, in a 3x3 matrix:
    |   c[1][1] c[1][2] c[1][3] |       |   (0) (1) (2) |
    |   c[2][1] c[2][2] c[2][3] | --->  |   (3) (4) (5) |
    |   c[3][1] c[3][2] c[3][3] |       |   (6) (7) (8) |

    By using this type of indexing formula, we can represent any NxM matrices
    
    Therefor, the 2D multiply function will take int* a, int* b, and int* c
    with the difference is that we now need to provide the size of each matrix
*/

cudaError_t Mul2DCuda(int* c,   // 1d array representation of multiplication of matrix C = A x B
        const int* a, // 1d array representation of matrix A
        const int* b, // 1d array representation of matrix B
        unsigned int sizeA, // 1D array size of matrix A
        unsigned int sizeB, // 1D array size of matrix B
        unsigned int widthA, // the width of matrix A
        unsigned int heightA) {
    int* dev_a = 0;
    int* dev_b = 0;
    int* dev_c = 0;
	cudaError_t cudaStatus;

    // Choose which GPU to run on, change this on a multi-GPU system.
    cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
        goto Error;
    }

    // Allocate GPU buffers for three vectors (two input, one output).
    int sizeC = heightA * heightA;
    cudaStatus = cudaMalloc((void**)&dev_c, sizeC * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_a, sizeA * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_b, sizeB * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    // Copy input vectors from host memory to GPU buffers.
    cudaStatus = cudaMemcpy(dev_a, a, sizeA * sizeof(int), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    cudaStatus = cudaMemcpy(dev_b, b, sizeB * sizeof(int), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    // Setting up the blocks and grids for GPU parallelization
    dim3 threadsPerBlock(widthA, widthA);   // going to use only threads in a block
    dim3 blocksPerGrid(1, 1);   // need only 1 block, for simplicity
    if (widthA * widthA > 512) {
        printf("More than 512\n");
        threadsPerBlock.x = 512;
        threadsPerBlock.y = 512;
        blocksPerGrid.x = ceil(double(widthA) / double(threadsPerBlock.x));
        blocksPerGrid.y = ceil(double(widthA) / double(threadsPerBlock.y));
    }

    // Launch a kernel on the GPU with one thread for each element.
    mul2DKernel <<<blocksPerGrid, threadsPerBlock >>> (dev_c, dev_a, dev_b, widthA, heightA);

    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
        goto Error;
    }

    // cudaDeviceSynchronize waits for the kernel to finish, and returns
    // any errors encountered during the launch.
    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
        goto Error;
    }

    // Copy output vector from GPU buffer to host memory.
    cudaStatus = cudaMemcpy(c, dev_c, sizeC * sizeof(int), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }


Error:
    cudaFree(dev_c);
    cudaFree(dev_a);
    cudaFree(dev_b);

    return cudaStatus;
}
