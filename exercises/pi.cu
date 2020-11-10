%%writefile src/blur-effect.cu 

#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <sys/time.h>
#include <cuda_runtime.h>
#include <helper_cuda.h>
#include <string.h>

#define ITERATIONS 1000000000

#define MIN(x, y) ((x < y) ? x : y)
#define CHECK(call)                                                            \
{                                                                              \
    const cudaError_t error = call;                                            \
    if (error != cudaSuccess)                                                  \
    {                                                                          \
        fprintf(stderr, "Error: %s:%d, ", __FILE__, __LINE__);                 \
        fprintf(stderr, "code: %d, reason: %s\n", error,                       \
                cudaGetErrorString(error));                                    \
    }                                                                          \
}

using namespace std;

#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 600
#else
__device__ double atomicAdd(double* a, double b) { return b; }
#endif

__global__ void calcPi(float *pi, long int chunkSize, long int iterations) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    
    long int start = idx * chunkSize;
    long int end = MIN(start + chunkSize, iterations);

    double partialPi = 0;

    for(long long int i = start; i < end; i++)
      partialPi += (i % 2 ? -1 : 1) * (4.0 / (2.0 * i + 1.0));

    pi[idx] = partialPi;
}

__global__ void red(float *pi) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    unsigned int tid = threadIdx.x;

    for(unsigned int s=1; s < blockDim.x; s *= 2) {
        if (tid % (2*s) == 0) {
            pi[tid] += pi[tid + s];
        }
        __syncthreads();
    }
}

int main(int argc, char *argv[]) {
    if(argc < 3) {
        printf("Wrong arguments!\n");
        return -1;
    }

    int BLOCKS = -1;
    if(argv[1] != NULL)
        sscanf(argv[1], "%d", &BLOCKS);
    
    int THREADS = -1;
    if(argv[2] != NULL)
        sscanf(argv[2], "%d", &THREADS);

    struct timeval after, before, result;
    gettimeofday(&before, NULL);

    bool verbose = true;
    
    int deviceCount = 0;
    CHECK(cudaGetDeviceCount(&deviceCount));

    if (deviceCount == 0) {
        printf("There are no available device(s) that support CUDA\n");
        return -1;
    }

    if(verbose) printf("Detected %d CUDA Capable device(s)\n", deviceCount);

    cudaSetDevice(0);
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, 0);

    int coresPerMP = _ConvertSMVer2Cores(deviceProp.major, deviceProp.minor);
    int multiProcessors = deviceProp.multiProcessorCount;

    if(verbose)
    printf("%d Multiprocessors, %d CUDA Cores/MP | %d CUDA Cores\nMaximum number of threads per block: %d\n",
            deviceProp.multiProcessorCount,
            _ConvertSMVer2Cores(deviceProp.major, deviceProp.minor),
            _ConvertSMVer2Cores(deviceProp.major, deviceProp.minor) * deviceProp.multiProcessorCount,
            deviceProp.maxThreadsPerBlock);

    int blocksPerGrid = BLOCKS, threadsPerBlock = THREADS;
    int totalThreads = (threadsPerBlock * blocksPerGrid);

    if(verbose)
      printf("Choosen %d blocks with %d threads", blocksPerGrid, threadsPerBlock);

    float *pi;
    CHECK(cudaMalloc((void **) &pi, totalThreads * sizeof(float)));
    CHECK(cudaMemset(pi, 0, totalThreads * sizeof(float)));

    long int chunkSize = ITERATIONS / totalThreads;

    calcPi<<<blocksPerGrid, threadsPerBlock>>>(pi, chunkSize, ITERATIONS);
    red<<<blocksPerGrid, threadsPerBlock>>>(pi);

    float res;
	  cudaMemcpy(&res, pi, sizeof(float), cudaMemcpyDeviceToHost);

    printf("TOTAL: %f", res);

    CHECK(cudaFree(pi));
    
    gettimeofday(&after, NULL);
    timersub(&after, &before, &result);

    if(verbose) printf("\nTime elapsed: %ld.%06ld\n", (long int) result.tv_sec, (long int) result.tv_usec);
    else
        printf("%ld.%06ld\n", (long int)result.tv_sec, (long int)result.tv_usec);
    
    return 0;
}