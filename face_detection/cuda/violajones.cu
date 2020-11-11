#include <iostream>
#include <vector>
#include <cuda_runtime.h>
#include <helper_cuda.h>

#define MIN(x, y) ((x < y) ? x : y)

using namespace std;

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

__global__ void applyFeature(char *d_x_feat, char *d_y_feat, bool *d_p_feat, char *d_img, short int *d_res, char img_size, int feature_size) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;

    short int acc = 0;
    char x, y;
    for(int i = 0; i < feature_size; i++) {
        x = d_x_feat[idx * feature_size + i];
        y = d_y_feat[idx * feature_size + i];
        if(x < img_size && y < img_size)
          acc += (d_img[x * img_size + y] * d_p_feat[i]);
    }

    d_res[idx] = acc;
}

using namespace std;

int main() {
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
    
    int blocksPerGrid, threadsPerBlock;
    int N_FEATURES = 1;
    int IMG_SIZE = 5;
    int TOTAL_IMGS = 1;
    
    char h_imgs[1][5][5]  = {
        { 
            1, 1, 1, 1,
            1, 1, 1, 1,
            1, 1, 1, 1,
            1, 1, 1, 1
        }
    };

    char x_features[1][16] = {
        0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0
    };
    char y_features[1][16] = {
        0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0
    };
    bool p_features[1][16] = {
        1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1
    };
    
    char *d_x_feat;
    CHECK(cudaMalloc((void **) &d_x_feat, sizeof(x_features)));
    CHECK(cudaMemcpy(d_x_feat, x_features, N_FEATURES * 16 * sizeof(char), cudaMemcpyHostToDevice));
    
    char *d_y_feat;
    CHECK(cudaMalloc((void **) &d_y_feat, sizeof(y_features)));
    CHECK(cudaMemcpy(d_y_feat, y_features, N_FEATURES * 16 * sizeof(char), cudaMemcpyHostToDevice));
    
    bool *d_p_feat;
    CHECK(cudaMalloc((void **) &d_p_feat, sizeof(p_features)));
    CHECK(cudaMemcpy(d_p_feat, p_features, N_FEATURES * 16 * sizeof(bool), cudaMemcpyHostToDevice));
    
    char *d_img;
    CHECK(cudaMalloc((void **) &d_img, IMG_SIZE * IMG_SIZE * sizeof(char)));

    short int *d_res;
    CHECK(cudaMalloc((void **) &d_res, N_FEATURES * sizeof(short int)));
    
    threadsPerBlock = MIN(coresPerMP, N_FEATURES);
    blocksPerGrid = floor(N_FEATURES / threadsPerBlock) + 1;

    short int h_res[N_FEATURES];
    for(int i = 0; i < TOTAL_IMGS; i++) {
        CHECK(cudaMemcpy(d_img, h_imgs[i], IMG_SIZE * IMG_SIZE * sizeof(char), cudaMemcpyHostToDevice));
        applyFeature<<<blocksPerGrid, threadsPerBlock>>>(d_x_feat, d_y_feat, d_p_feat, d_img, d_res, 19, 16);
        CHECK(cudaMemcpy(h_res, d_res, N_FEATURES * sizeof(short int), cudaMemcpyDeviceToHost));

        for(int j = 0; j < N_FEATURES; j++) {
            cout << h_res[j] << " ";
        }
    }
}