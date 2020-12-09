#include <iostream>
#include <vector>
#include <iostream>
#include <vector>
#include <cuda_runtime.h>
#include <helper_cuda.h>

#include "../src/F5/Boosting.cpp"
#include "../src/FileReader.cpp"
#include "../src/IntegralImage.cpp"
#include "../src/FaceDetector.cpp"
#include "../src/common.h"

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

int evaluate(FaceDetector *fd, vector<pair<Image, int>> *trainingData)
{
    int correct = 0, allNegatives = 0, allPositives = 0, trueNegatives = 0, falseNegatives = 0, truePositives = 0, falsePositives = 0, prediction = -1;
    double classification_time = 0;

    for (pair<Image, int> data : *trainingData)
    {
        if (data.second == 1)
            allPositives++;
        else
            allNegatives++;

        prediction = (*fd).classify(data.first);
        if (prediction == 1 && data.second == 0)
            falsePositives++;
        else if (prediction == 0 && data.second == 1)
            falseNegatives++;

        correct += (prediction == data.second ? 1 : 0);
    }

    printf("False Positive Rate: %d/%d (%f)", falsePositives, allNegatives, falsePositives / allNegatives);
    printf("False Negative Rate: %d/%d (%f)", falseNegatives, allPositives, falseNegatives / allPositives);
    printf("Accuracy: %d/%d (%f)", correct, trainingData->size(), correct / trainingData->size());
    printf("Average Classification Time: %f", classification_time / trainingData->size());
}

int loadSamples(string path, vector<pair<Image, int>> *trainingData, int init, int end, int classId, int limit)
{
    FileReader fr(path, init, end);

    vector<vector<unsigned char>> sample;
    int count = 0;

    while (fr.remainingSamples())
    {
        int res = fr.getSample(&sample, count == 0);

        if (!res)
        {
            cout << "Error opening the file";
            continue;
        }

        Image img = Image(sample, sample.size());

        (*trainingData).push_back(make_pair(Image(sample, sample.size()), classId));

        if (++count == limit)
            break;
    }

    return count;
}

int loadSamples(string path, vector<pair<Image, int>> *trainingData, int classId, int limit)
{
    FileReader fr(path);

    vector<vector<unsigned char>> sample;
    int count = 0;

    while (fr.remainingSamples())
    {
        int res = fr.getSample(&sample, count == 0);

        if (!res)
        {
            cout << "Error opening the file";
            continue;
        }

        Image img = Image(sample, sample.size());

        (*trainingData).push_back(make_pair(Image(sample, sample.size()), classId));

        if (++count == limit)
            break;
    }

    return count;
}

int loadSamples(string path, vector<pair<Image, int>> *trainingData, int classId)
{
    return loadSamples(path, trainingData, classId, -1);
}

int main(int argc, char *argv[]) {
    if(argc < 2) {
        printf("Wrong arguments!\n");
        return -1;
    }

    freopen("output.csv", "a+", stdout);

    int THREADS = -1, BLOCKS = -1;
    char *RUNNING_TYPE = argv[1];
    
    printf("%s, ", RUNNING_TYPE);

    if(argv[2] != NULL)
        sscanf(argv[2], "%d", &THREADS);

    bool _auto = (THREADS == -1);

    int vb;
    bool verbose = false;
    if(argv[3] != NULL) {
        sscanf(argv[3], "%d", &vb);
        verbose = (vb == 1);
    }    
    
    int deviceCount = 0;
    CHECK(cudaGetDeviceCount(&deviceCount));
  
    if (deviceCount == 0)
    {
      printf("There are no available device(s) that support CUDA\n");
      return;
    }
  
    if (verbose)
      printf("Detected %d CUDA Capable device(s)\n", deviceCount);
  
    cudaDeviceProp deviceProp;
    cudaSetDevice(0);
    cudaGetDeviceProperties(&deviceProp, 0);

    size_t cudaFreeMemory, cudaTotalMemory;
    cudaMemGetInfo(&cudaFreeMemory, &cudaTotalMemory);
    
    int coresPerMP = _ConvertSMVer2Cores(deviceProp.major, deviceProp.minor);
    int multiProcessors = deviceProp.multiProcessorCount;

    float cudaMemoryOccupation = 0.8;
    
    RunningType cd = { RUNNING_TYPE, THREADS, BLOCKS, coresPerMP, multiProcessors, cudaMemoryOccupation, _auto };
    
    if (verbose) {
        cout << "GPU " << " memory: free=" << cudaFreeMemory << ", total=" << cudaTotalMemory << endl;
        printf("%d Multiprocessors, %d CUDA Cores/MP | %d CUDA Cores\nMaximum number of threads per block: %d\n",
            deviceProp.multiProcessorCount,
            _ConvertSMVer2Cores(deviceProp.major, deviceProp.minor),
            _ConvertSMVer2Cores(deviceProp.major, deviceProp.minor) * deviceProp.multiProcessorCount,
                deviceProp.maxThreadsPerBlock);
    }

    vector<pair<Image, int>> trainingData;

    int n = 1000;

    int positiveSamples = loadSamples("./img/train/face/", &trainingData, 0, 3, 1, n);

    for(pair<Image, int> p : trainingData) {
        cout << p.second << endl;
    }

    int negativeSamples = loadSamples("./img/train/non-face/", &trainingData, 0, n);

    bool useF5 = true;

    // if (useF5) {
    //   trainF5(trainingData, cd, verbose);
    // } else {
    //   FaceDetector fd = FaceDetector(10);

    //   fd.train(trainingData, positiveSamples, negativeSamples);
    //   evaluate(&fd, &trainingData);
    // }
}