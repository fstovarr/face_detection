#ifndef BOOSTING_H
#define BOOSTING_H
#include <vector>
#include <algorithm>
#include <chrono>
#include <limits>
#include <stdlib.h>
#include <math.h>
#include <fstream>
#include <random>
#include <sys/time.h>
#include <omp.h>

#include "Features.cpp"
#include "Metrics.cpp"
#include "../Image.cpp"
#include "../nlohmann/json.hpp"
#include "../common.h"

#define MIN(x, y) ((x < y) ? x : y)

#define CHECK(call)                                          \
  {                                                          \
    const cudaError_t error = call;                          \
    if (error != cudaSuccess)                                \
    {                                                        \
      fprintf(stderr, "Error: %s:%d, ", __FILE__, __LINE__); \
      fprintf(stderr, "code: %d, reason: %s\n", error,       \
              cudaGetErrorString(error));                    \
    }                                                        \
  }

using json = nlohmann::json;

typedef vector<vector<int>> matrix;

struct RunningType
{
  char *type;
  int threads;
  int blocks;
  int coresPerMP;
  int multiProcessors;
  float cudaMemoryOccupation;
  bool _auto;
};

__global__ void applyFeature(char *d_x_feat, char *d_y_feat, bool *d_p_feat, int *d_img, unsigned short int *d_res, const int MEMORY_PER_IMAGE, char IMG_SIZE, int FEATURES_SIZE)
{
  int idx = blockDim.x * blockIdx.x + threadIdx.x;
  int currentImage = idx / FEATURES_SIZE;
  idx = idx % FEATURES_SIZE;
  const int single_feature_size = 16;

  unsigned short int acc = 0;
  char x, y;
  for (int i = 0; i < single_feature_size; i++)
  {
    x = d_x_feat[idx * single_feature_size + i];
    y = d_y_feat[idx * single_feature_size + i];
    if (x < IMG_SIZE && y < IMG_SIZE)
      acc += (d_img[currentImage * IMG_SIZE * IMG_SIZE + x * IMG_SIZE + y] * (d_p_feat[i] ? 1 : -1));
  }

  d_res[currentImage * FEATURES_SIZE + idx] = acc;
}

void applyFeatures(int *d_img, unsigned short int *d_res, char *d_x_feat, char *d_y_feat, bool *d_p_feat, vector<vector<unsigned short int>> &z, const int &batch, const int FEATURES_SIZE, const int &MEMORY_PER_IMAGE, const int &IMG_SIZE, const int &D_RES_SIZE, RunningType &runningType)
{
  const int TOTAL_THREADS = batch * FEATURES_SIZE;

  // TODO Reuse this array passing it through an pointer
  unsigned short int h_res[batch * FEATURES_SIZE];

  int blocksPerGrid, threadsPerBlock;
  if (runningType._auto)
  {
    threadsPerBlock = MIN(runningType.coresPerMP, TOTAL_THREADS);
    blocksPerGrid = floor(TOTAL_THREADS / threadsPerBlock) + 1;
  }
  else
  {
    threadsPerBlock = runningType.threads;
    blocksPerGrid = floor(TOTAL_THREADS / threadsPerBlock) + 1;
    runningType.blocks = blocksPerGrid;
  }

  applyFeature<<<blocksPerGrid, threadsPerBlock>>>(d_x_feat, d_y_feat, d_p_feat, d_img, d_res, MEMORY_PER_IMAGE, IMG_SIZE, FEATURES_SIZE);

  CHECK(cudaMemcpy(h_res, d_res, FEATURES_SIZE * batch * sizeof(unsigned short int), cudaMemcpyDeviceToHost));

  for (int i = 0; i < batch; i++)
    for (int j = 0; j < FEATURES_SIZE; j++)
      z[i][j] = h_res[i * FEATURES_SIZE + j];
}

void applyFeatures(matrix const &x, vector<Feature> const &features, vector<vector<unsigned short int>> &z, const int &BATCH_SIZE, RunningType &rt, bool use_omp)
{
  size_t n_features = features.size();

  int threads = omp_get_num_threads();

  if (rt._auto == false)
  {
    threads = rt.threads;
  }

  for (int i = 0; i < BATCH_SIZE; i++)
#pragma omp parallel for num_threads(threads) if (use_omp)
    for (size_t j = 0; j < n_features; ++j)
    {
      z[i][j] = features[j](x);
    }
}

void copyFeaturesToCuda(vector<Feature> &features, char *d_x_feat, char *d_y_feat, bool *d_p_feat)
{
  int N_FEATURES = features.size();

  char x_features[N_FEATURES][16];
  char y_features[N_FEATURES][16];
  bool p_features[N_FEATURES][16];

  vector<int> vx;
  vector<int> vy;
  vector<int> vp;

  for (int i = 0; i < N_FEATURES; i++)
  {
    vx = features[i].getCoordsX();
    vy = features[i].getCoordsY();
    vp = features[i].getCoefs();

    for (int j = 0; j < 16; j++)
    {
      x_features[i][j] = (char)(j >= vx.size() ? 0 : vx[j]);
      y_features[i][j] = (char)(j >= vy.size() ? 0 : vy[j]);
      p_features[i][j] = (bool)(j >= vp.size() ? 0 : vp[j] > 0);
    }
  }

  CHECK(cudaMemcpy(d_x_feat, x_features, N_FEATURES * 16 * sizeof(char), cudaMemcpyHostToDevice));
  CHECK(cudaMemcpy(d_y_feat, y_features, N_FEATURES * 16 * sizeof(char), cudaMemcpyHostToDevice));
  CHECK(cudaMemcpy(d_p_feat, p_features, N_FEATURES * 16 * sizeof(bool), cudaMemcpyHostToDevice));
}

void freeFeaturesInCuda(tuple<char *, char *, bool *> cudaPointers)
{
  CHECK(cudaFree(get<0>(cudaPointers)));
  CHECK(cudaFree(get<1>(cudaPointers)));
  CHECK(cudaFree(get<2>(cudaPointers)));
}

void copyIntegralImagesToCuda(vector<matrix> const &xis, int *h_imgs, int *d_img, const int &start, const int &batch, const int &MEMORY_PER_IMAGE, const int &IMG_SIZE)
{
  for (size_t i = 0; i < batch; i++)
    for (int mm = 0; mm < IMG_SIZE; mm++)
      for (int nn = 0; nn < IMG_SIZE; nn++)
        h_imgs[i * IMG_SIZE * IMG_SIZE + mm * IMG_SIZE + nn] = (int)xis[i][mm][nn];

  CHECK(cudaMemcpy(d_img, h_imgs, batch * IMG_SIZE * IMG_SIZE * sizeof(int), cudaMemcpyHostToDevice));
}

vector<float> normalizeWeights(vector<float> &v)
{
  float S = 0.0;
  for (float x : v)
    S += x;
  for (size_t i = 0; i < v.size(); ++i)
  {
    v[i] /= S;
  }
  return v;
}

int sign(float x)
{
  if (x)
    return 1.0;
  return -1.0;
}

struct ThresholdPolarity
{
  float threshold;
  int polarity;
};

struct ClassifierResult
{
  float threshold;
  int polarity;
  float classification_error;
  Feature f;
};

float weakClassifierZ(float const &theta, int const &polarity, float const &z)
{
  return (sign((polarity * theta) - (polarity * z)) + 1) / 2;
}

struct WeakClassifierF5
{
  float threshold;
  int polarity;
  float alpha;
  Feature f;

  template <typename T>
  float operator()(vector<vector<T>> const &x) const
  {
    float theta = threshold;
    float z = f(x);
    return weakClassifierZ(theta, polarity, z);
  }

  json to_json()
  {
    json j;
    j["threshold"] = threshold;
    j["polarity"] = polarity;
    j["alpha"] = alpha;
    j["feature"] = {
        {"type", f.getType()},
        {"x", f.getX()},
        {"y", f.getX()},
        {"width", f.getWidth()},
        {"height", f.getHeight()},
    };
    return j;
  }
};

ostream &operator<<(ostream &os, WeakClassifierF5 const &clf)
{
  // cout << "WeakClassifierF5(threshold=" << clf.threshold << ", polarity=" << clf.polarity << ", alpha=" << clf.alpha << " f=" << clf.f << ")";
  return os;
}

template <typename T>
int strongClassifier(vector<vector<T>> const &x, vector<WeakClassifierF5> const &weak_classifiers)
{
  float sum_hypotheses = 0;
  float sum_alphas = 0;
  for (auto clf : weak_classifiers)
  {
    sum_hypotheses += clf.alpha * clf(x);
    sum_alphas += clf.alpha;
  }
  if (sum_hypotheses >= .5 * sum_alphas)
    return 1;
  return 0;
}

struct RunningSums
{
  float t_minus, t_plus;
  vector<float> s_minuses, s_pluses;
};

RunningSums buildRunningSums(vector<int> const &ys, vector<float> const &ws)
{
  size_t N = ys.size();
  float s_minus = 0, s_plus = 0;
  float t_minus = 0, t_plus = 0;
  vector<float> s_minuses(N), s_pluses(N);

  for (size_t i = 0; i < ys.size(); ++i)
  {
    float y = ys[i], w = ws[i];
    if (y < .5)
    {
      s_minus += w;
      t_minus += w;
    }
    else
    {
      s_plus += w;
      t_plus += w;
    }
    s_minuses[i] = s_minus;
    s_pluses[i] = s_plus;
  }
  return RunningSums{t_minus, t_plus, s_minuses, s_pluses};
}

ThresholdPolarity find_best_threshold(vector<float> const &zs, RunningSums const &rs)
{
  float min_e = numeric_limits<float>::max();
  float min_z = 0;
  int polarity = 0;
  float t_minus = rs.t_minus, t_plus = rs.t_plus;
  vector<float> s_minuses = rs.s_minuses, s_pluses = rs.s_pluses;

  for (size_t i = 0; i < zs.size(); ++i)
  {
    float z = zs[i], s_m = s_minuses[i], s_p = s_pluses[i];
    float error_1 = s_p + (t_minus - s_m);
    float error_2 = s_m + (t_plus - s_p);
    if (error_1 < min_e)
    {
      min_e = error_1;
      min_z = z;
      polarity = -1;
    }
    else if (error_2 < min_e)
    {
      min_e = error_2;
      min_z = z;
      polarity = 1;
    }
  }
  return ThresholdPolarity{min_z, polarity};
}

typedef pair<pair<float, float>, float> f3;
bool cmp(f3 const &lh, f3 const &rh) { return lh.second < rh.second; }

template <typename T>
ThresholdPolarity determineThresholdPolarity(vector<int> const &ys, vector<float> const &ws, vector<T> const &zs)
{
  size_t N = ys.size();
  // sort according to zs
  vector<f3> triplet(N);
  for (size_t i = 0; i < ys.size(); ++i)
    triplet[i] = {{ys[i], ws[i]}, zs[i]};

  sort(triplet.begin(), triplet.end(), cmp);

  vector<int> ys_sorted(N);
  vector<float> ws_sorted(N), zs_sorted(N);
  for (size_t i = 0; i < ys.size(); ++i)
  {
    float y = triplet[i].first.first, w = triplet[i].first.second;
    ys_sorted[i] = y;
    ws_sorted[i] = w;
    zs_sorted[i] = triplet[i].second;
  }

  RunningSums rs = buildRunningSums(ys_sorted, ws_sorted);

  return find_best_threshold(zs_sorted, rs);
}

template <typename T>
ClassifierResult trainWeakF5(Feature const &f, vector<T> const &zs, vector<int> const &ys, vector<float> const &ws)
{

  // determine best threshold
  ThresholdPolarity result = determineThresholdPolarity(ys, ws, zs);

  float classification_error = .0;
  for (size_t i = 0; i < zs.size(); ++i)
  {
    float h = weakClassifierZ(result.threshold, result.polarity, zs[i]);
    classification_error += ws[i] * abs(h - ys[i]);
  }

  return ClassifierResult{result.threshold, result.polarity, classification_error, f};
}

const int STATUS_EVERY = 2000;
const float KEEP_PROBABILITY = 1. / 4.;

typedef pair<vector<WeakClassifierF5>, vector<float>> TrainingResult;
TrainingResult buildWeakClassifiers(
    string prefix, int num_features, vector<matrix> const &xis, vector<int> ys, vector<Feature> &features, vector<float> &ws, RunningType &runningType, bool verbose)
{
  mt19937_64 rng;
  rng.seed(42);
  uniform_real_distribution<double> unif(0, 1);

  // initialize weights
  if (ws.empty())
  {
    int m = 0, l = 0;
    for (size_t i = 0; i < ys.size(); ++i)
    {
      if (ys[i] > .5)
      {
        l++;
      }
      else
      {
        m++;
      }
    }
    ws.resize(ys.size());
    for (size_t i = 0; i < ys.size(); ++i)
    {
      if (ys[i] > .5)
      {
        ws[i] = 1. / (2. * l);
      }
      else
      {
        ws[i] = 1. / (2. * m);
      }
    }
  }

  double avgTime = 0.0;
  auto start = chrono::high_resolution_clock::now();

  struct timeval after, before, result;
  gettimeofday(&before, NULL);

  long int BATCH_SIZE = 1;
  const int IMG_SIZE = xis[0].size();
  const int MEMORY_PER_IMAGE = IMG_SIZE * IMG_SIZE * sizeof(int);
  const int FEATURES_SIZE = features.size();

  // CUDA DEFINITIONS
  unsigned short int *d_res;
  char *d_x_feat, *d_y_feat;
  bool *d_p_feat;
  int *d_img;

  int D_RES_SIZE = BATCH_SIZE * FEATURES_SIZE * sizeof(unsigned short int);

  if (*(runningType.type) == 'C')
  {
    size_t memoryLimit;
    size_t cudaFreeMemory, cudaTotalMemory;
    const size_t FEATURES_MEMORY = FEATURES_SIZE * 16 * sizeof(char);
    const size_t IMG_SIZE_SQR = IMG_SIZE * IMG_SIZE;

    cudaMemGetInfo(&cudaFreeMemory, &cudaTotalMemory);
    memoryLimit = (size_t)(cudaFreeMemory * runningType.cudaMemoryOccupation);

    BATCH_SIZE = (long int)(memoryLimit - 3.0 * FEATURES_MEMORY) / (IMG_SIZE_SQR * sizeof(int) + IMG_SIZE_SQR * FEATURES_SIZE * sizeof(unsigned short int));

    if (BATCH_SIZE < 1)
      throw "No space left in device";

    D_RES_SIZE = BATCH_SIZE * FEATURES_SIZE * sizeof(unsigned short int);

    CHECK(cudaMalloc((void **)&d_res, D_RES_SIZE));
    CHECK(cudaMalloc((void **)&d_img, MEMORY_PER_IMAGE * BATCH_SIZE));

    CHECK(cudaMalloc((void **)&d_x_feat, FEATURES_MEMORY));
    CHECK(cudaMalloc((void **)&d_y_feat, FEATURES_MEMORY));
    CHECK(cudaMalloc((void **)&d_p_feat, FEATURES_MEMORY));

    copyFeaturesToCuda(features, d_x_feat, d_y_feat, d_p_feat);
  }

  vector<vector<int>> zs(FEATURES_SIZE, vector<int>(xis.size(), 0));

  vector<vector<unsigned short int>> z(BATCH_SIZE, vector<unsigned short int>(FEATURES_SIZE, (unsigned short int)0));

  int batch;

  int h_imgs[BATCH_SIZE * IMG_SIZE * IMG_SIZE];
  unsigned short int h_res[batch * FEATURES_SIZE];

  for (int i = 0; i < xis.size(); i += BATCH_SIZE)
  {
    if (verbose && i % 1000 == 0)
      cout << i << "\n";

    batch = MIN(xis.size() - i, BATCH_SIZE);

    if (*(runningType.type) == 'C')
    {
      copyIntegralImagesToCuda(xis, h_imgs, d_img, i, batch, MEMORY_PER_IMAGE, IMG_SIZE);
      applyFeatures(d_img, d_res, d_x_feat, d_y_feat, d_p_feat, z, batch, FEATURES_SIZE, MEMORY_PER_IMAGE, IMG_SIZE, D_RES_SIZE, runningType);
    }
    else
      applyFeatures(xis[i], features, z, batch, runningType, (*(runningType.type) == 'O'));

    // for (int ii = 0; ii < batch; ii++)
    //   for (size_t j = 0; j < features.size(); ++j)
    //     zs[j][i + ii] = z[ii][j];
  }

  if ((*(runningType.type) == 'C'))
  {
    CHECK(cudaFree(d_x_feat));
    CHECK(cudaFree(d_y_feat));
    CHECK(cudaFree(d_p_feat));
    CHECK(cudaFree(d_img));
    CHECK(cudaFree(d_res));
  }

  gettimeofday(&after, NULL);
  timersub(&after, &before, &result);

  auto stop = chrono::high_resolution_clock::now();
  auto duration = chrono::duration_cast<chrono::duration<double>>(stop - start);
  avgTime += duration.count();

  cout << (long int)result.tv_sec << "." << (long int)result.tv_usec << "," << avgTime << "," << runningType.threads << "," << runningType.blocks << "," << runningType.coresPerMP << "," << runningType.multiProcessors << "\n";

  auto total_start = chrono::high_resolution_clock::now();

  vector<WeakClassifierF5> weak_classifiers;
  // for (int f_i = 0; f_i < num_features; ++f_i)
  // {
  //   if (verbose)
  //     cout << "Building weak classifier " << f_i + 1 << "/" << num_features << "...\n";
  //   auto start = chrono::high_resolution_clock::now();

  //   ws = normalizeWeights(ws);

  //   int status_counter = STATUS_EVERY;

  //   ClassifierResult best{0, 0, numeric_limits<float>::max(), features[0]};
  //   bool improved = false;

  //   //vector<vector<int>>

  //   // Select best weak classifier for this round
  //   for (size_t i = 0; i < features.size(); ++i)
  //   {
  //     status_counter -= 1;
  //     improved = false;

  //     double skip_probability = unif(rng);
  //     if (skip_probability > KEEP_PROBABILITY)
  //     {
  //       if (status_counter == 0)
  //       {
  //         auto stop = chrono::high_resolution_clock::now();
  //         auto duration = chrono::duration_cast<chrono::duration<double>>(stop - start);
  //         auto total_duration = chrono::duration_cast<chrono::duration<double>>(stop - total_start);
  //         status_counter = STATUS_EVERY;
  //         if (verbose)
  //           cout << f_i + 1 << "/" << num_features << " (" << total_duration.count() << ")s "
  //                << " (" << duration.count() << "s in this stage)"
  //                << " "
  //                << 100. * i / features.size() << "% evaluated "
  //                << "...\n";
  //       }
  //       continue;
  //     }

  //     ClassifierResult result = trainWeakF5(features[i], zs[i], ys, ws);
  //     if (result.classification_error < best.classification_error)
  //     {
  //       improved = true;
  //       best = result;
  //     }

  //     // Print status every couple of iterations
  //     if (improved || status_counter == 0)
  //     {
  //       auto stop = chrono::high_resolution_clock::now();
  //       auto duration = chrono::duration_cast<chrono::duration<double>>(stop - start);
  //       auto total_duration = chrono::duration_cast<chrono::duration<double>>(stop - total_start);
  //       status_counter = STATUS_EVERY;
  //       if (improved)
  //       {
  //         if (verbose)
  //           cout << f_i + 1 << "/" << num_features << " (" << total_duration.count() << ")s "
  //                << " (" << duration.count() << "s in this stage)"
  //                << " "
  //                << 100. * i / features.size() << "% evaluated "
  //                << " Classification error improved to " << best.classification_error << " using " << best.f << "...\n";
  //       }
  //       else
  //       {
  //         if (verbose)
  //           cout << f_i + 1 << "/" << num_features << " (" << total_duration.count() << ")s "
  //                << " (" << duration.count() << "s in this stage)"
  //                << " "
  //                << 100. * i / features.size() << "% evaluated "
  //                << "...\n";
  //       }
  //     }
  //   }

  //   // After the best classifier was found, determine alpha
  //   float beta = best.classification_error / (1 - best.classification_error);
  //   float alpha = log(1. / beta);

  //   // Build the weak classifier
  //   WeakClassifierF5 clf{best.threshold, best.polarity, alpha, best.f};

  //   // Update the weights for misclassified examples

  //   for (size_t i = 0; i < xis.size(); ++i)
  //   {
  //     float h = clf(xis[i]);
  //     float e = abs(h - ys[i]);
  //     ws[i] = pow(beta, 1 - e);
  //   }

  //   // Register this weak classifier
  //   weak_classifiers.push_back(clf);
  // }
  // if (verbose)
  //   cout << "Done building " << num_features << " classifiers \n";
  return {weak_classifiers, ws};
}

void writeWeakClassifiers(string prefix, vector<WeakClassifierF5> const &clf_s)
{
  ofstream o(prefix + "clf_s.jsonl");
  for (auto clf : clf_s)
  {
    o << clf.to_json() << "\n";
  }
}

vector<WeakClassifierF5> readWeakClassifiers(string prefix, int num_features)
{
  ifstream in(prefix + "clf_s.jsonl");
  vector<WeakClassifierF5> v;
  json j;
  for (int i = 0; i < num_features; ++i)
  {
    in >> j;
    float threshold = j["threshold"];
    int polarity = j["polarity"];
    float alpha = j["alpha"];
    int x = j["feature"]["x"];
    int y = j["feature"]["y"];
    int width = j["feature"]["width"];
    int height = j["feature"]["height"];
    string type = j["feature"]["type"];

    WeakClassifierF5 clf{
        threshold,
        polarity,
        alpha,
        constructFeature(x, y, width, height, type)};
    v.push_back(clf);
  }
  return v;
}

void trainF5(vector<pair<Image, int>> const &trainingData, RunningType &cuda, bool verbose)
{
  size_t N = trainingData.size();
  vector<matrix> X(N);
  vector<int> y(N);

  for (size_t i = 0; i < N; ++i)
  {
    X[i] = to_integral(trainingData[i].first.getIntImage());
    y[i] = trainingData[i].second;
  }

  int window_size = X[0].size();
  vector<Feature> features = getFeatures(window_size - 1);
  if (verbose)
  {
    cout << "Using " << features.size() << " features\n";
    cout << "Using " << N << " train samples\n";
  }

  vector<float> ws;

  vector<WeakClassifierF5> weak_classifiers;

  string prefix = "f1";
  int num_features = 2;

  bool train = true;
  if (train)
  {
    TrainingResult tr = buildWeakClassifiers(prefix, num_features, X, y, features, ws, cuda, verbose);
    weak_classifiers = tr.first;
    writeWeakClassifiers(prefix, weak_classifiers);
  }
  else
  {
    weak_classifiers = readWeakClassifiers(prefix, num_features);
  }

  if (verbose)
  {
    cout << weak_classifiers[0] << "\n";
    cout << weak_classifiers[1] << "\n";
  }

  vector<int> y_pred(N);
  for (size_t i = 0; i < N; ++i)
  {
    y_pred[i] = strongClassifier(X[i], weak_classifiers);
  }

  if (verbose)
    cout << "Evaluating \n";

  evaluate(y, y_pred);
}

#endif
