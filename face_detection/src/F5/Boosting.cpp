#ifndef BOOSTING_H
#define BOOSTING_H
#define USE_OMP
#include <vector>
#include <algorithm>
#include <chrono>
#include <limits>
#include <stdlib.h>
#include <math.h>
#include <fstream>
#include "Features.cpp"
#include "Metrics.cpp"
#include "../Image.cpp"
#include "../nlohmann/json.hpp"

using json = nlohmann::json;


vector<float> normalizeWeights(vector<float> v) {
  float S = 0.0;
  for (float x : v) S += x;
  for (size_t i = 0; i < v.size(); ++i) {
    v[i] /= S;
  }
  return v;
}

int sign(float x) {
  if (x) return 1.0;
  return -1.0;
}

struct  ThresholdPolarity{
  float threshold;
  int polarity;
};

struct ClassifierResult {
  float threshold;
  int polarity;
  float classification_error;
  Feature f;
};

struct WeakClassifierF5 {
  float threshold;
  int polarity;
  float alpha;
  Feature f;


  template <typename T>
  float operator() (vector<vector<T>> x) {
    float theta = threshold;
    return (sign((polarity * theta) - (polarity * f(x))) + 1) / 2;
  }

  json to_json() {
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

ostream& operator << (ostream& os, WeakClassifierF5 const & clf) {
  cout << "WeakClassifierF5(threshold=" << clf.threshold << ", polarity=" << clf.polarity << ", alpha=" << clf.alpha << " f=" << clf.f << ")";
  return os;
}

template <typename T>
int strongClassifier(vector<vector<T>> const & x, vector<WeakClassifierF5> const & weak_classifiers) {
  float sum_hypotheses = 0;
  float sum_alphas = 0;
  for (auto clf : weak_classifiers) {
    sum_hypotheses += clf.alpha * clf(x);
    sum_alphas += clf.alpha;
  }
  if (sum_hypotheses >= .5 * sum_alphas) return 1;
  return 0;
}

struct RunningSums {
  float t_minus, t_plus;
  vector<float> s_minuses, s_pluses;
};

RunningSums buildRunningSums(vector<int> ys, vector<float> ws) {
  float s_minus = 0, s_plus = 0;
  float t_minus = 0, t_plus = 0;
  vector<float> s_minuses, s_pluses;

  for (size_t i = 0; i < ys.size(); ++i) {
    float y = ys[i], w = ws[i];
    if (y < .5) {
      s_minus += w;
      t_minus += w;
    } else {
      s_plus += w;
      t_plus += w;
    }
    s_minuses.push_back(s_minus);
    s_pluses.push_back(s_plus);
  }
  return RunningSums{t_minus, t_plus, s_minuses, s_pluses};
}

ThresholdPolarity find_best_threshold(vector<float> zs, RunningSums rs) {
  float min_e = numeric_limits<float>::max();
  float min_z = 0;
  int polarity = 0;
  float t_minus = rs.t_minus, t_plus = rs.t_plus;
  vector<float> s_minuses = rs.s_minuses, s_pluses = rs.s_pluses;

  for (size_t i = 0; i < zs.size(); ++i) {
    float z = zs[i], s_m = s_minuses[i], s_p = s_pluses[i];
    float error_1 = s_p + (t_minus - s_m);
    float error_2 = s_m + (t_plus - s_p);
    if (error_1 < min_e) {
      min_e = error_1;
      min_z = z;
      polarity = -1;
    } else if (error_2 < min_e) {
      min_e = error_2;
      min_z = z;
      polarity = 1;
    }
  }
  return ThresholdPolarity{min_z, polarity};
}

typedef pair<pair<float, float>, float> f3;
bool cmp(f3 lh, f3 rh) { return lh.second < rh.second; }

ThresholdPolarity determineThresholdPolarity(vector<int> ys, vector<float> ws, vector<float> zs) {
  // sort according to zs
  vector<f3> triplet(ys.size());
  for (size_t i = 0; i < ys.size(); ++i) triplet[i] = {{ys[i], ws[i]}, zs[i]};

  sort(triplet.begin(), triplet.end(), cmp);
  for (size_t i = 0; i < ys.size(); ++i) {
    float y = triplet[i].first.first, w = triplet[i].first.second;
    ys[i] = y;
    ws[i] = w;
  }

  RunningSums rs = buildRunningSums(ys, ws);

  return find_best_threshold(zs, rs);
}

typedef vector<vector<int>> matrix;

ClassifierResult applyFeature(Feature f, vector<matrix> xis, vector<int> ys, vector<float> ws) {

  // determine all feature values
  vector<float> zs(xis.size());
  #pragma omp parallel for if(USE_OMP)
  for (size_t i = 0; i < xis.size(); ++i) {
    zs[i] = f(xis[i]);
  }

  // determine best threshold
  ThresholdPolarity result = determineThresholdPolarity(ys, ws, zs);

  float classification_error = .0;
  WeakClassifierF5 clf{result.threshold, result.polarity, 0, f};
  for (size_t i = 0; i < xis.size(); ++i) {
    float h = clf(xis[i]);
    classification_error += ws[i] * abs(h - ys[i]);
  }

  return ClassifierResult{result.threshold, result.polarity, classification_error, f};
}

const int   STATUS_EVERY     = 2000;
const float KEEP_PROBABILITY = 1./4.;


typedef pair<vector<WeakClassifierF5>, vector<float>> TrainingResult;
TrainingResult buildWeakClassifiers(
    string prefix, int num_features, vector<matrix> xis, vector<int> ys, vector<Feature> features, vector<float> ws
  )
{
  // initialize weights
  if (ws.empty()) {
    int m = 0, l = 0;
    for (size_t i = 0; i < ys.size(); ++i) {
      if (ys[i] > .5) {
        l++;
      } else {
        m++;
      }
    }
    ws.resize(ys.size());
    for (size_t i = 0; i < ys.size(); ++i) {
      if (ys[i] > .5) {
        ws[i] = 1. / (2. * l);
      } else {
        ws[i] = 1. / (2. * m);
      }
    }
  }

  auto total_start = chrono::high_resolution_clock::now(); 

  vector<WeakClassifierF5> weak_classifiers;
  for (int f_i = 0; f_i < num_features; ++f_i) {
    cout << "Building weak classifier " << f_i + 1 << "/" << num_features << "...\n";
    auto start = chrono::high_resolution_clock::now(); 

    ws = normalizeWeights(ws);

    int status_counter = STATUS_EVERY;

    ClassifierResult best{0, 0, numeric_limits<float>::max(), features[0]};
    bool improved = false;

    // Select best weak classifier for this round
    for (size_t i = 0; i < features.size(); ++i) {
      status_counter -= 1;
      improved = false;

      //if (KEEP_PROBABILITY < .1) {
        float skip_probability = rand();
        if (skip_probability > KEEP_PROBABILITY) continue;
      //}

      ClassifierResult result = applyFeature(features[i], xis, ys, ws);
      if (result.classification_error < best.classification_error) {
        improved = true;
        best = result;
      }

      // Print status every couple of iterations
      if (improved || status_counter == 0) {
        auto stop = chrono::high_resolution_clock::now();
        chrono::duration<float, milli> duration = stop - start; 
        chrono::duration<float, milli> total_duration = stop - start; 
        status_counter = STATUS_EVERY;
        if (improved) {
          cout << f_i + 1 << "/" << num_features << " (" << total_duration.count() << ")s " << " (" << duration.count() << "s in this stage)" << " "
            << 100. * i / features.size() << "% evaluated " << " Classification error improved to " << best.classification_error << " using " << best.f << "...\n";
        } else {
          cout << f_i + 1 << "/" << num_features << " (" << total_duration.count() << ")s " << " (" << duration.count() << "s in this stage)" << " "
            << 100. * i / features.size() << "% evaluated " << "...\n";
        }

      }
    }

    // After the best classifier was found, determine alpha
    float beta = best.classification_error / (1 - best.classification_error);
    float alpha = log(1. / beta);

    // Build the weak classifier
    WeakClassifierF5 clf{best.threshold, best.polarity, alpha, best.f};

    // Update the weights for misclassified examples

    for (size_t i = 0; i < xis.size(); ++i) {
      float h = clf(xis[i]);
      float e = abs(h - ys[i]);
      ws[i] = pow(beta, 1 - e);
    }

    // Register this weak classifier
    weak_classifiers.push_back(clf);

  }
  cout << "Done building " << num_features << " classifiers \n";
  return {weak_classifiers, ws};
}

void writeWeakClassifiers(string prefix, vector<WeakClassifierF5> clf_s) {
  ofstream o(prefix + "clf_s.jsonl");
  for (auto clf : clf_s) {
    o << clf.to_json() << "\n";
  }

}

vector<WeakClassifierF5> readWeakClassifiers(string prefix, int num_features) {
  ifstream in(prefix + "clf_s.jsonl");
  vector<WeakClassifierF5> v;
  json j;
  for (int i = 0; i < num_features; ++i) {
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
      constructFeature(x, y, width, height, type)
    };
    v.push_back(clf);
  }
  return v;
}


void trainF5(vector<pair<Image, int>> trainingData) {
  size_t N = trainingData.size();
  vector<matrix> X(N);
  vector<int> y(N);

  for (size_t i = 0; i < N; ++i) {
    X[i] = to_integral(trainingData[i].first.getIntImage());
    y[i] = trainingData[i].second;
  }

  int window_size = X[0].size();
  vector<Feature> features = getFeatures(window_size - 1);
  cout << "Using " << features.size() << " features\n";

  vector<float> ws;

  vector<WeakClassifierF5> weak_classifiers;

  string prefix = "f1";
  int num_features = 2;

  bool train = false;
  if (train) {
    TrainingResult tr = buildWeakClassifiers(prefix, num_features, X, y, features, ws);
    weak_classifiers = tr.first;
    writeWeakClassifiers(prefix, weak_classifiers);
  } else {
    weak_classifiers = readWeakClassifiers(prefix, num_features);
  }



  cout << weak_classifiers[0] << "\n";
  cout << weak_classifiers[1] << "\n";
  vector<int> y_pred(N);
  for (int i = 0; i < N; ++i) {
    y_pred[i] = strongClassifier(X[i], weak_classifiers);
  }

  cout << "Evaluating \n";

  evaluate(y, y_pred);

}


#endif
