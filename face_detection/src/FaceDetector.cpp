#include <array>
#include <vector>
#include <math.h>

#include "Image.cpp"
#include "IntegralImage.cpp"
#include "RectangleRegion.cpp"
#include "Features.cpp"

using namespace std;

class FaceDetector
{
    int _weakClassifiers = 10;
    vector<> _alphas;
    vector<> _clfs;

public:
    FaceDetector(int weakClassifiers, vector<> alphas, vector<> clfs)
    {
        _weakClassifiers = weakClassifiers;
        _alphas = alphas;
        _clfs = clfs;
    }

    void train(vector<pair<Image, int>> training, int positiveSamples, int negativeSamples)
    {
        vector<double> weights = vector<double>(training.size(), 0);

        vector<vector<int>> X;
        vector<int> y(training.size());

        // initialize weights and y
        for (size_t i = 0; i < training.size(); i++)
        {
            pair<Image, int> sample = training[i];

            weights[i] = 1.0 / (2 * (sample.second == 1 ? positiveSamples : negativeSamples));
            y[i] = sample.second;
        }

        // use 5 features instead of 4
        bool useF5 = true;

        if (useF5) {
          vector<Feature> f5 = getFeatures(training);
          y = vector<int>(trainingData.size());

          X = applyFeaturesF5(f5, );


        } else {
          vector<IntegralImage> iis(training.size());
          for (size_t i = 0; i < training.size(); i++) {
              iis[i] = IntegralImage(training[i].first);
          }
          vector<vector<RectangleRegion *>> features = buildFeatures(iis[0].getSize(), iis[0].getSize());
          X = applyFeatures(features, iis);
        }

        // TODO: Select best 10% of samples

        for (int t = 0; t < _weakClassifiers; t++)
        {
            normalize(weights);
        }
    }

    void trainWeak(vector<vector<int>> X, vector<int> y, vector<vector<RectangleRegion *>> features, vector<double> weights)
    {
        double totalPos = 0;
        double totalNeg = 0;

        for (int i = 0; i < weights.size(); i++)
        {
            if (y[i] == 1)
                totalPos += weights[i];
            else
                totalNeg += weights[i];
        }

        vector<> classifiers;
        int totalFeatures = X.size();
        for(int i = 0; i < X.size(); i++) {
            if(classifiers.size() % 1000 == 0 && classifiers.size() > 0) {
                printf("Trained %d classifiers ", classifiers.size());
            }
        }
    }

    void normalize(vector<double> *x)
    {
        double norm = norm(x);

        for (int i = 0; i < (*x).size(); i++)
            (*x[i])[i] /= norm;
    }

    double norm(vector<double> x)
    {
        double ans = 0.0;
        for (double el : x)
        {
            ans += x * x;
        }
        return sqrt(ans);
    }

    vector<vector<RectangleRegion *>> buildFeatures(int imgWidth, int imgHeight)
    {
        vector<vector<RectangleRegion *>> features;
        int i = 0, j = 0;

        RectangleRegion *current, *right, *bottom, *right2, *bottom2, *bottomRight;
        RectangleRegion *dummy = &RectangleRegion(true);
        vector<RectangleRegion *> tmp = vector<RectangleRegion *>(4, dummy);

        for (int w = 1; w < imgWidth + 1; w++)
        {
            for (int h = 1; h < imgHeight + 1; h++)
            {
                i = 0;
                while (i + w < imgWidth)
                {
                    j = 0;
                    while (j + h < imgHeight)
                    {
                        current = &RectangleRegion(i, j, w, h);

                        right = &RectangleRegion(i + w, j, w, h);
                        if (i + 2 * w < imgWidth)
                        {
                            tmp[1] = tmp[3] = dummy;
                            tmp[0] = right;
                            tmp[2] = current;
                            features.push_back(tmp);
                        }

                        bottom = &RectangleRegion(i, j + h, w, h);
                        if (j + 2 * h < imgHeight)
                        {
                            tmp[1] = tmp[3] = dummy;
                            tmp[0] = current;
                            tmp[2] = bottom;
                            features.push_back(tmp);
                        }

                        right2 = &RectangleRegion(i + 2 * w, j, w, h);
                        if (i + 3 * w < imgWidth)
                        {
                            tmp[1] = dummy;
                            tmp[0] = right;
                            tmp[2] = right2;
                            tmp[3] = current;
                            features.push_back(tmp);
                        }

                        bottom2 = &RectangleRegion(i, j + 2 * h, w, h);
                        if (j + 3 * h < imgHeight)
                        {
                            tmp[1] = dummy;
                            tmp[0] = bottom;
                            tmp[2] = bottom2;
                            tmp[3] = current;
                            features.push_back(tmp);
                        }

                        bottomRight = &RectangleRegion(i + w, j + h, w, h);
                        if (i + 2 * w < imgWidth && j + 2 * h < imgHeight)
                        {
                            tmp[0] = right;
                            tmp[1] = bottom;
                            tmp[2] = current;
                            tmp[3] = bottomRight;
                            features.push_back(tmp);
                        }
                    }
                }
            }
        }

        return features;
    }

    vector<vector<int>> applyFeatures(vector<vector<RectangleRegion *>> features, vector<IntegralImage> trainingData)
    {
        vector<vector<int>> X = vector<vector<int>>(features.size(), vector<int>(trainingData.size(), 0));

        int i = 0;
        long int tmp = 0L;
        for (vector<RectangleRegion *> f : features)
        {
            tmp = 0L;
            for (IntegralImage td : trainingData)
            {
                for (int j = 0; j < f.size(); j++)
                {
                    if (!(*f[j]).isDummy())
                        tmp += td.getArea(*f[j]);
                }
            }

            X[i++].push_back(tmp);
        }

        return X;
    }
};
