#include <array>
#include <vector>
#include <math.h>
#include <tuple>
#include <float.h>

#include "WeakClassifier.cpp"

using namespace std;

struct WeakHelper
{
    double weight;
    vector<int> feature;
    int y;
};

class FaceDetector
{
    int _weakClassifiers = 10;
    vector<double> _alphas;
    double _alphasSum;
    vector<WeakClassifier> _clfs;

public:
    FaceDetector(int weakClassifiers)
    {
        _weakClassifiers = weakClassifiers;
    }

    void train(vector<pair<Image, int>> &training, int positiveSamples, int negativeSamples)
    {
        vector<double> weights = vector<double>(training.size(), 0);
        vector<pair<IntegralImage, int>> trainingData;

        for (int i = 0; i < training.size(); i++)
        {
            pair<Image, int> sample = training[i];

            trainingData.push_back(make_pair(IntegralImage(sample.first), sample.second));

            weights[i] = 1.0 / (2 * (sample.second == 1 ? positiveSamples : negativeSamples));
        }

        printf("RR %d \n", sizeof(RectangleRegion));

        vector<vector<RectangleRegion *>> features = buildFeatures(trainingData[0].first.getSize(), trainingData[0].first.getSize());

        cout << features.size() << " - " << features[0].size() << endl;

        int c = 0;
        for (int i = 0; i < features.size(); i++)
        {
            for (int j = 0; j < features[0].size(); j++)
            {
                if ((*features[i][j]).isDummy() == false)
                {
                    (*features[i][j]).print();
                    if (++c == 10)
                        break;
                }
            }
            if (c == 10)
                break;
        }

        cout << "FINISH ----------------------- " << features.size() << endl;

        int fSize = features.size();

        int trainSize = trainingData.size();
        vector<vector<int>> X(5000);
        for (int i = 0; i < X.size(); i++)
        {
            X[i] = vector<int>(10);
        }

        // applyFeatures(features, trainingData);
        cout << "----------------------- FINISH  " << endl;

        c = 0;
        for (int i = 0; i < features.size(); i++)
        {
            for (int j = 0; j < features[0].size(); j++)
            {
                if ((*features[i][j]).isDummy() == false)
                {
                    (*features[i][j]).print();
                    if (++c == 10)
                        break;
                }
            }
            if (c == 10)
                break;
        }

        // vector<vector<int>> X = featuresApplied.first;
        // vector<int> y = featuresApplied.second;

        // for (int t = 0; t < _weakClassifiers; t++)
        // {
        //     printf("2");
        //     // normalize(weights); // NORM
        //     vector<WeakClassifier> weakClassifiers = trainWeak(X, y, features, weights);
        //     tuple<WeakClassifier, double, vector<double>> best = selectBest(weakClassifiers, weights, trainingData);
        //     WeakClassifier bestClf = get<0>(best);
        //     double bestError = get<1>(best);
        //     vector<double> bestAccuracy = get<2>(best);
        //     double beta = bestError / (1.0 / bestError);
        //     for (int i = 0; i < bestAccuracy.size(); i++)
        //         weights[i] *= (pow(beta, 1 - bestAccuracy[i]));

        //     double alpha = log(1.0 / beta);
        //     _alphas.push_back(alpha);
        //     _alphasSum += alpha;
        //     _clfs.push_back(bestClf);
        //     printf("Chose classifier: %d with accuracy: and alpha: %f", t, alpha);
        // }
    }

    vector<vector<RectangleRegion *>> buildFeatures(int imgWidth, int imgHeight)
    {
        int i = 0, j = 0;

        vector<vector<RectangleRegion *>> features;
        RectangleRegion current, right, bottom, right2, bottom2, bottomRight;
        RectangleRegion dummy = RectangleRegion();
        vector<RectangleRegion *> tmp = vector<RectangleRegion *>(4, &dummy);

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
                        current = RectangleRegion(i, j, w, h);

                        if (i + 2 * w < imgWidth)
                        {
                            right = RectangleRegion(i + w, j, w, h);
                            tmp[1] = tmp[3] = &dummy;
                            tmp[0] = &right;
                            tmp[2] = &current;
                            features.push_back(tmp);
                        }

                        if (i + 3 * w < imgWidth)
                        {
                            right = RectangleRegion(i + w, j, w, h);
                            right2 = RectangleRegion(i + 2 * w, j, w, h);
                            tmp[1] = &dummy;
                            tmp[0] = &right;
                            tmp[2] = &right2;
                            tmp[3] = &current;
                            features.push_back(tmp);
                        }

                        if (j + 2 * h < imgHeight)
                        {
                            bottom = RectangleRegion(i, j + h, w, h);
                            tmp[1] = tmp[3] = &dummy;
                            tmp[0] = &current;
                            tmp[2] = &bottom;

                            features.push_back(tmp);
                        }

                        if (j + 3 * h < imgHeight)
                        {
                            bottom = RectangleRegion(i, j + h, w, h);
                            bottom2 = RectangleRegion(i, j + 2 * h, w, h);

                            tmp[1] = &dummy;
                            tmp[0] = &bottom;
                            tmp[2] = &bottom2;
                            tmp[3] = &current;
                            features.push_back(tmp);
                        }

                        if (i + 2 * w < imgWidth && j + 2 * h < imgHeight)
                        {
                            right = RectangleRegion(i + w, j, w, h);
                            bottom = RectangleRegion(i, j + h, w, h);
                            bottomRight = RectangleRegion(i + w, j + h, w, h);
                            tmp[0] = &right;
                            tmp[1] = &bottom;
                            tmp[2] = &current;
                            tmp[3] = &bottomRight;

                            features.push_back(tmp);
                        }
                        j++;
                    }
                    i++;
                }
            }
        }

        return features;
    }

    int applyFeatures(vector<vector<RectangleRegion *>> &features, vector<pair<IntegralImage, int>> &trainingData)
    {
        // vector<int> y = vector<int>(trainSize);

        // for (int i = 0; i < trainSize; i++)
        //     y[i] = trainingData[i].second;

        // int i = 0;
        // long int tmp = 0L;
        // for (vector<RectangleRegion *> f : features)
        // {
        //     tmp = 0L;
        //     for (pair<IntegralImage, int> td : *trainingData)
        //     {
        //         for (RectangleRegion *rr : f)
        //         {
        //             if ((*rr).isDummy() == false)
        //                 (*rr).print();
        //             if (++i == 10)
        //                 break;
        //             // tmp += td.first.getArea(*f[j]);
        //         }
        //         if (i == 10)
        //             break;
        //     }
        //     if (i == 10)
        //         break;

        //     X[i++].push_back(tmp);
        // }

        // return make_pair(X, y);
        return 1;
    }

    vector<WeakClassifier> trainWeak(vector<vector<int>> X, vector<int> y, vector<vector<RectangleRegion *>> features, vector<double> weights)
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

        vector<WeakClassifier> classifiers;
        int totalFeatures = X.size();
        for (int i = 0; i < X.size(); i++)
        {
            if (classifiers.size() % 1000 == 0 && classifiers.size() > 0)
            {
                printf("Trained %d classifiers ", classifiers.size());
            }

            // Sort appliedFeature
            vector<WeakHelper> appliedFeature;
            for (int j = 0; j < weights.size(); j++)
            {
                appliedFeature.push_back(WeakHelper());
                appliedFeature[i].weight = weights[i];
                appliedFeature[i].feature = X[i];
                appliedFeature[i].y = y[i];
            }

            int posSeen = 0, negSeen = 0;
            double posWeights = 0.0, negWeights = 0.0;
            double minError = DBL_MAX, bestPolarity = 0.0, error = 0.0;
            vector<int> bestThreshold;
            vector<RectangleRegion *> bestFeature;

            for (auto wh : appliedFeature)
            {
                error = min(negWeights + totalPos - posWeights, posWeights + totalNeg - negWeights);
                if (error < minError)
                {
                    minError = error;
                    bestFeature = features[i];
                    bestThreshold = wh.feature;
                    bestPolarity = posSeen > negSeen ? 1 : -1;
                }

                if (wh.y == 1)
                {
                    posSeen += 1;
                    posWeights += wh.weight;
                }
                else
                {
                    negSeen += 1;
                    negWeights += wh.weight;
                }
            }
        }

        return classifiers;
    }

    // void normalize(vector<double> *x)
    // {
    //     double norm = norm(x);

    //     for (int i = 0; i < (*x).size(); i++)
    //         (*x[i])[i] /= norm;
    // }

    // double norm(vector<double> x)
    // {
    //     double ans = 0.0;
    //     for (double el : x)
    //     {
    //         ans += x * x;
    //     }
    //     return sqrt(ans);
    // }

    tuple<WeakClassifier, double, vector<double>> selectBest(vector<WeakClassifier> classifiers, vector<double> weights, vector<pair<IntegralImage, int>> trainingData)
    {
        double bestError, error, correctness;
        vector<double> bestAccuracy;
        bestError = error = correctness = 0.0;

        WeakClassifier bestClf = classifiers[0];

        vector<double> accuracy;
        for (WeakClassifier clf : classifiers)
        {
            correctness = error = 0.0;
            accuracy.clear();

            for (int i = 0; i < weights.size(); i++)
            {
                correctness = abs(clf.classify(trainingData[i].first) - weights[i]);
                accuracy.push_back(correctness);
                error += correctness;
            }
            error /= trainingData.size();
            if (error < bestError)
            {
                bestClf = clf;
                bestError = error;
                bestAccuracy = accuracy;
            }
        }

        return make_tuple(bestClf, bestError, bestAccuracy);
    }

    int classify(Image image)
    {
        double total = 0.0;
        IntegralImage ii = IntegralImage(image);
        for (int i = 0; i < _clfs.size(); i++)
            total += _alphas[i] * _clfs[i].classify(ii);
        return total >= (0.5 * _alphasSum) ? 1 : 0;
    }
};
