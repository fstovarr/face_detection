#include <iostream>
#include <vector>

using namespace std;

class WeakClassifier
{
    vector<RectangleRegion> _regions;
    double _threshold;
    double _polarity;

public:
    WeakClassifier(vector<RectangleRegion> *regions, double threshold, double polarity)
    {
        _regions = *regions;
        _threshold = threshold;
        _polarity = polarity;
    }

    int classify(IntegralImage ii)
    {
        int accPos = 0L, accNeg = 0L;

        // 0 - 1 positive 2 - 4 negative
        for (int k = 0; k < _regions.size(); k++)
            if (k <= 1)
                accPos += (ii.getArea(_regions[k]));
            else
                accNeg += (ii.getArea(_regions[k]));

        if ((_polarity * (accPos - accNeg)) < (_polarity * _threshold))
            return 1;
        return 0;
    }
};