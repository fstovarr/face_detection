#include <iostream>
#include <vector>

using namespace std;

class WeakClassifier
{
    vector<RectangleRegion *> _positiveRegions;
    vector<RectangleRegion *> _negativeRegions;
    double _threshold;
    double _polarity;

public:
    WeakClassifier(vector<RectangleRegion *> positiveRegions, vector<RectangleRegion *> negativeRegions, double threshold, double polarity)
    {
        _positiveRegions = positiveRegions;
        _negativeRegions = negativeRegions;
        _threshold = threshold;
        _polarity = polarity;
    }

    int classify(IntegralImage ii)
    {
        long int accPos = 0L;
        for (RectangleRegion *region : _positiveRegions)
        {
            accPos += ii.getArea(*region);
        }

        long int accNeg = 0L;
        for (RectangleRegion *region : _negativeRegions)
        {
            accNeg += ii.getArea(*region);
        }

        if (_polarity * (accPos - accNeg) < _polarity * _threshold)
            return 1;
        return 0;
    }
};