#include <array>
#include <vector>

using namespace std;

class IntegralImage
{
    vector<vector<long int>> _integral;
    int _size;

    long int _getSubArea(int x, int y)
    {
        if (x < 0 || y < 0 || x > _size || y > _size)
            return 0L;
        return _integral[x][y];
    }

public:
    IntegralImage(vector<vector<unsigned char>> *image, int size)
    {
        _integral = vector<vector<long int>>(size, vector<long int>(size, 0));
        _size = size;

        long int top = 0, left = 0, current = 0, topLeft = 0;

        for (int x = 0; x < size; x++)
        {
            for (int y = 0; y < size; y++)
            {
                top = (x - 1) >= 0 ? _integral[x - 1][y] : 0;
                left = (y - 1) >= 0 ? _integral[x][y - 1] : 0;
                topLeft = (x - 1) >= 0 && (y - 1) >= 0 ? _integral[x - 1][y - 1] : 0;
                current = (int)(*image)[x][y];
                _integral[x][y] = top + left - topLeft + current;
            }
        }
    }

    long int getArea(pair<int, int> topLeft, pair<int, int> bottomRight)
    {
        if (topLeft.first < 0 || topLeft.second < 0 || bottomRight.first < 0 || bottomRight.second < 0 ||
            topLeft.first >= _size || topLeft.second >= _size || bottomRight.first >= _size || bottomRight.second >= _size)
            throw "Out of bounds";

        pair<int, int> topRight = make_pair(topLeft.first - 1, bottomRight.second - 1),
                       bottomLeft = make_pair(bottomRight.first - 1, topLeft.second - 1);

        long int brArea = _getSubArea(bottomRight.first, bottomRight.second);
        long int tlArea = _getSubArea(topLeft.first, topLeft.second);
        long int blArea = _getSubArea(bottomLeft.first, bottomLeft.second);
        long int trArea = _getSubArea(topRight.first, topRight.second);

        return brArea + tlArea - blArea - blArea;
    }

    void print()
    {
        for (int i = 0; i < _size; i++)
        {
            for (int j = 0; j < _size; j++)
            {
                printf("%ld\t", _integral[i][j]);
            }
            printf("\n");
        }
    }
};