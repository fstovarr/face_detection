#include <array>
#include <vector>

using namespace std;

class IntegralImage
{
    vector<vector<long int>> _integral;
    int _size;

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
        if (topLeft.first < 0 || topLeft.second < 0 || bottomRight.first < 0 || bottomRight.second < 0)
            throw "Out of bounds";

        topLeft.first = (topLeft.first - 1) >= 0 ? topLeft.first - 1 : 0;
        topLeft.second = (topLeft.second - 1) >= 0 ? topLeft.second - 1 : 0;

        pair<int, int> topRight = make_pair(topLeft.first, bottomRight.second),
                       bottomLeft = make_pair(bottomRight.first, topLeft.second);

        return _integral[bottomRight.first][bottomRight.second] +
               _integral[topLeft.first][topLeft.second] -
               _integral[bottomLeft.first][bottomLeft.second] -
               _integral[topRight.first][topRight.second];
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