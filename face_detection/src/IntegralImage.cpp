#include <array>
#include <vector>

using namespace std;

class IntegralImage
{
    vector<vector<long int>> _integral;

public:
    IntegralImage(vector<vector<unsigned char>> *image, int size)
    {
        _integral = vector<vector<long int>>(size, vector<long int>(size, 0));

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

        for (int i = 0; i < size; i++)
        {
            for (int j = 0; j < size; j++)
            {
                printf("%ld\t", _integral[i][j]);
            }
            printf("\n");
        }
    }
};