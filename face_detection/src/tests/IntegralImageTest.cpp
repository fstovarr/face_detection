#include <iostream>
#include <vector>
#include "../IntegralImage.cpp"

using namespace std;

int run()
{
    vector<vector<unsigned char>> tmp = vector<vector<unsigned char>>(5, vector<unsigned char>(5, 0));
    for (int i = 0; i < 5; i++)
    {
        for (int j = 0; j < 5; j++)
        {
            tmp[i][j] = (char)(i + j);
        }
    }

    for (int row = 0; row < 5; row++)
    {
        for (int col = 0; col < 5; col++)
            cout << (int)tmp[row][col] << "\t";
        cout << endl;
    }

    cout << "-----------------------" << endl;

    IntegralImage ii(&tmp, 5);
    ii.print();

    long int area = ii.getArea(make_pair(0, 0), make_pair(4, 4));
    printf("\n%d\n", area);
}