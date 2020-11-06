#include <iostream>
#include <vector>
#include "FileReader.cpp"
#include "IntegralImage.cpp"

using namespace std;

int main(int argc, char *argv[])
{
    FileReader fr("./img/train/face/");
    vector<vector<unsigned char>> sample;
    while (fr.remainingSamples())
    {
        cout << fr.remainingSamples();

        int res = fr.getSample(&sample);

        if (!res)
        {
            cout << "Error opening the file";
            // continue;
        }

        // TODO: change function (O(N))
        sample.clear();
    }
}