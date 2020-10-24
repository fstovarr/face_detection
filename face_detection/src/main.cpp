#include <iostream>
#include <vector>
#include "FileReader.cpp"

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
            continue;
        }

        for (int row = 0; row < sample.size(); row++)
        {
            for (int col = 0; col < 19; col++)
                cout << (int)sample[row][col] << "\t";
            cout << endl;
        }
        cout << "-----------";
        
        // TODO: change function (O(N))
        sample.clear();
    }
}