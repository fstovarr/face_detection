#include <iostream>
#include "FileReader.cpp"
#include <array>

using namespace std;

int main(int argc, char *argv[])
{
    array<array<unsigned char, 19>, 19> sample;

    FileReader fr("./img/train/face/");
    while (fr.samplesLeft())
    {
        int res = fr.getSample(&sample);

        if(!res) {
            cout << "Error opening the file";
            continue;
        }

        for (int row = 0; row < 19; row++)
        {
            for (int col = 0; col < 19; col++)
                cout << (int)sample[row][col] << "\t";
            cout << endl;
        }
        cout << "-----------";
    }
}