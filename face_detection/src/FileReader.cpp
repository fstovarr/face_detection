#include <string>
#include <fstream>
#include <sstream>
#include <array>

using namespace std;

class FileReader
{
    string file;

public:
    FileReader(string filename)
    {
        file = filename;
    }

    array<array<unsigned char, 19>, 19> getSample()
    {
        array<array<unsigned char, 19>, 19> array;

        ifstream infile("./img/train/face/face00001.pgm");
        stringstream ss;
        string inputLine = "";

        getline(infile, inputLine);

        int row = 0, col = 0, numrows = 0, numcols = 0;
        ss << infile.rdbuf();
        ss >> numcols >> numrows;
        ss >> inputLine;
        unsigned char tmp;

        for (row = 0; row < numrows; ++row)
            for (col = 0; col < numcols; ++col)
            {
                ss >> tmp;
                array[row][col] = tmp;
            }

        infile.close();
        return array;
    }
};