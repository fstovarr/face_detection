#include <string>
#include <fstream>
#include <sstream>
#include <array>
#include <dirent.h>
#include <vector>

using namespace std;

class FileReader
{
    vector<string> files;
    vector<string>::iterator currentFile;

public:
    FileReader(string folder)
    {
        DIR *dir;
        struct dirent *ent;
        string tmp;
        if ((dir = opendir(folder.c_str())) != NULL)
        {
            while ((ent = readdir(dir)) != NULL)
            {
                tmp = ent->d_name;
                if(tmp.size() > 2)
                    files.push_back(folder + "/" + tmp);
            }
            closedir(dir);
        }
        else
        {
            perror("");
        }

        currentFile = files.begin();
    }

    int samplesLeft() {
        return files.end() - currentFile;
    }

    int getSample(array<array<unsigned char, 19>, 19> *array)
    {
        ifstream infile(*currentFile);
        if (!infile.good())
        {
            perror((*currentFile).c_str());
            return NULL;
        }

        ++currentFile;

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
                ss >> (*array)[row][col];

        infile.close();
        return numrows * numcols;
    }
};