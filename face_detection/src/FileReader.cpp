#include <string>
#include <fstream>
#include <sstream>
#include <array>
#include <dirent.h>
#include <vector>

using namespace std;

class FileReader
{
    vector<string> _files;
    vector<string>::iterator _currentFile;

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
                if (tmp.size() > 2)
                    _files.push_back(folder + "/" + tmp);
            }
            closedir(dir);
        }
        else
        {
            perror("");
        }

        _currentFile = _files.begin();
    }

    int remainingSamples()
    {
        return _files.end() - _currentFile;
    }

    int getSample(vector<vector<unsigned char>> *array, bool initArray)
    {
        ifstream infile(*_currentFile);
        if (!infile.good())
        {
            perror((*_currentFile).c_str());
            return NULL;
        }

        ++_currentFile;

        stringstream ss;
        string inputLine = "";

        getline(infile, inputLine);

        int row = 0, col = 0, numrows = 0, numcols = 0;
        ss << infile.rdbuf();
        ss >> numcols >> numrows;
        ss >> inputLine;
        unsigned char tmp;

        for (row = 0; row < numrows; ++row)
        {
            vector<unsigned char> tmpVec;
            for (col = 0; col < numcols; ++col)
            {
                ss >> tmp;
                if (!initArray)
                    (*array)[row][col] = tmp;
                else
                    tmpVec.push_back(tmp);
            }

            if (initArray)
                (*array).push_back(tmpVec);
        }

        infile.close();
        return numrows * numcols;
    }
};