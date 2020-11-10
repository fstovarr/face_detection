#ifndef IMAGE_H
#define IMAGE_H
#include <vector>

using namespace std;

class Image
{
    vector<vector<unsigned char>> _image;
    int _size;

public:
    Image(vector<vector<unsigned char>> image, int size)
    {
        _image = image;
        _size = size;
    }

    vector<vector<unsigned char>> getImage()
    {
        return _image;
    }

    vector<vector<int>> getIntImage() {
      vector<vector<int>> m(_size, vector<int>(_size));
      for (int i = 0; i < _size; ++i) {
        for (int j = 0; j < _size; ++j) {
          m[i][j] = _image[i][j];
        }
      }
      return m;
    }

    int getSize()
    {
        return _size;
    }
};


#endif
