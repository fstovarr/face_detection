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

    int getSize()
    {
        return _size;
    }
};