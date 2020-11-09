#include <iostream>

using namespace std;

class RectangleRegion
{
    pair<int, int> _topLeft;
    pair<int, int> _bottomRight;
    int _dummy = 0;

public:
    RectangleRegion()
    {
        _dummy = 1;
        Constructor(make_pair(-1, -1), make_pair(-1, -1));
    }

    RectangleRegion(const RectangleRegion &old_obj)
    {
        _dummy = old_obj._dummy;
        _topLeft = old_obj._topLeft;
        _bottomRight = old_obj._bottomRight;
    }

    RectangleRegion(int x, int y, int h, int w)
    {
        Constructor(make_pair(x, y), make_pair(x + h, y + w));
    }

    RectangleRegion(pair<int, int> topLeft, pair<int, int> bottomRight)
    {
        Constructor(topLeft, bottomRight);
    }

    pair<int, int> getTopLeft()
    {
        return _topLeft;
    }

    pair<int, int> getBottomRight()
    {
        return _bottomRight;
    }

    bool isDummy()
    {
        return _dummy == 1;
    }

    void print()
    {
        cout << "IS DUMMY " << _dummy << endl;
        cout << "(" << _topLeft.first << ", " << _topLeft.second << ") ";
        cout << "(" << _bottomRight.first << ", " << _bottomRight.second << ") " << endl;
    }

private:
    void Constructor(pair<int, int> topLeft, pair<int, int> bottomRight)
    {
        _topLeft = topLeft;
        _bottomRight = bottomRight;
    }
};