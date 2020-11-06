#include <iostream>

using namespace std;

class RectangleRegion
{
    pair<int, int> _topLeft;
    pair<int, int> _bottomRight;
    bool _dummy;

public:
    RectangleRegion(int x, int y, int h, int w)
    {
        Constructor(make_pair(x, y), make_pair(x + h, y + w));
    }

    RectangleRegion(pair<int, int> topLeft, pair<int, int> bottomRight)
    {
        Constructor(topLeft, bottomRight);
    }

    RectangleRegion(bool dummy)
    {
        _dummy = true;
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
        return _dummy;
    }

private:
    void Constructor(pair<int, int> topLeft, pair<int, int> bottomRight)
    {
        _topLeft = topLeft;
        _bottomRight = bottomRight;
        _dummy = false;
    }
};