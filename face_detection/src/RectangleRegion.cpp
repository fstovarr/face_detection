#include <iostream>

using namespace std;

class RectangleRegion
{
    pair<int, int> _topLeft;
    pair<int, int> _bottomRight;

public:
    RectangleRegion(pair<int, int> topLeft, pair<int, int> bottomRight)
    {
        _topLeft = topLeft;
        _bottomRight = bottomRight;
    }

    pair<int, int> getTopLeft() {
        return _topLeft;
    }

    pair<int, int> getBottomRight() {
        return _bottomRight;
    }
};