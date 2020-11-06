#include <vector>
#include <iostream>
#include <string>
#include <assert.h>

using namespace std;

template <typename T>
vector<vector<T>> to_integral(vector<vector<T>> const &img) {
  int rows = img.size() + 1, cols = img[0].size() + 1;
  vector<vector<T>> ii(rows, vector<T>(cols, 0));

  for (int i = 1; i < rows; ++i) {
    for (int j = 1; j < cols; ++j) {
      ii[i][j] = ii[i][j - 1] + ii[i - 1][j] - ii[i - 1][j - 1] + img[i - 1][j - 1];
    }
  }

  return ii;
}
template <typename T >
ostream& operator << (ostream& os, const vector<vector<T>>& v) {
    os << "[";
    for (vector<T> row : v) {
      os << "[";
      for (T x : row)
        os << " " << x << ", ";
      os << "],\n";
    }
    os << " ]";
    return os;
}

template <typename T>
T coordSum(vector<int> coords_x, vector<int> coords_y, vector<int> coeffs, vector<vector<T>> const& ii) {
  T sum = 0;
  int x, y;

  for (size_t i = 0; i < coords_x.size(); ++i) {
    x = coords_x[i], y = coords_y[i];
    // NOTE that x and y are reversed.
    sum += ii[y][x] * coeffs[i];
  }
  return sum;
}

struct Box {
  int x, y, width, height;
  vector<int> coords_x, coords_y, coeffs;
  Box(int x, int y, int width, int height) : x(x), y(y), width(width), height(height) {
    coords_x = {x, x + width, x,          x + width};
    coords_y = {y, y,         y + height, y + height};
    coeffs =   {1, -1,        -1,         1};
  }

  template <typename T>
  T operator() (vector<vector<T>> ii) {
    return coordSum(coords_x, coords_y, coeffs, ii);
  }

};


class Feature {
  protected:
    int x, y, width, height;
    vector<int> coords_x, coords_y, coeffs;
    string type;
  public:
    Feature(int x, int y, int width, int height, vector<int> coords_x, vector<int> coords_y, vector<int> coeffs, string type) : x(x), y(y), width(width), height(height), coords_x(coords_x), coords_y(coords_y), coeffs(coeffs), type(type) {} ;

    template <typename T>
    T operator() (vector<vector<T>> ii) {
      return coordSum(coords_x, coords_y, coeffs, ii);
    }

    friend ostream& operator<<(ostream& os, const Feature& ft);
};

ostream& operator<<(ostream& os, const Feature& ft) {
  os << ft.type << "(x=" << ft.x << ", y=" << ft.y << ", width=" << ft.width << ", height=" << ft.height << ")";
  return os;
}


Feature feature2h(int x, int y, int width, int height) {
  int hw = width / 2;
  vector<int> coords_x{x,      x + hw,    x,      x + hw,
                       x + hw, x + width, x + hw, x + width};
  vector<int> coords_y{y,      y,          y + height, y + height,
                       y,      y,          y + height, y + height};
  vector<int> coeffs{1,     -1,         -1,          1,
                     -1,     1,          1,         -1};
  return Feature{x, y, width, height, coords_x, coords_y, coeffs, "Feature2h"};
}

Feature feature2v(int x, int y, int width, int height) {
  int hh = height / 2;
  vector<int> coords_x{x,      x + width,  x,          x + width,
                       x,      x + width,  x,          x + width};
  vector<int> coords_y{y,      y,          y + hh,     y + hh,
                       y + hh, y + hh,     y + height, y + height};
  vector<int> coeffs{-1,     1,          1,         -1,
                     1,     -1,         -1,          1};
  return Feature{x, y, width, height, coords_x, coords_y, coeffs, "Feature2v"};
}

Feature feature3h(int x, int y, int width, int height) {
  int tw = width / 3;
  vector<int> coords_x{x,        x + tw,    x,          x + tw,
                       x + tw,   x + 2*tw,  x + tw,     x + 2*tw,
                       x + 2*tw, x + width, x + 2*tw,   x + width};

  vector<int> coords_y{y,        y,         y + height, y + height,
                       y,        y,         y + height, y + height,
                       y,        y,         y + height, y + height};
  vector<int> coeffs{-1,       1,         1,         -1,
                     1,      -1,        -1,          1,
                     -1,       1,         1,         -1};
  return Feature{x, y, width, height, coords_x, coords_y, coeffs, "Feature3h"};
}


Feature feature3v(int x, int y, int width, int height) {
  int th = height / 3;
  vector<int> coords_x{x,        x + width,  x,          x + width,
                       x,        x + width,  x,          x + width,
                       x,        x + width,  x,          x + width};
  vector<int> coords_y{y,        y,          y + th,     y + th,
                       y + th,   y + th,     y + 2*th,   y + 2*th,
                       y + 2*th, y + 2*th,   y + height, y + height};
  vector<int> coeffs{-1,       1,         1,         -1,
                     1,      -1,        -1,          1,
                     -1,       1,         1,         -1};
  return Feature{x, y, width, height, coords_x, coords_y, coeffs, "Feature3v"};
}


Feature feature4(int x, int y, int width, int height) {
  int hw = width / 2, hh = height / 2;
  vector<int> coords_x{x,      x + hw,     x,          x + hw,     // upper row
                       x + hw, x + width,  x + hw,     x + width,
                       x,      x + hw,     x,          x + hw,     // lower row
                       x + hw, x + width,  x + hw,     x + width};
  vector<int> coords_y{y,      y,          y + hh,     y + hh,     // upper row
                       y,      y,          y + hh,     y + hh,
                       y + hh, y + hh,     y + height, y + height, // lower row
                       y + hh, y + hh,     y + height, y + height};
  vector<int> coeffs{1,     -1,         -1,          1,          // upper row
                     -1,     1,          1,         -1,
                     -1,     1,          1,         -1,          // lower row
                      1,    -1,         -1,          1};
  return Feature{x, y, width, height, coords_x, coords_y, coeffs, "Feature4"};
}



void test_feature2h() {
  vector<vector<int>> img{
    {5, 2, 3, 4, 1},
    {1, 5, 4, 2, 3},
    {2, 2, 1, 3, 4},
    {3, 5, 6, 4, 5},
    {4, 1, 3, 2, 6}
  };
  vector<vector<int>> ii = to_integral(img), sample_integral = ii;

  Feature f2h = feature2h(3, 1, 2, 4);

  assert(f2h(ii) == -7);

  vector<vector<int>> ones{
    {1, 1, 1, 1},
    {1, 1, 1, 1},
    {1, 1, 1, 1},
    {1, 1, 1, 1},
  };


  f2h = feature2h(0, 0, 4, 4);

  assert(f2h(to_integral(ones)) == 0);
  Box(0, 1, 4, 2);

  int expected = - Box(0, 1, 4, 2)(sample_integral) + Box(0, 3, 4, 2)(sample_integral);
  int actual = feature2v(0, 1, 4, 4)(sample_integral);
  assert(expected == actual);

  expected = - Box(0, 0, 1, 2)(sample_integral) + Box(1, 0, 1, 2)(sample_integral) - Box(2, 0, 1, 2)(sample_integral);
  actual = feature3h(0, 0, 3, 2)(sample_integral);
  assert(expected == actual);

  expected = - Box(0, 0, 2, 1)(sample_integral) + Box(0, 1, 2, 1)(sample_integral) - Box(0, 2, 2, 1)(sample_integral);
  actual = feature3v(0, 0, 2, 3)(sample_integral);
  assert(expected == actual);

  expected = Box(0, 0, 2, 2)(sample_integral) - Box(2, 0, 2, 2)(sample_integral) - Box(0, 2, 2, 2)(sample_integral) + Box(2, 2, 2, 2)(sample_integral);
  actual = feature4(0, 0, 4, 4)(sample_integral);
  assert(expected == actual);

}


