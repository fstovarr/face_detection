#ifndef LOCATIONS_H
#define LOCATIONS_H
#include <vector>
#include <iostream>
#include <numeric>

const int WINDOW_SIZE=19;
using namespace std;

struct Size{
  int height, width;
};


ostream& operator << (ostream& os, Size sz) {
  os << "Size(height=" << sz.height << ", width=" << sz.width << ")";
  return os;
}

struct Location{
  int left, top;
};

ostream& operator << (ostream& os, Location loc) {
  os << "Location(top=" << loc.top << ", left=" << loc.left << ")";
  return os;
}

vector<int> possiblePositions(int size, int window_size = WINDOW_SIZE) {
  vector<int> v(window_size - size + 1);
  iota(v.begin(), v.end(), 0);
  return v;
}

vector<Location> possibleLocations(Size base_shape, int window_size = WINDOW_SIZE) {
  vector<Location> v;
  for (int x : possiblePositions(base_shape.width, window_size)) {
    for (int y : possiblePositions(base_shape.height, window_size)) {
      v.push_back(Location{x, y}); //x=left, y=top
    }
  }
  return v;
}

vector<Size> possibleShapes(Size base_shape, int window_size = WINDOW_SIZE) {
  vector<Size> v;
  int base_height = base_shape.height;
  int base_width = base_shape.width;

  for (int width = base_width; width < window_size + 1; width += base_width) {
    for (int height = base_height; height < window_size + 1; height += base_height) {
      v.push_back(Size{height, width});
    }
  }
  return v;
}

void test_locations() {
  for (auto x : possibleShapes(Size{1, 2}, 5)) {
    cout << x << "\n";
  }
  for (auto x : possibleLocations(Size{1, 2}, 5)) {
    cout << x << "\n";
  }
}
#endif

