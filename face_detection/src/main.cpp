#include <iostream>
#include <vector>
#include "FileReader.cpp"
#include "IntegralImage.cpp"
#include "Features.cpp"
#include "Locations.cpp"

using namespace std;

void test() {
  cout << "Running test\n";
  test_feature2h();
  test_locations();
  vector<Feature> features = getFeatures(15);
  cout << features.size();
}

int main(int argc, char *argv[])
{
    test();
    return 0;
    FileReader fr("./img/train/face/");
    vector<vector<unsigned char>> sample;

    int i = 0;
    while (fr.remainingSamples())
    {
        cout << fr.remainingSamples() << "\n";

        int res = fr.getSample(&sample);

        if (!res)
        {
            cout << "Error opening the file";
            // continue;
        }
        cout << sample.size() << " " << sample[0].size() << "\n";
        IntegralImage ii = IntegralImage(sample, sample.size());
        ii.print();
        // TODO: change function (O(N))
        sample.clear();
        i += 1;
        if (i == 10) break;
    }
}
