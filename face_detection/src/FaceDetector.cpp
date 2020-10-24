#include <array>

using namespace std;

class FaceDetector
{
    int _weakClassifiers = 10;

public:
    FaceDetector(int weakClassifiers)
    {
        _weakClassifiers = weakClassifiers;
    }

    void train(array<array<unsigned char, 19>, 19> *image, int classification, int positiveSamples, int negativeSamples) {
        
    }
}