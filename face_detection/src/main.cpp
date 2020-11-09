#include <iostream>
#include <vector>

#include "FileReader.cpp"
#include "IntegralImage.cpp"

#include "FaceDetector.cpp"

// #include "Features.cpp"

using namespace std;

int evaluate(FaceDetector *fd, vector<pair<Image, int>> *trainingData)
{
    int correct = 0, allNegatives = 0, allPositives = 0, trueNegatives = 0, falseNegatives = 0, truePositives = 0, falsePositives = 0, prediction = -1;
    double classification_time = 0;

    printf("ASDASDASDASD");

    for (pair<Image, int> data : *trainingData)
    {
        if (data.second == 1)
            allPositives++;
        else
            allNegatives++;

        prediction = (*fd).classify(data.first);
        if (prediction == 1 && data.second == 0)
            falsePositives++;
        else if (prediction == 0 && data.second == 1)
            falseNegatives++;

        correct += (prediction == data.second ? 1 : 0);
    }

    printf("ASDASDASDASD 2222222");

    printf("False Positive Rate: %d/%d (%f)", falsePositives, allNegatives, falsePositives / allNegatives);
    printf("False Negative Rate: %d/%d (%f)", falseNegatives, allPositives, falseNegatives / allPositives);
    printf("Accuracy: %d/%d (%f)", correct, trainingData->size(), correct / trainingData->size());
    printf("Average Classification Time: %f", classification_time / trainingData->size());
}

int loadSamples(string path, vector<pair<Image, int>> *trainingData, int classId, int limit)
{
    FileReader fr(path);

    vector<vector<unsigned char>> sample;
    int count = 0;

    while (fr.remainingSamples())
    {
        int res = fr.getSample(&sample, count == 0);

        if (!res)
        {
            cout << "Error opening the file";
            continue;
        }

        Image img = Image(sample, sample.size());

        (*trainingData).push_back(make_pair(Image(sample, sample.size()), classId));

        if (++count == limit)
            break;
    }

    return count;
}

int loadSamples(string path, vector<pair<Image, int>> *trainingData, int classId)
{
    return loadSamples(path, trainingData, classId, -1);
}

int main(int argc, char *argv[])
{
    vector<pair<Image, int>> trainingData;

    int positiveSamples = loadSamples("./img/train/face/", &trainingData, 1, 1);
    cout << "OUT " << positiveSamples << endl;

    int negativeSamples = loadSamples("./img/train/non-face/", &trainingData, 0, 1);
    cout << "OUT " << positiveSamples << endl;

    cout << trainingData.size() << " <----------" << endl;

    FaceDetector fd = FaceDetector(10);
    fd.train(trainingData, positiveSamples, negativeSamples);
    // evaluate(&fd, &trainingData);
}
