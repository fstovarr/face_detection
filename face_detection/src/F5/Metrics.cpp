#include <vector>
#include <assert.h>
#include <stdio.h>

using namespace std;
void evaluate(vector<int> y_true_s, vector<int> y_pred_s) {
  int N = y_true_s.size();
  assert(y_true_s.size() == y_pred_s.size());

  int correct = 0, allNegatives = 0, allPositives = 0, falseNegatives = 0, falsePositives = 0;

  for (int i = 0; i < N; ++i) {
    int y_true = y_true_s[i], y_pred = y_pred_s[i];
    if (y_true == 1)
      allPositives++;
    else
      allNegatives++;

    if (y_pred == 1 && y_true == 0)
      falsePositives++;
    else if (y_pred == 0 && y_true == 1)
      falseNegatives++;

    correct += (y_true == y_pred ? 1 : 0);
  }


  printf("False Positive Rate: %d/%d (%f)\n", falsePositives, allNegatives, falsePositives * 1. / allNegatives);
  printf("False Negative Rate: %d/%d (%f)\n", falseNegatives, allPositives, falseNegatives * 1. / allPositives);
  printf("Accuracy: %d/%d (%f)\n", correct, N, correct * 1. / N);
}
