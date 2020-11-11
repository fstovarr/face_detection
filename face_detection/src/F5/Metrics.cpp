#include <vector>
#include <assert.h>
#include <stdio.h>
#include <iostream>
#include <string.h>

using namespace std;
void evaluate(vector<int> const & y_true_s, vector<int> const & y_pred_s) {
  int N = y_true_s.size();
  assert(y_true_s.size() == y_pred_s.size());

  int conf[4][4];
  for (int i = 0; i < 2; ++i) {
    for (int j = 0; j < 2; ++j) {
      conf[i][j] = 0;
    }
  }

  for (int i = 0; i < N; ++i) {
    int y_true = y_true_s[i], y_pred = y_pred_s[i];
    conf[y_true][y_pred] += 1;
  }

  int allNegatives = conf[0][1] + conf[0][0], allPositives = conf[1][0] + conf[1][1];
  int correct = conf[0][0] + conf[1][1];

  // for (int i = 0; i < 2; ++i) {
  //   for (int j = 0; j < 2; ++j) {
  //     cout << conf[i][j] << "\t";
  //   }
  //   cout << "\n";
  // }


  // printf("allNegatives: %d allPositives: %d\n", allNegatives, allPositives);

  // printf("False Positive Rate: %d/%d (%f)\n", conf[0][1], allNegatives, conf[0][1] * 1. / allNegatives);
  // printf("False Negative Rate: %d/%d (%f)\n", conf[1][0], allPositives, conf[0][1] * 1. / allPositives);
  // printf("Accuracy: %d/%d (%f)\n", correct, N, correct * 1. / N);
  // printf("Recall: %d/%d (%f)\n", conf[1][1], allPositives, conf[1][1] * 1. / N);
}
