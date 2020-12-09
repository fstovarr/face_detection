#include <iostream>
#include <vector>

#include "FileReader.cpp"
#include "IntegralImage.cpp"

#include "F5/Boosting.cpp"
#include <mpi.h>

using namespace std;

int loadSamples(string path, vector<pair<Image, int>> *trainingData, int init, int end, int classId, int limit)
{
    FileReader fr(path, init, end);

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


vector<vector<int>> applyFeaturesMPI(vector<matrix> const & xis, vector<Feature> const & features, int id) {

  vector<vector<int>> zs(features.size(), vector<int>(xis.size(), 0));



  return zs;
}

void MPImain(int argc, char *argv[], int total_n) {
  MPI_Status status;
  int id;
  int p;
  int ierr;
  //
  //  Initialize MPI.
  //
  ierr = MPI_Init ( &argc, &argv );

  if ( ierr != 0 ) {
    cout << "\n";
    cout << "COMMUNICATOR_MPI - Fatal error!";
    cout << "  MPI_Init returned nonzero ierr.\n";
    exit ( 1 );
  }
  //
  //  Get the number of processes.
  //
  ierr = MPI_Comm_size ( MPI_COMM_WORLD, &p );
  //
  //  Get the individual process ID.
  //
  ierr = MPI_Comm_rank ( MPI_COMM_WORLD, &id );
  //
  //  Process 0 prints an introductory message.
  //
  if ( id == 0 ) {
    freopen("output_mpi.csv", "a+", stdout);
    // cout << "\n";
    // cout << "COMMUNICATOR_MPI - Master process:\n";
    // cout << "  C++/MPI version\n";
    // cout << "  An MPI example program.\n";
    // cout << "\n";
    // cout << "  The number of processes is " << p << "\n";
    // cout << "\n";
  }

  // doesn't handle p < n, assume n >> p

  size_t chunkSize = total_n / min(total_n, p);

  int rangeStart = chunkSize * id;
  int rangeEnd = chunkSize * (id + 1);

  /*
  if ( id == p - 1 ) {
    rangeEnd = total_n;
  }
  */


  vector<pair<Image, int>> trainingData;

  string path = "/home/mpiuser/face_detection/face_detection";

  int positiveSamples = loadSamples(path + "/img/train/face/", &trainingData, rangeStart, rangeEnd, 1, total_n);
  int negativeSamples = loadSamples(path + "/img/train/non-face/", &trainingData, rangeStart, rangeEnd, 0, total_n);

  // cout << "process: " << id << "\n";
  // cout << positiveSamples + negativeSamples << "\n";

  size_t N = trainingData.size();
  vector<matrix> X(N);
  vector<int> y(N);

  for (size_t i = 0; i < N; ++i) {
    X[i] = to_integral(trainingData[i].first.getIntImage());
    y[i] = trainingData[i].second;
  }

  int window_size = X[0].size();
  vector<Feature> features = getFeatures(window_size - 1);
  size_t num_features = features.size();


  vector<int> flattened(chunkSize * num_features);

  auto start = chrono::high_resolution_clock::now();
  for (size_t i = 0; i < num_features; ++i) {
    for (size_t j = 0; j < chunkSize; ++j) {
      flattened[i * chunkSize + j] = features[i](X[j]);
    }
  }
  auto stop = chrono::high_resolution_clock::now();
  auto duration = chrono::duration_cast<chrono::duration<double>>(stop - start);

  // cout << "Process: " << id << "\n";
  // cout << "Duration: " << duration.count() << "\n";

  vector<int> complete;
  if (id == 0) {
    complete = vector<int>(p * chunkSize * num_features);
    MPI_Gather(
      &flattened[0], //send_data
      chunkSize * num_features, // send_count
      MPI_INT, // send_datatype
      &complete[0], // recv_data
      chunkSize * num_features, // recv_count
      MPI_INT, // recv_datatype
      0,
      MPI_COMM_WORLD);


  } else {
    MPI_Gather(
      &flattened[0], //send_data
      chunkSize * num_features, // send_count
      MPI_INT, // send_datatype
      NULL, // recv_data
      chunkSize * num_features, // recv_count
      MPI_INT, // recv_datatype
      0,
      MPI_COMM_WORLD);
  }


  if (id == 0) {
    stop = chrono::high_resolution_clock::now();
    duration = chrono::duration_cast<chrono::duration<double>>(stop - start);
    cout << p << ", " << duration.count() << endl;
    // cout << "Finished receiving!\n" << duration.count();
  }


  //
  //  Terminate MPI.
  //
  ierr = MPI_Finalize ( );
  //
  //  Terminate
  //
  if ( id == 0 )
  {
    // cout << "\n";
    // cout << "COMMUNICATOR_MPI:\n";
    // cout << "  Normal end of execution.\n";
    // cout << "\n";
  }
}

int main(int argc, char *argv[]) {
    bool useMPI = true;

    int n = 500;

    if (useMPI) {
      MPImain(argc, argv, n);
      return 0;
    }
    vector<pair<Image, int>> trainingData;


    int positiveSamples = loadSamples("./img/train/face/", &trainingData, 1, n);

    int negativeSamples = loadSamples("./img/train/non-face/", &trainingData, 0, n);

    size_t N = trainingData.size();
    vector<matrix> X(N);
    vector<int> y(N);

    for (size_t i = 0; i < N; ++i) {
      X[i] = to_integral(trainingData[i].first.getIntImage());
      y[i] = trainingData[i].second;
    }

    int window_size = X[0].size();
    vector<Feature> features = getFeatures(window_size - 1);


    return 0;

}
