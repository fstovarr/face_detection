import os

max_threads = 15
experiments = 10

for threads in range(1, max_threads):
    for _ in range(experiments):
        os.system("mpirun -np {} --hostfile mpi-hosts ./main".format(int(threads)))