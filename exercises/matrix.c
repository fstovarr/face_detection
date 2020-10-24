// gcc -o pi -fopenmp omp_hello.c -lpthread 
#define _GNU_SOURCE         /* See feature_test_macros(7) */
#include <string.h>
#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include <fcntl.h>           /* For O_* constants */
#include <sys/stat.h>        /* For mode constants */
#include <unistd.h>
#include <sys/time.h>

int main() {
	int b = -0;

	for(int i = 0; i < 5; i++) {
		printf("TEST %d\n", i || -1);
	}
	// struct timeval after, before, result;
    // gettimeofday(&before, NULL);

    // void *m1 = (double *) calloc(SIZE, sizeof(double));
    // void *m2 = (double *) calloc(SIZE, sizeof(double));

    // int tmp = 0;
    // for(int i = 0; i < SIZE; i++) {
    //     tmp = SIZE - i;
    //     mempcpy(m1 + i * sizeof(double), &i, sizeof(double));
    //     mempcpy(m2 + i * sizeof(double), &i, sizeof(double));
    // }

	// pthread_t *ids = calloc(THREADS, sizeof(pthread_t));
	// int chunk = SIZE / THREADS;

	// void *d = (struct Data*) calloc(THREADS, sizeof(struct Data));
	// double pi = 0.0;
	// int init, end;

	// void *tmp;

	// for(int i = 0; i < THREADS; i++) 
	// {
	// 	tmp = (d + i * sizeof(struct Data));
	// 	init = i * chunk;
	// 	end = (i + 1) * chunk;	

	// 	mempcpy(tmp, &pi, sizeof(pi));
	// 	mempcpy(tmp + sizeof(pi), &init, sizeof(init));
	// 	mempcpy(tmp + sizeof(pi) + sizeof(end), &end, sizeof(end));

	// 	pthread_create(&ids[i], NULL, (void *(*)(void *)) calc, tmp);
	// }

	// for(int i = 0; i < THREADS; i++) 
    //     pthread_join(ids[i], NULL);

	// for(int i = 0; i < THREADS; i++) {
	// 	tmp = (d + i * sizeof(struct Data));
	// 	pi += *((double *) tmp);
	// }

	// free(d);
	// free(ids);

	// gettimeofday(&after, NULL);
    // timersub(&after, &before, &result);

	// printf("%d, %.20f, %ld.%06ld\n", THREADS, pi, (long int) result.tv_sec, (long int) result.tv_usec);
}
