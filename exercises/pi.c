// gcc -o pi -fopenmp pi.c -lpthread  && ./pi
#define _GNU_SOURCE         /* See feature_test_macros(7) */
#include <string.h>
#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include <fcntl.h>           /* For O_* constants */
#include <sys/stat.h>        /* For mode constants */
#include <unistd.h>
#include <sys/time.h>

#define THREADS 4
#define ITERATIONS 1000000000

struct Data
{
	double pi;
	int init;
	int end;
};

void calc(struct Data *arr) {
	double partialPi = 0;

	for(int i = arr->init; i < arr->end; i++)
		partialPi += (i % 2 ? -1 : 1) * (4.0 / (2.0 * i + 1.0));

	mempcpy(&arr->pi, &partialPi, sizeof(partialPi));
}

int main() {
	struct timeval after, before, result;
    gettimeofday(&before, NULL);

	pthread_t *ids = calloc(THREADS, sizeof(pthread_t));
	int chunk = ITERATIONS / THREADS;

	void *d = (struct Data*) calloc(THREADS, sizeof(struct Data));
	double pi = 0.0;
	int init, end;

	void *tmp;

	for(int i = 0; i < THREADS; i++) 
	{
		tmp = (d + i * sizeof(struct Data));
		init = i * chunk;
		end = (i + 1) * chunk;	

		mempcpy(tmp, &pi, sizeof(pi));
		mempcpy(tmp + sizeof(pi), &init, sizeof(init));
		mempcpy(tmp + sizeof(pi) + sizeof(end), &end, sizeof(end));

		pthread_create(&ids[i], NULL, (void *(*)(void *)) calc, tmp);
	}

	for(int i = 0; i < THREADS; i++) 
        pthread_join(ids[i], NULL);

	for(int i = 0; i < THREADS; i++) {
		tmp = (d + i * sizeof(struct Data));
		pi += *((double *) tmp);
	}

	free(d);
	free(ids);

	gettimeofday(&after, NULL);
    timersub(&after, &before, &result);

	printf("%d, %.20f, %ld.%06ld\n", THREADS, pi, (long int) result.tv_sec, (long int) result.tv_usec);
}
