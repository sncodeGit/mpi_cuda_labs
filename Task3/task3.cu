#include <iostream>
#include "cuda.h"
#include "stdio.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

using namespace std;

#define N  10

__global__
void add11(int* v1, int* v2, int* res) {
	int i = 0;
	while (i < N) {
		res[i] = v1[i] + v2[i];
		i += 1;
	}
}

__global__
void addn1(int* v1, int* v2, int* res) {
	int i = blockIdx.x;
	res[i] = v1[i] + v2[i];
}

__global__
void add1n(int* v1, int* v2, int* res) {
	int i = threadIdx.x;
	res[i] = v1[i] + v2[i];
}

__global__
void add(int* v1, int* v2, int* res) {
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	int stride = blockDim.x * gridDim.x;
	for (int i = index; i < N; i += stride)
		res[i] = v1[i] + v2[i];
}

int main(void) {
	int* v1;
	int* v2;
	int* res;

	cudaMallocManaged(&a, N * sizeof(int));
	cudaMallocManaged(&b, N * sizeof(int));
	cudaMallocManaged(&c, N * sizeof(int));
	for (int i = 0; i < N; i++) {
		v1[i] = -i;
		v2[i] = i * i;
	}

	add11 << <1, 1 >> > (v1, v2, res);
	cudaDeviceSynchronize();
	cout << "1, 1:" << endl;
	for (int i = 0; i < N; i++) 
		cout << v1[i] << " + " << v2[i] << " = " << res[i] << endl;

	add1n << <1, N >> > (v1, v2, res);
	cudaDeviceSynchronize();
	
	cout << endl;
	cout << "1, N: " << endl;
	for (int i = 0; i < N; i++) 
		cout << v1[i] << " + " << v2[i] << " = " << res[i] << endl;

	addn1 << <N, 1 >> > (v1, v2, res);
	cudaDeviceSynchronize();

	cout << endl;
	cout << "N, 1: " << endl;
	for (int i = 0; i < N; i++)
		cout << v1[i] << " + " << v2[i] << " = " << res[i] << endl;

	int blockSize = 256;
	int numBlocks = (N + blockSize - 1) / blockSize;
	add << <numBlocks, blockSize >> > (a, b, c);
	cudaDeviceSynchronize();

	cout << endl;
	cout << "Blocks and threads: " << endl;
	for (int i = 0; i < N; i++) 
		cout << v1[i] << " + " << v2[i] << " = " << res[i] << endl;

	cudaFree(v1);
	cudaFree(v2);
	cudaFree(v3);
	return 0;
}