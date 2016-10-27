#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <stdlib.h>
#define N 23 * 1024
#define ThreadsPerBlock 256
#define BlocksPerGrid (N + ThreadsPerBlock - 1) / ThreadsPerBlock

__global__ void addKernel(const int *a, int *r)
{
	__shared__ int cache[ThreadsPerBlock];
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	int cacheIndex = threadIdx.x;

	// copy data to shared memoryfrom global memory
	cache[cacheIndex] = a[tid];
	__syncthreads();

	// add these data using reduce
	for (int i = 1; i < blockDim.x; i *= 2)
	{
		if ((tid % (2 * i)) == 0)
		{
			cache[cacheIndex] += cache[cacheIndex + i];
		}
		__syncthreads();
	}

	// copy the result of reduce to global memory
	if (cacheIndex == 0)
		r[blockIdx.x] = cache[cacheIndex];
}

int main()
{
	int *a, *result;
	int *dev_a, *dev_result;
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	// allocate memory in CPU
	a = (int*)malloc(N * sizeof(int));
	result = (int*)malloc(BlocksPerGrid * sizeof(int));
	// allocate memory in GPU
	cudaMalloc((void**)&dev_a, N * sizeof(int));
	cudaMalloc((void**)&dev_result, BlocksPerGrid * sizeof(int));

	// generate data for a
	for (int i = 0; i < N; ++i)
		a[i] = 1;

	cudaEventRecord(start, 0);
	// copy data from CPU to GPU
	cudaMemcpy(dev_a, a, N * sizeof(int), cudaMemcpyHostToDevice);

	addKernel<<<BlocksPerGrid, ThreadsPerBlock>>>(dev_a, dev_result);

	// copy result from GPU to CPU
	cudaMemcpy(result, dev_result, BlocksPerGrid * sizeof(int), cudaMemcpyDeviceToHost);

	for (int i = 1; i < BlocksPerGrid; ++i)
		result[0] += result[i];
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	float elapsedTime;
	cudaEventElapsedTime(&elapsedTime, start, stop);

	printf("Is the result %d equal to %d?\n", result[0], N);
	printf("Time of computing is %.3lf", elapsedTime);

	// free
	cudaEventDestroy(start);
	cudaEventDestroy(stop);
	cudaFree(dev_a);
	cudaFree(dev_result);
	free(a);
	free(result);
}