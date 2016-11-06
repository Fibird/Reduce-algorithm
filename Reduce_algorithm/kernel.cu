#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <stdlib.h>
#define N 50 * 1024
#define ThreadsPerBlock 256
#define BlocksPerGrid (N + ThreadsPerBlock - 1) / ThreadsPerBlock

// kernel 1 without bank conflict 
__global__ void NBC_addKernel1(const int *a, int *r)
{
	__shared__ int cache[ThreadsPerBlock];
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	int cacheIndex = threadIdx.x;

	// copy data to shared memory from global memory
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

// kernel 2 without bank conflict 
__global__ void NBC_addKernel2(const int *a, int *r)
{
	__shared__ int cache[ThreadsPerBlock];
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	int cacheIndex = threadIdx.x;

	// copy data to shared memory from global memory
	cache[cacheIndex] = a[tid];
	__syncthreads();

	// add these data using reduce
	for (int i = blockDim.x / 2; i > 0; i /= 2)
	{
		if (cacheIndex < i)
		{
			cache[cacheIndex] += cache[cacheIndex + i];
		}
		__syncthreads();
	}

	// copy the result of reduce to global memory
	if (cacheIndex == 0)
		r[blockIdx.x] = cache[cacheIndex];
}

// kernel with bank conflict
__global__ void BC_addKernel(const int *a, int *r)
{
	__shared__ int cache[ThreadsPerBlock];
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	int cacheIndex = threadIdx.x;

	// copy data to shared memory from global memory
	cache[cacheIndex] = a[tid];
	__syncthreads();

	// add these data using reduce
	for (int i = 1; i < blockDim.x; i *= 2)
	{
		int index = 2 * i * cacheIndex;
		if (index < blockDim.x)
		{
			cache[index] += cache[index + i];
		}
		__syncthreads();
	}

	// copy the result of reduce to global memory
	if (cacheIndex == 0)
		r[blockIdx.x] = cache[cacheIndex];
}

float reduce_sum1(int *a, int *result)
{
	int *dev_a, *dev_result;
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	// allocate memory in GPU
	cudaMalloc((void**)&dev_a, N * sizeof(int));
	cudaMalloc((void**)&dev_result, BlocksPerGrid * sizeof(int));
	cudaEventRecord(start, 0);
	// copy data from CPU to GPU
	cudaMemcpy(dev_a, a, N * sizeof(int), cudaMemcpyHostToDevice);

	BC_addKernel<<<BlocksPerGrid, ThreadsPerBlock>>>(dev_a, dev_result);

	// copy result from GPU to CPU
	cudaMemcpy(result, dev_result, BlocksPerGrid * sizeof(int), cudaMemcpyDeviceToHost);
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	float elapsedTime;
	cudaEventElapsedTime(&elapsedTime, start, stop);
	// free GPU memory
	cudaFree(dev_a);
	cudaFree(dev_result);
	return elapsedTime;
}

float reduce_sum2(int *a, int *result)
{
	int *dev_a, *dev_result;
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	// allocate memory in GPU
	cudaMalloc((void**)&dev_a, N * sizeof(int));
	cudaMalloc((void**)&dev_result, BlocksPerGrid * sizeof(int));
	cudaEventRecord(start, 0);
	// copy data from CPU to GPU
	cudaMemcpy(dev_a, a, N * sizeof(int), cudaMemcpyHostToDevice);

	NBC_addKernel1<<<BlocksPerGrid, ThreadsPerBlock>>>(dev_a, dev_result);

	// copy result from GPU to CPU
	cudaMemcpy(result, dev_result, BlocksPerGrid * sizeof(int), cudaMemcpyDeviceToHost);
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	float elapsedTime;
	cudaEventElapsedTime(&elapsedTime, start, stop);
	// free GPU memory
	cudaFree(dev_a);
	cudaFree(dev_result);
	return elapsedTime;
}

float reduce_sum3(int *a, int *result)
{
	int *dev_a, *dev_result;
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	// allocate memory in GPU
	cudaMalloc((void**)&dev_a, N * sizeof(int));
	cudaMalloc((void**)&dev_result, BlocksPerGrid * sizeof(int));
	cudaEventRecord(start, 0);
	// copy data from CPU to GPU
	cudaMemcpy(dev_a, a, N * sizeof(int), cudaMemcpyHostToDevice);

	NBC_addKernel2<<<BlocksPerGrid, ThreadsPerBlock>>>(dev_a, dev_result);

	// copy result from GPU to CPU
	cudaMemcpy(result, dev_result, BlocksPerGrid * sizeof(int), cudaMemcpyDeviceToHost);
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	float elapsedTime;
	cudaEventElapsedTime(&elapsedTime, start, stop);
	// free GPU memory
	cudaFree(dev_a);
	cudaFree(dev_result);

	return elapsedTime;
}

int main()
{
	int *a, *result;
	float elapsedTime;

	// allocate memory in CPU
	a = (int*)malloc(N * sizeof(int));
	result = (int*)malloc(BlocksPerGrid * sizeof(int));

	// generate data for a
	for (int i = 0; i < N; ++i)
		a[i] = 1;
/**********************Reduce V1***************************/	
	// reduce sum
	elapsedTime = reduce_sum1(a, result);
	for (int i = 1; i < BlocksPerGrid; ++i)
		result[0] += result[i];

	// print final result and time
	printf("***Result of Reduce sum V1***\n");
	printf("Is the result %d equal to %d?\n", result[0], N);
	printf("Time of computing is %.3lfms\n", elapsedTime);
/**********************Reduce V2***************************/
	// reduce sum
	elapsedTime = reduce_sum2(a, result);
	for (int i = 1; i < BlocksPerGrid; ++i)
		result[0] += result[i];

	// print final result and time
	printf("***Result of Reduce sum V2***\n");
	printf("Is the result %d equal to %d?\n", result[0], N);
	printf("Time of computing is %.3lfms\n", elapsedTime);
/**********************Reduce V3***************************/
	// reduce sum
	elapsedTime = reduce_sum3(a, result);
	for (int i = 1; i < BlocksPerGrid; ++i)
		result[0] += result[i];

	// print final result and time
	printf("***Result of Reduce sum V3***\n");
	printf("Is the result %d equal to %d?\n", result[0], N);
	printf("Time of computing is %.3lfms\n", elapsedTime);
	// free
	free(a);
	free(result);
}