# 归约算法

我们可以有以下两种方式实现归约算法：

不连续的方式：

![vector-reduce1](https://github.com/Sunlcy/My-blogs/blob/master/cuda/vector_reduce1.PNG)

连续的方式：

![vector-reduce2](https://github.com/Sunlcy/My-blogs/blob/master/cuda/vector_reduce2.PNG)

下面我们用具体的代码来实现上述两种方法。

```
// 非连续的归约求和
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
```
上述代码实现的是非连续的归约求和，从`int index = 2 * i * cacheIndex`和`cache[index] += cache[index + i];`两条语句，我们可以很容易判断这种实现方式会产生bank冲突。当`i=1`时，步长s=2xi=2，会产生两路的bank冲突；当`i=2`时，步长s=2xi=4，会产生四路的bank冲突...当`i=n`时，步长s=2xn=2n。可以看出每一次步长都是偶数，因此这种方式会产生严重的bank冲突。

**NOTE:**在《GPU高性能运算之CUDA》这本书中对实现不连续的归约算法有两种代码实现方式，但笔者发现书中的提到(p179)的两种所谓相同计算逻辑的函数`reduce0`和`reduce1`，其实具有本质上的不同。前者不会发生bank冲突，而后者(即本文中所使用的)才会产生bank冲突。由于前者线程ID要求的条件比较“苛刻”，只有满足`tid % (2 * s) == 0`的线程才会执行求和操作(`sdata[tid]+=sdata[tid+i`])；而后者只要满足index(`2 * s * tid`，即线程ID的2xs倍)小于线程块的大小(`blockDim.x`)即可。总之，前者在进行求和操作(`sdata[tid]+=sdata[tid+i`])时，线程的使用同样是不连续的，即当`s=1`时，线程编号为0,2,4,...,1022；而后者的线程使用是连续的，即当`s=1`时，前512个线程(0,1,2,...,511)在进行求和操作(`sdata[tid]+=sdata[tid+i`])，而后512个线程是闲置的。前者不会出现多个线程访问同一bank的不同字地址，而后者正如书中所说会产生严重的bank冲突。当然这些只是笔者的想法，如有不同，欢迎来与我讨论，邮箱:<chaoyanglius@outlook.com>。

```
// 连续的归约求和
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
```

由于每个线程的ID与操作的数据编号一一对应，因此上述的代码很明显不会产生bank冲突。
