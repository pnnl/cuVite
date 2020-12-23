#ifndef LOUVAIN_CUDA_H
#define LOUVAIN_CUDA_H

#include "louvain.hpp"
#include "louvain_cuda_constants.cuh"
// #include "louvain_cuda_struct.hpp"

#include <cuda.h>

#define ASSERT(x)    if (!(x))  { printf("Assert Failed! <%s:%d>\n", __FILE__, __LINE__);  }

#define CUDA_SAFE(x) if ( cudaSuccess != (x) ) { printf("CUDA CALL FAILED AT %d\n", __LINE__ ); exit(1);}
#define CUDA_SAFE_MALLOC(DP, SIZE)  (cudaMalloc((void**)&DP, SIZE))

#define WEIGHTED   0
#define UNWEIGHTED 1

static __inline__ __device__ double my_func_atomicAdd(double *address, double val);
// #if !defined(__cuda_arch__) || __cuda_arch__ >= 600
#if __cuda_arch__ >= 600

#else
static __inline__ __device__ double my_func_atomicAdd(double *address, double val) {
  unsigned long long int* address_as_ull = (unsigned long long int*)address;
  unsigned long long int old = *address_as_ull, assumed;
  if (val==0.0)
    return __longlong_as_double(old);
  do {
    assumed = old;
    old = atomicCAS(address_as_ull, assumed, __double_as_longlong(val +__longlong_as_double(assumed)));
  } while (assumed != old);
  return __longlong_as_double(old);
}

static __inline__ __device__ long my_func_atomicAdd(long *address, int val) {
  unsigned long long int* address_as_ull = (unsigned long long int*)address;
  unsigned long long int old = *address_as_ull, assumed;
  if (val==0.0)
    return old;
  do {
    assumed = old;
    old = atomicCAS(address_as_ull, assumed, (unsigned long long int)(val +assumed));
  } while (assumed != old);
  return old;
}
#endif

void find_RemoteEdgeList_tail_comm (
    GraphElem size_edgeList, GraphElem size_remoteComm,
    GraphElem* graph_edgeList_tail, GraphWeight* graph_edgeList_weight,
    GraphElem* currComm, GraphElem* tcomm_vec,
    GraphElem* remoteComm_v, GraphElem* remoteComm_comm,
    const GraphElem base, const GraphElem bound);

#endif /* LOUVAIN_CUDA_H */
