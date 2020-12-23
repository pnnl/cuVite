#include "louvain_cuda.cuh"
// #include "louvain_cuda_cpp_interface.hpp"
#include "GpuGraph.cuh"

#include <cstring>
#include <sstream>

#include <sys/time.h>
#include <time.h>

#include <cooperative_groups.h>

// #define PRINT_HYBRID
// #define PRINT_TIMEDS

#define L_THREADBLOCK_SIZE 1024
#define M_THREADBLOCK_SIZE 512
#define S_THREADBLOCK_SIZE 256
namespace cg = cooperative_groups;
using namespace CuVite;

__device__ GraphWeight f_weight_reduce(
               cg::thread_group g, GraphWeight *x, GraphWeight val)
{
    int lane = g.thread_rank();
    // Each iteration halves the number of active threads
    // Each thread adds its partial sum[i] to sum[lane+i]
    for (int i = g.size() / 2; i > 0; i /= 2) {
        x[lane] = val;  g.sync();
        val += x[lane+i]; g.sync();
    }

    return val; // note: only thread 0 will return full sum
}

template<int tile_sz>
__global__
void compute_modularity(
    const GraphElem nv,
    GraphWeight* localCinfo_degree,
    GraphWeight* clusterWeight,
    GraphWeight* sumDegree,
    GraphWeight* sumWeight
)
{

  __shared__ GraphWeight shared_localCinfo_degree[L_THREADBLOCK_SIZE];
  __shared__ GraphWeight shared_clusterWeight[L_THREADBLOCK_SIZE];

  int ii = threadIdx.x;
  GraphElem i = blockIdx.x * blockDim.x + threadIdx.x ;

  shared_localCinfo_degree[ii] = 0.0;
  shared_clusterWeight[ii] = 0.0;

  /// Create cooperative groups
  auto thb_g = cg::this_thread_block();
  // auto tileIdx = thb_g.thread_rank()/tile_sz;
#if __cuda_arch__ >= 700
  auto tile = cg::partition<tile_sz>(cg::this_thread_block());
#else
  auto tile = cg::tiled_partition<tile_sz>(cg::this_thread_block());
#endif

  auto tileIdx = thb_g.thread_rank()/tile.size();
  unsigned ti = tileIdx*tile.size();

  if(i < nv) {
    shared_localCinfo_degree[ii] = localCinfo_degree[i];
    shared_clusterWeight[ii] = clusterWeight[i];
  }

  shared_localCinfo_degree[ii] *= shared_localCinfo_degree[ii];

  thb_g.sync();

  GraphWeight block_sum_weight = 0.0;
  GraphWeight block_sum_degree = 0.0;

  block_sum_weight = f_weight_reduce(thb_g, shared_clusterWeight,
        shared_clusterWeight[thb_g.thread_rank()]);

  block_sum_degree = f_weight_reduce(thb_g, shared_localCinfo_degree,
        shared_localCinfo_degree[thb_g.thread_rank()]);

  if(thb_g.thread_rank() == 0) {
#if __cuda_arch__ >= 600
    atomicAdd(&sumDegree[0], block_sum_degree);
    atomicAdd(&sumWeight[0], block_sum_weight);
#else
#ifndef USE_32_BIT_GRAPH
    my_func_atomicAdd(&sumDegree[0], block_sum_degree);
    my_func_atomicAdd(&sumWeight[0], block_sum_weight);
#else
    atomicAdd(&sumDegree[0], block_sum_degree);
    atomicAdd(&sumWeight[0], block_sum_weight);
#endif
#endif
  }
}

void gpu_reduce_mod(CommVector &localCinfo, const GraphWeightVector &clusterWeight, 
        GpuGraph &gpu_graph, GraphWeight le_la_xx[], const GraphElem nv) {

  GraphWeight* dev_clusterWeight = gpu_graph.get_clusterWeight();
  GraphWeight* dev_localCinfo_degree = gpu_graph.get_ModlocalCinfo_degree();

  gpu_graph.cpyVecTodev(clusterWeight, dev_clusterWeight);
  GraphWeight* temp_ModlocalCinfo_degree = gpu_graph.getPinned_ModlocalCinfo_degree(); 

  GraphWeight* dev_sumDegree = gpu_graph.get_sum_degree();
  GraphWeight* dev_sumWeight = gpu_graph.get_sum_weight();

  CUDA_SAFE(cudaMemset(dev_sumDegree, 0, sizeof(GraphWeight)));
  CUDA_SAFE(cudaMemset(dev_sumWeight, 0, sizeof(GraphWeight)));

#pragma omp parallel default(none), \
       shared(localCinfo, temp_ModlocalCinfo_degree)
#pragma omp for schedule(guided)
  for(int ii=0; ii<localCinfo.size(); ii++) {
    temp_ModlocalCinfo_degree[ii] = localCinfo[ii].degree;
  }

  gpu_graph.cpyArrayTodev(temp_ModlocalCinfo_degree, 
     dev_localCinfo_degree, localCinfo.size());

  dim3 numBlocks01( (nv-1) / M_THREADBLOCK_SIZE + 1);
  dim3 Block_dim01(M_THREADBLOCK_SIZE);

  compute_modularity<PHY_WRP_SZ><<<numBlocks01,Block_dim01>>>(
    nv,
    dev_localCinfo_degree,
    dev_clusterWeight,
    dev_sumDegree,
    dev_sumWeight);

  CUDA_SAFE(cudaMemcpy(&le_la_xx[0], dev_sumWeight,
     sizeof(GraphWeight), cudaMemcpyDeviceToHost));
  CUDA_SAFE(cudaMemcpy(&le_la_xx[1], dev_sumDegree,
     sizeof(GraphWeight), cudaMemcpyDeviceToHost));
}

