#include "louvain_cuda.cuh"
#include "louvain_cuda_cpp_interface.hpp"
#include "GpuGraph.cuh"

#include <cstring>
#include <sstream>

#include <sys/time.h>
#include <time.h>

#include <cooperative_groups.h>

// #define PRINT_HYBRID
// #define PRINT_TIMEDS
// #define USE_HYBRID_CPU_GPU

#define L_THREADBLOCK_SIZE 512
#define M_THREADBLOCK_SIZE 256
// #define S_THREADBLOCK_SIZE 128
#define S_THREADBLOCK_SIZE 640
// #define MS_THREADBLOCK_SIZE 32
#define MS_THREADBLOCK_SIZE 512
#define ARRAY_REDUCE_THREADBLOCK_SIZE 32

// #define S_BLOCK_TILE ( ARRAY_REDUCE_THREADBLOCK_SIZE / PHY_WRP_SZ ) 
#define S_BLOCK_TILE ( S_THREADBLOCK_SIZE / PHY_WRP_SZ ) 
#define MS_BLOCK_TILE ( MS_THREADBLOCK_SIZE / PHY_WRP_SZ ) 

#define FINDING_UNIQCOMM_BLOCK_TILE ( FINDING_UNIQCOMM_BLOCK_SIZE / PHY_WRP_SZ ) 

#define CUT_SIZE_NUM_EDGES2 9400000000
// #define CUT_SIZE_NUM_EDGES1 4096
// #define CUT_SIZE_NUM_EDGES12 4096
#define CUT_SIZE_NUM_EDGES1 4096
#define CUT_SIZE_NUM_EDGES12 4096
// #define CUT_SIZE_NUM_EDGES1 3310720

namespace cg = cooperative_groups;
using namespace CuVite;

double timer ( void )
{
    struct timeval tv;
    struct timezone tz;
    gettimeofday ( &tv, &tz );
    return (tv.tv_sec +  0.000001 * tv.tv_usec);
}

void updateLocalTarget_gpu (
    GraphElem nv, 
    const CommunityVector &currComm,
    CommunityVector &targetComm,
    const GraphWeightVector &vDegree,
    CommMap &remoteCupdate,
    CommunityVector &temp_targetComm,
    CommVector &localCupdate,
    const GraphElem base, const GraphElem bound, int numIters
     ) {

    // omp_set_num_threads(7);
#pragma omp parallel default(none), shared(nv, localCupdate, currComm, \
        targetComm, vDegree, remoteCupdate, \
        temp_targetComm, numIters)
#pragma omp for schedule(guided)
  for (int i = 0; i < nv; i++) {
    GraphElem localTarget, ModlocalTarget;
    bool currCommIsLocal;
    bool targetCommIsLocal;

    GraphElem cc = currComm[i];
    if(cc >= base && cc < bound) currCommIsLocal = true;
    ModlocalTarget = temp_targetComm[i];
    localTarget = ModlocalTarget;
    /// is the Target Local?
    if (ModlocalTarget >= base && ModlocalTarget < bound) targetCommIsLocal = true;

    /// Modify current if >= bound using stored map
    if (cc < base || cc >= bound) {
      currCommIsLocal = false;
    } 

    /// Modify ModlocalTarget if >= bound using stored map
    /// Stored map is no more required 
    if (ModlocalTarget < base || ModlocalTarget >= bound) {
      targetCommIsLocal = false;
    }
    // std::cout << "GPU i[" << i << "]; cc[" << cc << "]; localTarget[" 
    //    << localTarget << "]" << std::endl;

    // current and target comm are local - atomic updates to vectors
    if((localTarget != cc) && (localTarget != -1) && 
      currCommIsLocal && targetCommIsLocal) {
#ifdef DEBUG_PRINTF      
      assert( base < localTarget < bound);
      assert( base < cc < bound);
      assert( cc - base < localCupdate.size());   
      assert( (localTarget - base) < (GraphElem)localCupdate.size());   
#endif

      #pragma omp atomic update
      localCupdate[localTarget-base].degree += vDegree[i];
      #pragma omp atomic update
      localCupdate[localTarget-base].size++;
      #pragma omp atomic update
      localCupdate[cc-base].degree -= vDegree[i];
      #pragma omp atomic update
      localCupdate[cc-base].size--;
    } 

    /// current is local, target is not - do atomic on local, 
    /// accumulate in Maps for remote
    if ((localTarget != cc) && (localTarget != -1) && 
       currCommIsLocal && !targetCommIsLocal) {
      #pragma omp atomic update
      localCupdate[cc-base].degree -= vDegree[i];
      #pragma omp atomic update
      localCupdate[cc-base].size--;

      /// Search target in remoteCupdate 
      CommMap::iterator iter=remoteCupdate.find(localTarget);
      #pragma omp atomic update
      iter->second.degree += vDegree[i];
      #pragma omp atomic update
      iter->second.size++;
    }

    /// current is remote, target is local 
    /// accumulate for current, atomic on local
    if ((localTarget != cc) && (localTarget != -1) && 
      !currCommIsLocal && targetCommIsLocal) {
      #pragma omp atomic update
      localCupdate[localTarget-base].degree += vDegree[i];
      #pragma omp atomic update
      localCupdate[localTarget-base].size++;

      /// Search in remoteCupdate 
      CommMap::iterator iter=remoteCupdate.find(cc);

      #pragma omp atomic update
      iter->second.degree -= vDegree[i];
      #pragma omp atomic update
      iter->second.size--;
    }

    /// Current and target are remote - accumulate for both
    if ((localTarget != cc) && (localTarget != -1) && 
      !currCommIsLocal && !targetCommIsLocal) {
      
      // search current 
      CommMap::iterator iter=remoteCupdate.find(cc);
  
      #pragma omp atomic update
      iter->second.degree -= vDegree[i];
      #pragma omp atomic update
      iter->second.size--;
   
      // search target
      iter=remoteCupdate.find(localTarget);
  
      #pragma omp atomic update
      iter->second.degree += vDegree[i];
      #pragma omp atomic update
      iter->second.size++;
    }
#ifdef DEBUG_PRINTF
    assert(localTarget != -1);
#endif
    targetComm[i] = localTarget;
    // std::cout << "GPU i[" << i << "]; cc[" << cc << "]; localTarget[" 
    //    << localTarget << "]" << std::endl;
  }

}

__device__ GraphWeight weight_reduce(
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

template <int tile_sz>
__device__ GraphWeight weight_reduce_sum_tile_shfl(
               cg::thread_block_tile<tile_sz> g, GraphWeight val)
{
    // Each iteration halves the number of active threads
    // Each thread adds its partial sum[i] to sum[lane+i]
    for (int i = g.size() / 2; i > 0; i /= 2) {
        val += g.shfl_down(val, i);
    }

    return val; // note: only thread 0 will return full sum
}

#ifndef USE_32_BIT_GRAPH
template <int tile_sz>
__device__ GraphElem reduce_sum_tile_shfl(
               cg::thread_block_tile<tile_sz> g, GraphElem val)
{
    // Each iteration halves the number of active threads
    // Each thread adds its partial sum[i] to sum[lane+i]
    for (int i = g.size() / 2; i > 0; i /= 2) {
        val += g.shfl_down(val, i);
    }

    return val; // note: only thread 0 will return full sum
}
#endif

template <int tile_sz>
__device__ int reduce_sum_tile_shfl(
               cg::thread_block_tile<tile_sz> g, int val)
{
    // Each iteration halves the number of active threads
    // Each thread adds its partial sum[i] to sum[lane+i]
    for (int i = g.size() / 2; i > 0; i /= 2) {
        val += g.shfl_down(val, i);
    }

    return val; // note: only thread 0 will return full sum
}

template<int tile_sz>
__global__
void computeMaxIndex_large_thb(
    GraphElem nv, 
    GraphElem* ocurrComm, GraphElem* currComm, 
    GraphElem* localCinfo_size, GraphWeight* localCinfo_degree,
    GraphElem* localCinfo_oComm, 
    GraphWeight* selfLoop,
    GraphElem* uniq_clus_vec, GraphWeight* counter, 
    GraphElem* clmap_loc,
    GraphElem* clmap_comm, GraphWeight* clmap_weight, 
    GraphWeight* vDegree_vec,
    GraphElem* localTarget,
    GraphWeight* clusterWeight,
    const double constant,
    const GraphElem base, const GraphElem bound
)
{

  __shared__ int shared_num_uniq_cl[S_THREADBLOCK_SIZE];
  __shared__ GraphWeight t_my_counter[S_THREADBLOCK_SIZE];
  __shared__ GraphWeight t_shared_curGain[S_THREADBLOCK_SIZE];
  __shared__ GraphElem t_shared_Index[S_THREADBLOCK_SIZE];
  __shared__ GraphElem t_shared_Size[S_THREADBLOCK_SIZE];
  __shared__ GraphWeight t_shared_maxGain[S_BLOCK_TILE];
  __shared__ GraphElem t_shared_maxIndex[S_BLOCK_TILE];
  __shared__ GraphElem t_shared_maxSize[S_BLOCK_TILE];

  int ii = threadIdx.x;
  GraphElem i = blockIdx.x * blockDim.x + threadIdx.x ;

  shared_num_uniq_cl[ii] = 0;
  if(i < nv) {
    shared_num_uniq_cl[ii] = uniq_clus_vec[i]; 
  }

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

  GraphWeight *my_counter = &t_my_counter[tileIdx * tile.size()];
  GraphWeight *shared_curGain = &t_shared_curGain[tileIdx * tile.size()];
  GraphElem *shared_Index = &t_shared_Index[tileIdx * tile.size()];
  GraphElem *shared_Size = &t_shared_Size[tileIdx * tile.size()];
  GraphWeight *shared_maxGain = &t_shared_maxGain[tileIdx];
  GraphElem *shared_maxIndex = &t_shared_maxIndex[tileIdx];
  GraphElem *shared_maxSize = &t_shared_maxSize[tileIdx];

  GraphElem num_cluster = 0;
  GraphWeight ay = 0.0, eiy = 0.0;
  GraphWeight eix;
  GraphElem size;
  GraphWeight curGain = 0.0;
  GraphWeight vDegree, ax;
  GraphElem cc;
  // GraphWeight my_counter;
  GraphWeight currDegree;

  tile.sync();
  cg::sync(thb_g); 
  for( int wii = 0; wii < thb_g.size(); wii++) {
    // num_cluster = shared_num_uniq_cl[tileIdx*tile.size()+wii];
    num_cluster = shared_num_uniq_cl[wii];
    // if(num_cluster >= (GraphElem)tile_sz) 
    if(num_cluster > CUT_SIZE_NUM_EDGES12) {
// if(tile.thread_rank() == 0) printf("num_cluster[%ld]; \n",
//      num_cluster);
      if(tile.thread_rank() == 0) shared_maxGain[0] = 0.0;
      tile.sync();
      __syncwarp();
      shared_Index[tile.thread_rank()] = 0;
      // GraphElem ver_loc = (GraphElem)(blockIdx.x*blockDim.x+tileIdx*tile.size()+wii);
      GraphElem ver_loc = (GraphElem)(blockIdx.x*blockDim.x+wii);
      cc = currComm[ver_loc]; 
      if(tile.thread_rank() == 0) 
      {
        if(cc >= bound) {
          shared_maxIndex[0] = ocurrComm[ver_loc];
        } else {
          shared_maxIndex[0] = cc;
        }
        shared_maxSize[0] = localCinfo_size[cc - base];
      }
      tile.sync();
      __syncwarp();
      // my_counter[tile.thread_rank()] = counter[blockIdx.x*blockDim.x+tileIdx*tile.size()+wii]; 
      my_counter[tile.thread_rank()] = counter[blockIdx.x*blockDim.x+wii]; 
      eix = my_counter[tile.thread_rank()] - selfLoop[ver_loc];
      vDegree = vDegree_vec[ver_loc];
      currDegree = localCinfo_degree[cc - base];
      ax = currDegree - vDegree;
      // for(int k = 0; k < ((num_cluster-1)/tile.size()+1); k++)
      for(int k = 0; k < ((num_cluster-1)/thb_g.size()+1); k++)
      {
        // GraphElem thread_indx = k*tile.size() + tile.thread_rank();
        GraphElem thread_indx = k*thb_g.size() + thb_g.thread_rank();
        shared_Index[tile.thread_rank()] = -1;
        shared_curGain[tile.thread_rank()] = 0.0;
        if(thread_indx < num_cluster) {
          cg::coalesced_group active = cg::coalesced_threads();
          GraphElem tcomm = clmap_comm[
             clmap_loc[ver_loc]+thread_indx];
          ay = localCinfo_degree[tcomm - base];
          eiy = clmap_weight[
             clmap_loc[ver_loc]+thread_indx];
          curGain = 2.0 * (eiy - eix) - 2.0 * vDegree * (ay - ax) * constant; 
          shared_Size[tile.thread_rank()] = localCinfo_size[tcomm - base];
          if(tcomm >= bound) {
            shared_Index[tile.thread_rank()] = localCinfo_oComm[tcomm - bound];
          } else {
            shared_Index[tile.thread_rank()] = tcomm;
          }

          if((curGain > shared_maxGain[0]) && tcomm != cc ||
             (curGain == shared_maxGain[0] && curGain != 0.0 && tcomm != cc &&
              shared_Index[tile.thread_rank()] < shared_maxIndex[0]) ) {
            shared_curGain[tile.thread_rank()] = curGain;
            shared_Size[tile.thread_rank()] = localCinfo_size[tcomm - base]; 
          } else {
            shared_Index[tile.thread_rank()] = -1;
            shared_curGain[tile.thread_rank()] = 0.0;
            shared_Size[tile.thread_rank()] = 0;
          }
          /// Perform reduction
          active.sync(); 
#pragma unroll
          for (int s =1; s < tile.size(); s *=2) 
          {
            int indx = 2 * s * tile.thread_rank();
            // int indx = tileIdx*tile.size() + 2 * s * tile.thread_rank();
    
            if(indx < tile.size()) {
              if(shared_Index[indx+s] != -1) {
                if((shared_curGain[indx+s] > shared_curGain[indx]) ||
                    (shared_Index[indx] == -1) ||
                   (shared_curGain[indx+s] == shared_curGain[indx] &&
                   shared_curGain[indx+s] != 0.0 && shared_Index[indx] != -1 &&
                   shared_Index[indx+s] < shared_Index[indx])
                  ) {
                  shared_curGain[indx] = shared_curGain[indx+s];
                  shared_Index[indx] = shared_Index[indx+s];
                  shared_Size[indx] = shared_Size[indx+s];
                }
              } else if(shared_Index[indx+s] == -1 && shared_Index[indx] == -1) {
                  shared_curGain[indx] = 0.0; 
                  shared_Index[indx] = -1;
                  shared_Size[indx] = 0;
              }
            }
            active.sync();
          }
          if(tile.thread_rank() == 0) {
            if(shared_curGain[0] > shared_maxGain[0] ||
               shared_curGain[0] == shared_maxGain[0] &&
               shared_curGain[0] != 0.0 &&
               shared_Index[0] < shared_maxIndex[0]
              ) {
              shared_maxGain[0] = shared_curGain[0];
              shared_maxIndex[0] = shared_Index[0];
              shared_maxSize[0] = shared_Size[0];
            }
          }
        active.sync();
        }
        tile.sync();
        __syncwarp();
      }
      thb_g.sync();
      /// Perform reduction at threadblock level
      for (int s =1; s < S_BLOCK_TILE; s *=2) 
      {
        int indx = 2 * s * thb_g.thread_rank();

        if(indx < S_BLOCK_TILE) {
          // active = cg::coalesced_threads();
          if(t_shared_maxIndex[indx+s] != -1) {
            if((t_shared_maxGain[indx+s] > t_shared_maxGain[indx]) ||
                (t_shared_maxIndex[indx] == -1) ||
               (t_shared_maxGain[indx+s] == t_shared_maxGain[indx] &&
               t_shared_maxGain[indx+s] != 0.0 && t_shared_maxIndex[indx] != -1 &&
               t_shared_maxIndex[indx+s] < t_shared_maxIndex[indx])
              ) {
              t_shared_maxGain[indx] = t_shared_maxGain[indx+s];
              t_shared_maxIndex[indx] = t_shared_maxIndex[indx+s];
              t_shared_maxSize[indx] = t_shared_maxSize[indx+s];
            }
          } else if(t_shared_maxIndex[indx+s] == -1 && t_shared_maxIndex[indx] == -1) {
              t_shared_maxGain[indx] = 0.0; 
              t_shared_maxIndex[indx] = -1;
              t_shared_maxSize[indx] = 0;
          }
          // active.sync();
        }
      }
      thb_g.sync();

      // if(tile.thread_rank() == 0) 
      if(thb_g.thread_rank() == 0) 
      {
        GraphElem currSize = localCinfo_size[cc - base];
        if(cc >= bound) cc = ocurrComm[ver_loc];
        if((t_shared_maxSize[0] == 1) && 
           (currSize == 1) && 
           (t_shared_maxIndex[0] > cc)) {
          t_shared_maxIndex[0] = cc;
        } 
        clusterWeight[ver_loc] += counter[ver_loc];
        localTarget[ver_loc] = t_shared_maxIndex[0];
      }
      thb_g.sync();
    }  // num_cluster loop
    // tile.sync();
    // __syncwarp();
    thb_g.sync();
  }  // outer loop
}

template<int tile_sz>
__global__
void computeMaxIndex_large(
    GraphElem nv, 
    GraphElem* ocurrComm, GraphElem* currComm, 
    GraphElem* localCinfo_size, GraphWeight* localCinfo_degree,
    GraphElem* localCinfo_oComm, 
    GraphWeight* selfLoop,
    GraphElem* uniq_clus_vec, GraphWeight* counter, 
    GraphElem* clmap_loc,
    GraphElem* clmap_comm, GraphWeight* clmap_weight, 
    GraphWeight* vDegree_vec,
    GraphElem* localTarget,
    GraphWeight* clusterWeight,
    const double constant,
    const GraphElem base, const GraphElem bound
)
{

  __shared__ int shared_num_uniq_cl[S_THREADBLOCK_SIZE];
  __shared__ GraphWeight t_my_counter[S_THREADBLOCK_SIZE];
  __shared__ GraphWeight t_shared_curGain[S_THREADBLOCK_SIZE];
  __shared__ GraphElem t_shared_Index[S_THREADBLOCK_SIZE];
  __shared__ GraphElem t_shared_Size[S_THREADBLOCK_SIZE];
  __shared__ GraphWeight t_shared_maxGain[S_BLOCK_TILE];
  __shared__ GraphElem t_shared_maxIndex[S_BLOCK_TILE];
  __shared__ GraphElem t_shared_maxSize[S_BLOCK_TILE];

  int ii = threadIdx.x;
  GraphElem i = blockIdx.x * blockDim.x + threadIdx.x ;

  shared_num_uniq_cl[ii] = 0;
  if(i < nv) {
    shared_num_uniq_cl[ii] = uniq_clus_vec[i]; 
  }
// if(i == 42609) printf("vertex[%ld]; num_clusters[%ld]\n", i, uniq_clus_vec[i]);

  /// Create cooperative groups
  auto g = cg::this_thread_block();
  // auto tileIdx = g.thread_rank()/tile_sz;
#if __cuda_arch__ >= 700
  auto tile = cg::partition<tile_sz>(cg::this_thread_block());
#else
  auto tile = cg::tiled_partition<tile_sz>(cg::this_thread_block());
#endif

  auto tileIdx = g.thread_rank()/tile.size();
  unsigned ti = tileIdx*tile.size();

  GraphWeight *my_counter = &t_my_counter[tileIdx * tile.size()];
  GraphWeight *shared_curGain = &t_shared_curGain[tileIdx * tile.size()];
  GraphElem *shared_Index = &t_shared_Index[tileIdx * tile.size()];
  GraphElem *shared_Size = &t_shared_Size[tileIdx * tile.size()];
  GraphWeight *shared_maxGain = &t_shared_maxGain[tileIdx];
  GraphElem *shared_maxIndex = &t_shared_maxIndex[tileIdx];
  GraphElem *shared_maxSize = &t_shared_maxSize[tileIdx];

  GraphElem num_cluster = 0;
  GraphWeight ay = 0.0, eiy = 0.0;
  GraphWeight eix;
  GraphElem size;
  GraphWeight curGain = 0.0;
  GraphWeight vDegree, ax;
  GraphElem cc;
  // GraphWeight my_counter;
  GraphWeight currDegree;

  tile.sync();
  for( int wii = 0; wii < tile.size(); wii++) {
    num_cluster = shared_num_uniq_cl[tileIdx*tile.size()+wii];
    if(num_cluster > CUT_SIZE_NUM_EDGES12) {
      if(tile.thread_rank() == 0) shared_maxGain[0] = 0.0;
      tile.sync();
      __syncwarp();
      shared_Index[tile.thread_rank()] = 0;
      GraphElem ver_loc = (GraphElem)(blockIdx.x*blockDim.x+tileIdx*tile.size()+wii);
      cc = currComm[ver_loc]; 
      if(tile.thread_rank() == 0) 
      {
        if(cc >= bound) {
          shared_maxIndex[0] = ocurrComm[ver_loc];
        } else {
          shared_maxIndex[0] = cc;
        }
        shared_maxSize[0] = localCinfo_size[cc - base];
      }
      tile.sync();
      __syncwarp();
      my_counter[tile.thread_rank()] = counter[blockIdx.x*blockDim.x+tileIdx*tile.size()+wii]; 
      eix = my_counter[tile.thread_rank()] - selfLoop[ver_loc];
      vDegree = vDegree_vec[ver_loc];
      currDegree = localCinfo_degree[cc - base];
      ax = currDegree - vDegree;
      for(int k = 0; k < ((num_cluster-1)/tile.size()+1); k++)
      {
        GraphElem thread_indx = k*tile.size() + tile.thread_rank();
        shared_Index[tile.thread_rank()] = -1;
        shared_curGain[tile.thread_rank()] = 0.0;
        if(thread_indx < num_cluster) {
          cg::coalesced_group active = cg::coalesced_threads();
          GraphElem tcomm = clmap_comm[
             clmap_loc[ver_loc]+thread_indx];
          ay = localCinfo_degree[tcomm - base];
          eiy = clmap_weight[
             clmap_loc[ver_loc]+thread_indx];
          curGain = 2.0 * (eiy - eix) - 2.0 * vDegree * (ay - ax) * constant; 
          shared_Size[tile.thread_rank()] = localCinfo_size[tcomm - base];
          if(tcomm >= bound) {
            shared_Index[tile.thread_rank()] = localCinfo_oComm[tcomm - bound];
          } else {
            shared_Index[tile.thread_rank()] = tcomm;
          }

          if((curGain > shared_maxGain[0]) && tcomm != cc ||
             (curGain == shared_maxGain[0] && curGain != 0.0 && tcomm != cc &&
              shared_Index[tile.thread_rank()] < shared_maxIndex[0]) ) {
            shared_curGain[tile.thread_rank()] = curGain;
            shared_Size[tile.thread_rank()] = localCinfo_size[tcomm - base]; 
          } else {
            shared_Index[tile.thread_rank()] = -1;
            shared_curGain[tile.thread_rank()] = 0.0;
            shared_Size[tile.thread_rank()] = 0;
          }
          /// Perform reduction
          active.sync(); 
#pragma unroll
          for (int s =1; s < tile.size(); s *=2) 
          {
            int indx = 2 * s * tile.thread_rank();
            // int indx = tileIdx*tile.size() + 2 * s * tile.thread_rank();
    
            if(indx < tile.size()) {
              if(shared_Index[indx+s] != -1) {
                if((shared_curGain[indx+s] > shared_curGain[indx]) ||
                    (shared_Index[indx] == -1) ||
                   (shared_curGain[indx+s] == shared_curGain[indx] &&
                   shared_curGain[indx+s] != 0.0 && shared_Index[indx] != -1 &&
                   shared_Index[indx+s] < shared_Index[indx])
                  ) {
                  shared_curGain[indx] = shared_curGain[indx+s];
                  shared_Index[indx] = shared_Index[indx+s];
                  shared_Size[indx] = shared_Size[indx+s];
                }
              } else if(shared_Index[indx+s] == -1 && shared_Index[indx] == -1) {
                  shared_curGain[indx] = 0.0; 
                  shared_Index[indx] = -1;
                  shared_Size[indx] = 0;
              }
            }
            active.sync();
          }
          if(tile.thread_rank() == 0) {
            if(shared_curGain[0] > shared_maxGain[0] ||
               shared_curGain[0] == shared_maxGain[0] &&
               shared_curGain[0] != 0.0 &&
               shared_Index[0] < shared_maxIndex[0]
              ) {
              shared_maxGain[0] = shared_curGain[0];
              shared_maxIndex[0] = shared_Index[0];
              shared_maxSize[0] = shared_Size[0];
            }
          }
        active.sync();
        }
        tile.sync();
        __syncwarp();
      }
      if(tile.thread_rank() == 0) {
        GraphElem currSize = localCinfo_size[cc - base];
        if(cc >= bound) cc = ocurrComm[ver_loc];
        if((shared_maxSize[0] == 1) && 
           (currSize == 1) && 
           (shared_maxIndex[0] > cc)) {
          shared_maxIndex[0] = cc;
        } 
        clusterWeight[ver_loc] += counter[ver_loc];
        localTarget[ver_loc] = shared_maxIndex[0];
      }
      tile.sync();
      __syncwarp();
    }
    tile.sync();
    __syncwarp();
  }
}

template<int tile_sz>
__global__
void computeMaxIndex(
    GraphElem nv, 
    GraphElem* ocurrComm, GraphElem* currComm, 
    GraphElem* localCinfo_size, GraphWeight* localCinfo_degree,
    GraphElem* localCinfo_oComm, 
    GraphWeight* selfLoop,
    GraphElem* uniq_clus_vec, GraphWeight* counter, 
    GraphElem* clmap_loc,
    GraphElem* clmap_comm, GraphWeight* clmap_weight, 
    GraphWeight* vDegree_vec,
    GraphElem* localTarget,
    GraphWeight* clusterWeight,
    const double constant,
    const GraphElem base, const GraphElem bound
)
{

  __shared__ int shared_num_uniq_cl[S_THREADBLOCK_SIZE];
  __shared__ GraphWeight t_my_counter[S_THREADBLOCK_SIZE];
  __shared__ GraphWeight t_shared_curGain[S_THREADBLOCK_SIZE];
  __shared__ GraphElem t_shared_Index[S_THREADBLOCK_SIZE];
  __shared__ GraphElem t_shared_Size[S_THREADBLOCK_SIZE];
  __shared__ GraphWeight t_shared_maxGain[S_BLOCK_TILE];
  __shared__ GraphElem t_shared_maxIndex[S_BLOCK_TILE];
  __shared__ GraphElem t_shared_maxSize[S_BLOCK_TILE];

  int ii = threadIdx.x;
  GraphElem i = blockIdx.x * blockDim.x + threadIdx.x ;

  shared_num_uniq_cl[ii] = 0;
  if(i < nv) {
    shared_num_uniq_cl[ii] = uniq_clus_vec[i]; 
  }

  /// Create cooperative groups
  auto g = cg::this_thread_block();
  // auto tileIdx = g.thread_rank()/tile_sz;
#if __cuda_arch__ >= 700
  auto tile = cg::partition<tile_sz>(cg::this_thread_block());
#else
  auto tile = cg::tiled_partition<tile_sz>(cg::this_thread_block());
#endif

  auto tileIdx = g.thread_rank()/tile.size();
  unsigned ti = tileIdx*tile.size();

  GraphWeight *my_counter = &t_my_counter[tileIdx * tile.size()];
  GraphWeight *shared_curGain = &t_shared_curGain[tileIdx * tile.size()];
  GraphElem *shared_Index = &t_shared_Index[tileIdx * tile.size()];
  GraphElem *shared_Size = &t_shared_Size[tileIdx * tile.size()];
  GraphWeight *shared_maxGain = &t_shared_maxGain[tileIdx];
  GraphElem *shared_maxIndex = &t_shared_maxIndex[tileIdx];
  GraphElem *shared_maxSize = &t_shared_maxSize[tileIdx];

  GraphElem num_cluster = 0;
  GraphWeight ay = 0.0, eiy = 0.0;
  GraphWeight eix;
  GraphElem size;
  GraphWeight curGain = 0.0;
  GraphWeight vDegree, ax;
  GraphElem cc;
  // GraphWeight my_counter;
  GraphWeight currDegree;

  tile.sync();
  for( int wii = 0; wii < tile.size(); wii++) {
    num_cluster = shared_num_uniq_cl[tileIdx*tile.size()+wii];
    if(num_cluster >= (GraphElem)tile_sz && num_cluster <= CUT_SIZE_NUM_EDGES12) {
      if(tile.thread_rank() == 0) shared_maxGain[0] = 0.0;
      tile.sync();
      __syncwarp();
      shared_Index[tile.thread_rank()] = 0;
      GraphElem ver_loc = (GraphElem)(blockIdx.x*blockDim.x+tileIdx*tile.size()+wii);
      cc = currComm[ver_loc]; 
      if(tile.thread_rank() == 0) 
      {
        if(cc >= bound) {
          shared_maxIndex[0] = ocurrComm[ver_loc];
        } else {
          shared_maxIndex[0] = cc;
        }
        shared_maxSize[0] = localCinfo_size[cc - base];
      }
      tile.sync();
      __syncwarp();
      my_counter[tile.thread_rank()] = counter[blockIdx.x*blockDim.x+tileIdx*tile.size()+wii]; 
      eix = my_counter[tile.thread_rank()] - selfLoop[ver_loc];
      vDegree = vDegree_vec[ver_loc];
      currDegree = localCinfo_degree[cc - base];
      ax = currDegree - vDegree;
      for(int k = 0; k < ((num_cluster-1)/tile.size()+1); k++)
      {
        GraphElem thread_indx = k*tile.size() + tile.thread_rank();
        shared_Index[tile.thread_rank()] = -1;
        shared_curGain[tile.thread_rank()] = 0.0;
        if(thread_indx < num_cluster) {
          cg::coalesced_group active = cg::coalesced_threads();
          GraphElem tcomm = clmap_comm[
             clmap_loc[ver_loc]+thread_indx];
          ay = localCinfo_degree[tcomm - base];
          eiy = clmap_weight[
             clmap_loc[ver_loc]+thread_indx];
          curGain = 2.0 * (eiy - eix) - 2.0 * vDegree * (ay - ax) * constant; 
          shared_Size[tile.thread_rank()] = localCinfo_size[tcomm - base];
          if(tcomm >= bound) {
            shared_Index[tile.thread_rank()] = localCinfo_oComm[tcomm - bound];
          } else {
            shared_Index[tile.thread_rank()] = tcomm;
          }

          if((curGain > shared_maxGain[0]) && tcomm != cc ||
             (curGain == shared_maxGain[0] && curGain != 0.0 && tcomm != cc &&
              shared_Index[tile.thread_rank()] < shared_maxIndex[0]) ) {
            shared_curGain[tile.thread_rank()] = curGain;
            shared_Size[tile.thread_rank()] = localCinfo_size[tcomm - base]; 
          } else {
            shared_Index[tile.thread_rank()] = -1;
            shared_curGain[tile.thread_rank()] = 0.0;
            shared_Size[tile.thread_rank()] = 0;
          }
          /// Perform reduction
          active.sync(); 
#pragma unroll
          for (int s =1; s < tile.size(); s *=2) 
          {
            int indx = 2 * s * tile.thread_rank();
            // int indx = tileIdx*tile.size() + 2 * s * tile.thread_rank();
    
            if(indx < tile.size()) {
              if(shared_Index[indx+s] != -1) {
                if((shared_curGain[indx+s] > shared_curGain[indx]) ||
                    (shared_Index[indx] == -1) ||
                   (shared_curGain[indx+s] == shared_curGain[indx] &&
                   shared_curGain[indx+s] != 0.0 && shared_Index[indx] != -1 &&
                   shared_Index[indx+s] < shared_Index[indx])
                  ) {
                  shared_curGain[indx] = shared_curGain[indx+s];
                  shared_Index[indx] = shared_Index[indx+s];
                  shared_Size[indx] = shared_Size[indx+s];
                }
              } else if(shared_Index[indx+s] == -1 && shared_Index[indx] == -1) {
                  shared_curGain[indx] = 0.0; 
                  shared_Index[indx] = -1;
                  shared_Size[indx] = 0;
              }
            }
            active.sync();
          }
          if(tile.thread_rank() == 0) {
            if(shared_curGain[0] > shared_maxGain[0] ||
               shared_curGain[0] == shared_maxGain[0] &&
               shared_curGain[0] != 0.0 &&
               shared_Index[0] < shared_maxIndex[0]
              ) {
              shared_maxGain[0] = shared_curGain[0];
              shared_maxIndex[0] = shared_Index[0];
              shared_maxSize[0] = shared_Size[0];
            }
          }
        active.sync();
        }
        tile.sync();
        __syncwarp();
      }
      if(tile.thread_rank() == 0) {
        GraphElem currSize = localCinfo_size[cc - base];
        if(cc >= bound) cc = ocurrComm[ver_loc];
        if((shared_maxSize[0] == 1) && 
           (currSize == 1) && 
           (shared_maxIndex[0] > cc)) {
          shared_maxIndex[0] = cc;
        } 
        clusterWeight[ver_loc] += counter[ver_loc];
        localTarget[ver_loc] = shared_maxIndex[0];
      }
      tile.sync();
      __syncwarp();
    }
    tile.sync();
    __syncwarp();
  }
  ///  Now implement for vertices with num_clusters < 4
  // if(i >= nv) return;
  curGain = 0.0; 
  GraphWeight maxGain = 0.0; 
  num_cluster = shared_num_uniq_cl[ii];
  if(num_cluster < (GraphElem)tile_sz && num_cluster > 0 && i < nv) {
  cc = currComm[i];
  GraphElem maxIndex;
  if(cc >= bound) {
    maxIndex = ocurrComm[i]; 
  } else {
    maxIndex = cc;
  }
  localTarget[i] = -1; // cc;
  GraphElem currSize = localCinfo_size[cc - base];
  currDegree = localCinfo_degree[cc - base];
  GraphElem maxSize = currSize;

    t_my_counter[ii] = counter[i];
    eix = t_my_counter[ii] - selfLoop[i];
    vDegree = vDegree_vec[i]; 
    ax = currDegree - vDegree;
    GraphElem tcomm, otcomm; 
    for(GraphElem k = 0; k < num_cluster; k++) {
      tcomm = clmap_comm[clmap_loc[i]+k];
      if (tcomm != cc) {
        ay = localCinfo_degree[tcomm - base];
        size = localCinfo_size[tcomm - base];
        eiy = clmap_weight[clmap_loc[i]+k];
        curGain = 2.0 * (eiy - eix) - 2.0 * vDegree * (ay - ax) * constant; 
        if(tcomm >= bound) {
          otcomm = localCinfo_oComm[tcomm - bound];
        } else {
          otcomm = tcomm;
        }
        if((curGain > maxGain) || 
           (curGain == maxGain && curGain != 0.0 && otcomm < maxIndex) ) {
          maxGain = curGain;
          maxIndex = otcomm;
          maxSize = size;
        }
      }
    }
    if(cc >= bound) cc = ocurrComm[i]; 
    if((maxSize == 1) && (currSize == 1) && (maxIndex > cc)) {
      maxIndex = cc;
    } 
    clusterWeight[i] += counter[i]; 
    localTarget[i] = maxIndex;
  } else if(num_cluster == 0 && i < nv) {
    localTarget[i] = ocurrComm[i];
  }
}

template<int tile_sz>
__global__ 
void distGetMaxIndex_large_new(
    const int me, const int numIters,
    const GraphElem nv, GraphElem nv_chunk_size, 
    const GraphElem size_lt_cs2, GraphElem* list_lt_cs2,
    GraphElem max_comm_size,
    GraphElem* unique_comm_array_g, 
    GraphWeight* unique_weight_array_g, 
    GraphElem* e0, // GraphElem* e1, 
    GraphElem* graph_edgeList_tail, GraphWeight* graph_edgeList_weight,
    GraphElem* currComm,
    GraphElem* clmap_loc,
    GraphElem* clmap_comm, GraphWeight* clmap_weight, 
    GraphElem* List_numEdges,
    GraphElem* uniq_clus_vec, GraphWeight* counter, 
    const GraphElem base // , const GraphElem bound
    ) {
  // __shared__ GraphElem t_shared_begin_loc[FINDING_UNIQCOMM_BLOCK_TILE];
  // __shared__ GraphElem t_shared_comm[FINDING_UNIQCOMM_BLOCK_SIZE];
  // __shared__ GraphWeight t_shared_weight[FINDING_UNIQCOMM_BLOCK_SIZE];

  /// Create cooperative groups
  auto thb_g = cg::this_thread_block();
  auto tileIdx = thb_g.thread_rank()/tile_sz;
  // GraphElem *shared_begin_loc = &t_shared_begin_loc[tileIdx];
  // GraphElem *shared_comm = &t_shared_comm[tileIdx];
  // GraphWeight* shared_weight = &t_shared_weight[tileIdx];
  auto tile = cg::tiled_partition<tile_sz>(cg::this_thread_block());

  int ii = threadIdx.x;
  GraphElem* unique_comm_array = 
             &unique_comm_array_g[blockIdx.x*FINDING_UNIQCOMM_ARRAY_SIZE];
  GraphWeight* unique_weight_array = 
             &unique_weight_array_g[blockIdx.x*FINDING_UNIQCOMM_ARRAY_SIZE];

  for(GraphElem i_nv = 0; i_nv < nv_chunk_size; ++i_nv) {
    GraphElem i_index = FINDING_UNIQCOMM_NUM_BLOCKS * i_nv + blockIdx.x;
    if(i_index < size_lt_cs2) {
    GraphElem i = list_lt_cs2[i_index];
    if(i < nv) {
    GraphElem cc = currComm[i];

    GraphElem num_edges = List_numEdges[i];
    if(num_edges > CUT_SIZE_NUM_EDGES1) {
      GraphElem clmap_loc_i = clmap_loc[i];
// if(threadIdx.x == 0)
//     printf("me[%d]; blockIdx.x[%d]; i[%ld]; base[%ld]\n", me, blockIdx.x, i, base);
// 
      for(GraphElem k = 0; k < ((max_comm_size*FINDING_UNIQCOMM_FACTOR-1)/thb_g.size()+1); k++) {
        // GraphElem thread_indx = k*tile.size() + tile.thread_rank();
        GraphElem thread_indx = k*thb_g.size() + thb_g.thread_rank();
        if(thread_indx < max_comm_size*FINDING_UNIQCOMM_FACTOR) {
           unique_comm_array[thread_indx] = -1;
           unique_weight_array[thread_indx] = 0.0;
        }
      }
      thb_g.sync();

// if(thb_g.thread_rank() == 0)
//     printf("me[%d]; i[%ld]; num_edges[%ld]\n", me, i, num_edges);
// 
      for(GraphElem k = 0; k < ((num_edges-1)/thb_g.size()+1); k++) {
        // GraphElem thread_indx = k*tile.size() + tile.thread_rank();
        GraphElem thread_indx = k*thb_g.size() + thb_g.thread_rank();
        if(thread_indx < num_edges) {
          GraphElem th_tail_indx = e0[i]+thread_indx;
          GraphElem tail = graph_edgeList_tail[th_tail_indx];
          // unique_comm_array[currComm[tail - base]] = 1;
// if(i == 169230)
//     printf("me[%d]; thread_indx[%ld]; currComm[%ld]\n", me, thread_indx, currComm[tail - base]);
#if __cuda_arch__ >= 600
          atomicAdd(&unique_comm_array[currComm[tail - base]-base], 1);
          atomicAdd(&unique_weight_array[currComm[tail - base]-base], 
             graph_edgeList_weight[th_tail_indx]);
#else
#ifndef USE_32_BIT_GRAPH
          my_func_atomicAdd(&unique_comm_array[currComm[tail - base]-base], 1);
          my_func_atomicAdd(&unique_weight_array[currComm[tail - base]-base],
             graph_edgeList_weight[th_tail_indx]);
#else
          atomicAdd(&unique_comm_array[currComm[tail - base]-base], 1);
          atomicAdd(&unique_weight_array[currComm[tail - base]-base],
             graph_edgeList_weight[th_tail_indx]);
#endif
#endif
  // printf("new vertex[%ld]; comm_array[%ld]; weight_array[%e] \n", 
  //        i, currComm[tail - base],
  //        unique_weight_array[currComm[tail - base]]);

        }
      }
      thb_g.sync();

      // Make unique cluster vectors of comm and weights
      for(GraphElem k = 0; k < ((max_comm_size*FINDING_UNIQCOMM_FACTOR-1)/thb_g.size()+1); k++) {
        GraphElem thread_indx = k*thb_g.size() + thb_g.thread_rank();
        if(thread_indx < max_comm_size*FINDING_UNIQCOMM_FACTOR) {
           if(unique_comm_array[thread_indx] != -1) {
             cg::coalesced_group active = cg::coalesced_threads();
             GraphElem index_loc;
             if(active.thread_rank() == 0) {
#if __cuda_arch__ >= 600
               index_loc = atomicAdd(&uniq_clus_vec[i], active.size()); 
#else
#ifndef USE_32_BIT_GRAPH
               index_loc = my_func_atomicAdd(&uniq_clus_vec[i], active.size()); 
#else
               index_loc = atomicAdd(&uniq_clus_vec[i], active.size()); 
#endif
#endif
     // printf("vertex using distGetMaxIndex_large_new[%ld]; num_edges[%ld]; active.size[%d]\n"
     //    , i, num_edges, active.size());
             }
             active.sync();
  // printf("new vertex[%ld]; thread_index[%ld]; shared_begin_loc[0][%ld] \n",
  //        i, thread_indx, shared_begin_loc[0]);
             if (cc == thread_indx+base) {
               counter[i] = unique_weight_array[thread_indx];
  // printf("vertex using distGetMaxIndex_large_new[%ld]; counter[%e]\n", i, counter[i]);
             }
             clmap_comm[clmap_loc_i+active.shfl(index_loc, 0)+active.thread_rank()] =
                thread_indx+base;
             clmap_weight[clmap_loc_i+active.shfl(index_loc, 0)+active.thread_rank()] =
                unique_weight_array[thread_indx];
             // clmap_comm[clmap_loc_i+my_loc] = thread_indx;
             // clmap_weight[clmap_loc_i+my_loc] = unique_weight_array[thread_indx];
  // printf("new vertex[%ld]; clmap_comm[%ld]; clmap_weight[%e] \n", 
  //        i, clmap_comm[shared_begin_loc[0]-active.size()+active.thread_rank()],
  //        clmap_weight[shared_begin_loc[0]-active.size()+active.thread_rank()]);
           }
        }
        thb_g.sync();
      }
      thb_g.sync();
// if(thb_g.thread_rank() == 0) 
//     printf("me[%d]; vertex[%ld]; cc[%ld]; uniq_cls_vec_size[%ld]\n", 
//        me, i, cc, uniq_clus_vec[i]);
    }  // (num_edges > CUT_SIZE_NUM_EDGES1) loop


  }  // if(i >= nv) 
  }  // if(i_index >= size_lt_cs2)
  }  // chunk size loop
}

template<int tile_sz>
__global__ 
void distGetMaxIndex_large(
    GraphElem nv, 
    GraphElem* e0, GraphElem* e1, 
    GraphElem* graph_edgeList_tail, GraphWeight* graph_edgeList_weight,
    GraphElem* currComm,
    GraphElem* clmap_loc,
    GraphElem* clmap_comm, GraphWeight* clmap_weight, 
    GraphElem* List_numEdges,
    GraphElem* uniq_clus_vec, GraphWeight* counter, 
    const GraphElem base, const GraphElem bound
    ) {

  int ii = threadIdx.x;
  GraphElem i = blockIdx.x * blockDim.x + threadIdx.x ;

  __shared__ GraphElem t_shared_num_edges[S_THREADBLOCK_SIZE];
  __shared__ GraphWeight t_shared_weight[S_THREADBLOCK_SIZE];
  __shared__ GraphElem t_shared_clmap_loc[S_THREADBLOCK_SIZE];
  __shared__ GraphWeight t_red_shared_weight[S_BLOCK_TILE];

  t_shared_num_edges[ii] = 0; 
  // if(i >= nv) return;

  if(i < nv) {
    t_shared_num_edges[ii] = List_numEdges[i];
    t_shared_clmap_loc[ii] = clmap_loc[i];
  }

  /// Create cooperative groups
  auto thb_g = cg::this_thread_block();
  auto tileIdx = thb_g.thread_rank()/tile_sz;
  GraphElem* shared_num_edges = &t_shared_num_edges[tileIdx * tile_sz];
  GraphWeight* shared_weight= &t_shared_weight[tileIdx * tile_sz];
  GraphElem* shared_clmap_loc = &t_shared_clmap_loc[tileIdx * tile_sz];
  GraphWeight *shared_red_weight = &t_red_shared_weight[tileIdx];
  auto tile = cg::tiled_partition<tile_sz>(cg::this_thread_block());
  // tile.sync();
  thb_g.sync();

  // GraphElem tcomm;
  GraphElem Tail;
  GraphElem num_edges;
  /// Cater to only vertices with num of edges >= tile_sz
  /// each thread work on each edge
  // for( int wii = 0; wii < tile.size(); wii++) 
  for( int wii = 0; wii < thb_g.size(); wii++) {
    num_edges = t_shared_num_edges[wii];
    if(num_edges > CUT_SIZE_NUM_EDGES1) {
      // for(int k = 0; k < ((num_edges-1)/tile.size()+1); k++)
      for(int k = 0; k < ((num_edges-1)/thb_g.size()+1); k++) {
        // GraphElem thread_indx = k*tile.size() + tile.thread_rank();
        GraphElem thread_indx = k*thb_g.size() + thb_g.thread_rank();
        if(thread_indx < num_edges) {
          GraphElem th_tail_indx = 
             e0[blockIdx.x*blockDim.x+wii]+thread_indx;
          Tail = graph_edgeList_tail[th_tail_indx];
          clmap_comm[t_shared_clmap_loc[wii]+thread_indx] =
             currComm[Tail - base];
          clmap_weight[t_shared_clmap_loc[wii]+thread_indx] =
             graph_edgeList_weight[th_tail_indx];
        }
      }
    }
  }
  thb_g.sync();

  /// Now find out unique clusters and accumulate weights 
  GraphElem cc; 
  // for( int wii = 0; wii < tile_sz; wii++)
  for( int wii = 0; wii < thb_g.size(); wii++) {
    num_edges = t_shared_num_edges[wii];
    // if (blockIdx.x*blockDim.x+tileIdx*tile_sz+wii < nv)
    if (blockIdx.x*blockDim.x+wii < nv)
      // cc = currComm[blockIdx.x*blockDim.x+tileIdx*tile_sz+wii]; 
      cc = currComm[blockIdx.x*blockDim.x+wii]; 
    GraphWeight tile_sum_Weight;
    if(num_edges > CUT_SIZE_NUM_EDGES1) {
      GraphElem store_indx = -1;
      // if (tile.thread_rank() == 0)
          // uniq_clus_vec[blockIdx.x*blockDim.x+tileIdx*tile_sz+wii] = 0;
      if (thb_g.thread_rank() == 0)
          uniq_clus_vec[blockIdx.x*blockDim.x+wii] = 0;
      for(GraphElem ko = 0; ko < num_edges; ko++) {
        tile_sum_Weight = 0.0;
        GraphElem comm_pos = clmap_comm[t_shared_clmap_loc[wii]+ko];
        if(comm_pos != -1) {
          // if (tile.thread_rank() == 0)
          if (thb_g.thread_rank() == 0) {
            store_indx += 1;
            clmap_comm[t_shared_clmap_loc[wii]+store_indx] = comm_pos;
          }
          // if (tile.thread_rank() == 0)
          if (thb_g.thread_rank() == 0)
            clmap_weight[t_shared_clmap_loc[wii]+store_indx] =
                clmap_weight[t_shared_clmap_loc[wii]+ko]; 
          // if (tile.thread_rank() == 0)
          if (thb_g.thread_rank() == 0)
               uniq_clus_vec[blockIdx.x*blockDim.x+wii] += 1;
          // if (tile.thread_rank() == 0)
          if (thb_g.thread_rank() == 0)
               shared_red_weight[0] = 0.0; 
          shared_weight[thb_g.thread_rank()] = 0.0;
          // for(GraphElem k = 0; k < ((num_edges-1)/tile.size()+1); k++)
          for(GraphElem k = 0; k < ((num_edges-ko-1)/thb_g.size()+1); k++) {
            GraphElem thread_indx = ko + 1 + k*thb_g.size() + thb_g.thread_rank();
            if(thread_indx < num_edges) {
              if (comm_pos == clmap_comm[t_shared_clmap_loc[wii]+thread_indx]) {
                 shared_weight[thb_g.thread_rank()] +=
                    clmap_weight[t_shared_clmap_loc[wii]+thread_indx];
                 clmap_comm[t_shared_clmap_loc[wii]+thread_indx] = -1;
              }
            }
          }
            // tile.sync();
            thb_g.sync();
            /// Perform reduction to accumulate weights
            tile_sum_Weight = weight_reduce(thb_g, t_shared_weight, 
               t_shared_weight[thb_g.thread_rank()]);
               // shared_weight[tile.thread_rank()]);
            if(thb_g.thread_rank() ==0) shared_red_weight[0] += tile_sum_Weight; 
          // thb_g.sync();
          // /// Perform reduction at threadblock level
          // for (int s =1; s < S_BLOCK_TILE; s *=2) 
          // {
          //   int indx = 2 * s * thb_g.thread_rank();

          //   if(indx < S_BLOCK_TILE) {
          //     shared_red_weight[indx] = shared_red_weight[indx+s];
          //   }
          // }
          // thb_g.sync();

          /// Add weights to cluster map
          // if (tile.thread_rank() == 0)
          if (thb_g.thread_rank() == 0) {
            // clmap_weight[t_shared_clmap_loc[wii]+store_indx] += tile_sum_Weight;
            clmap_weight[t_shared_clmap_loc[wii]+store_indx] += 
               shared_red_weight[0];
    // printf("vertex[%ld]; weight[%ld][%e]; ko[%ld] comm_pos[%ld]\n", 
    // blockIdx.x*blockDim.x+wii, t_shared_clmap_loc[wii]+store_indx, 
    // clmap_weight[t_shared_clmap_loc[wii]+store_indx], ko, comm_pos);
            // if(comm_pos == cc) counter[blockIdx.x*blockDim.x+tileIdx*tile_sz+wii] =
            //    clmap_weight[shared_clmap_loc[wii]+store_indx];
            if(comm_pos == cc) counter[blockIdx.x*blockDim.x+wii] =
               clmap_weight[t_shared_clmap_loc[wii]+store_indx];
          }

        }
        // tile.sync();
        thb_g.sync();
      }
      
    }
  }  // end of old implementation




}





template<int tile_sz>
__global__ 
void distGetMaxIndex(
    GraphElem nv, 
    GraphElem* e0, // GraphElem* e1, 
    GraphElem* graph_edgeList_tail, GraphWeight* graph_edgeList_weight,
    GraphElem* currComm,
    GraphElem* clmap_loc,
    GraphElem* clmap_comm, GraphWeight* clmap_weight, 
    GraphElem* List_numEdges,
    GraphElem* uniq_clus_vec, GraphWeight* counter, 
    const GraphElem base // , const GraphElem bound
    ) {

  int ii = threadIdx.x;
  GraphElem i = blockIdx.x * blockDim.x + threadIdx.x ;

  __shared__ GraphElem t_shared_num_edges[S_THREADBLOCK_SIZE];
  __shared__ GraphWeight t_shared_weight[S_THREADBLOCK_SIZE];
  __shared__ GraphElem t_shared_clmap_loc[S_THREADBLOCK_SIZE];

  t_shared_num_edges[ii] = 0; 
  // if(i >= nv) return;

  if(i < nv) {
    t_shared_num_edges[ii] = List_numEdges[i];
    t_shared_clmap_loc[ii] = clmap_loc[i];
  }

  /// Create cooperative groups
  auto g = cg::this_thread_block();
  auto tileIdx = g.thread_rank()/tile_sz;
  GraphElem* shared_num_edges = &t_shared_num_edges[tileIdx * tile_sz];
  GraphWeight* shared_weight= &t_shared_weight[tileIdx * tile_sz];
  GraphElem* shared_clmap_loc = &t_shared_clmap_loc[tileIdx * tile_sz];
  auto tile = cg::tiled_partition<tile_sz>(cg::this_thread_block());
  tile.sync();

  // GraphElem tcomm;
  GraphElem Tail;
  GraphElem num_edges;
  /// Cater to only vertices with num of edges >= tile_sz
  /// each thread work on each edge
  for( int wii = 0; wii < tile_sz; wii++) {
    num_edges = shared_num_edges[wii];
    if(num_edges >= (GraphElem)tile_sz && num_edges <= CUT_SIZE_NUM_EDGES1) {
      for(int k = 0; k < ((num_edges-1)/tile.size()+1); k++) {
        GraphElem thread_indx = k*tile.size() + tile.thread_rank();
        if(thread_indx < num_edges) {
          GraphElem th_tail_indx = 
             e0[blockIdx.x*blockDim.x+tileIdx*tile_sz+wii]+thread_indx;
          Tail = graph_edgeList_tail[th_tail_indx];
          clmap_comm[shared_clmap_loc[wii]+thread_indx] =
             currComm[Tail - base];
          clmap_weight[shared_clmap_loc[wii]+thread_indx] =
             graph_edgeList_weight[th_tail_indx];
        }
      }
    }
  }
  tile.sync();
  /// Cater to only vertices with num of edges < 16
  /// each thread work on each vertex
  num_edges = shared_num_edges[tile.thread_rank()];
  if(num_edges < (GraphElem)tile_sz && num_edges > 0 && i < nv) { 
    GraphElem edge_low = e0[i];
    for (GraphElem j = 0; j < num_edges; j++) {
      Tail = graph_edgeList_tail[edge_low+j];
        clmap_comm[shared_clmap_loc[tile.thread_rank()]+j] = currComm[Tail - base];
        clmap_weight[shared_clmap_loc[tile.thread_rank()]+j] = graph_edgeList_weight[edge_low+j];
    }
  }
  /// Now find out unique clusters and accumulate weights 
  GraphElem cc; 
  for( int wii = 0; wii < tile_sz; wii++) {
    num_edges = shared_num_edges[wii];
    if (blockIdx.x*blockDim.x+tileIdx*tile_sz+wii < nv)
      cc = currComm[blockIdx.x*blockDim.x+tileIdx*tile_sz+wii]; 
    GraphWeight tile_sum_Weight;
    if(num_edges >= (GraphElem)tile_sz && num_edges <= CUT_SIZE_NUM_EDGES1) {
      GraphElem store_indx = -1;
      if (tile.thread_rank() == 0)
          uniq_clus_vec[blockIdx.x*blockDim.x+tileIdx*tile_sz+wii] = 0;
      for(GraphElem ko = 0; ko < num_edges; ko++) {
        tile_sum_Weight = 0.0;
        GraphElem comm_pos = clmap_comm[shared_clmap_loc[wii]+ko];
        if(comm_pos != -1) {
          if (tile.thread_rank() == 0) {
            store_indx += 1;
            clmap_comm[shared_clmap_loc[wii]+store_indx] = comm_pos;
          }
          if (tile.thread_rank() == 0)
            clmap_weight[shared_clmap_loc[wii]+store_indx] =
                clmap_weight[shared_clmap_loc[wii]+ko]; 
          if (tile.thread_rank() == 0)
               uniq_clus_vec[blockIdx.x*blockDim.x+tileIdx*tile_sz+wii] += 1;
          for(GraphElem k = 0; k < ((num_edges-ko-1)/tile.size()+1); k++) {
            GraphElem thread_indx = ko + 1 + k*tile.size() + tile.thread_rank();
            shared_weight[tile.thread_rank()] = 0.0;
            if(thread_indx < num_edges) {
              if (comm_pos == clmap_comm[shared_clmap_loc[wii]+thread_indx]) {
                 shared_weight[tile.thread_rank()] =
                    clmap_weight[shared_clmap_loc[wii]+thread_indx];
                 clmap_comm[shared_clmap_loc[wii]+thread_indx] = -1;
              }
            }
            tile.sync();
            /// Perform reduction to accumulate weights
           tile_sum_Weight += weight_reduce_sum_tile_shfl<tile_sz>
               (tile, shared_weight[tile.thread_rank()]);
          }
          /// Add weights to cluster map
          if (tile.thread_rank() == 0) {
            clmap_weight[shared_clmap_loc[wii]+store_indx] += tile_sum_Weight;
            if(comm_pos == cc) counter[blockIdx.x*blockDim.x+tileIdx*tile_sz+wii] =
               clmap_weight[shared_clmap_loc[wii]+store_indx];
          }

        }
        tile.sync();
      }
      
    }
  } 

  /// repeate for num_edges < 16
  // if( i < nv) return; 
  num_edges = shared_num_edges[tile.thread_rank()];
  if(num_edges < (GraphElem)tile_sz && num_edges > 0 && i < nv) { 
    cc = currComm[i];
    uniq_clus_vec[i] = 0;
    GraphElem store_indx = -1;
    int counter_switch = 1;
    for(GraphElem ko = 0; ko < num_edges; ko++) {
      GraphElem comm_pos = clmap_comm[shared_clmap_loc[tile.thread_rank()]+ko];
      // GraphElem comm_count = 1;
      if(comm_pos != -1) {
        uniq_clus_vec[i] += 1;
        store_indx += 1;
        clmap_comm[shared_clmap_loc[tile.thread_rank()]+store_indx] = comm_pos;
        clmap_weight[shared_clmap_loc[tile.thread_rank()]+store_indx] = 
            clmap_weight[shared_clmap_loc[tile.thread_rank()]+ko]; 
        for(GraphElem k = ko+1; k < num_edges; k++) {
          if (comm_pos == clmap_comm[shared_clmap_loc[tile.thread_rank()]+k]) {
             clmap_comm[shared_clmap_loc[tile.thread_rank()]+k] = -1;
             clmap_weight[shared_clmap_loc[tile.thread_rank()]+store_indx] += 
                clmap_weight[shared_clmap_loc[tile.thread_rank()]+k]; 
          }
        }
        if (comm_pos == cc && counter_switch == 1) {
          counter_switch = 0;
          counter[i] += clmap_weight[shared_clmap_loc[tile.thread_rank()]+store_indx]; 
        }
      }
    }
  }
}

template<int tile_sz>
__global__ 
void distBuildLocalMapCounter(
    /// Accumulate selfLoopVec for all vertices
    GraphElem nv,
    GraphElem* e0, GraphElem* e1,
    GraphElem* graph_edgeList_tail, GraphWeight* graph_edgeList_weight,
    GraphElem* List_numEdges, GraphWeight* selfLoopVec,
    const GraphElem base // , const GraphElem bound
    ) {

  int ii = threadIdx.x;
  GraphElem i = blockIdx.x * blockDim.x + threadIdx.x ;

  __shared__ GraphElem t_shared_num_edges[S_THREADBLOCK_SIZE];
  __shared__ GraphWeight t_shared_block_weight[S_THREADBLOCK_SIZE];
  t_shared_num_edges[ii] = 0; 
  // if(i >= nv) return;

  if(i < nv) {
    t_shared_num_edges[ii] = List_numEdges[i];
  }
#if __cuda_arch__ >= 700
  auto tile = cg::partition<tile_sz>(cg::this_thread_block());
#else
  auto tile = cg::tiled_partition<tile_sz>(cg::this_thread_block());
#endif
  auto g = cg::this_thread_block();
  auto tileIdx = g.thread_rank()/tile_sz;
  GraphElem *shared_num_edges = &t_shared_num_edges[tileIdx * tile_sz];
  GraphWeight *shared_block_weight = &t_shared_block_weight[tileIdx * tile_sz];
  tile.sync();
  /// Cater to only vertices with num of edges >= tile_sz
  /// and work on each vertex
  /// this implementation uses cooperative groups and 
  /// uses a large thread block size say 128 to increase occupancy
  for( int wii = 0; wii < tile_sz; wii++) {
    GraphElem num_edges = shared_num_edges[wii];
    shared_block_weight[tile.thread_rank()] = 0.0;
    if(num_edges >= (GraphElem)tile_sz) {
      for(int k = 0; k < ((num_edges-1)/tile.size()+1); k++) {
        GraphElem thread_indx = k*tile.size() + tile.thread_rank();
        if(thread_indx < num_edges) {
          if(graph_edgeList_tail[
            e0[blockIdx.x*blockDim.x+tileIdx*tile_sz+wii]+thread_indx] ==
            blockIdx.x * blockDim.x + tileIdx*tile_sz+wii + base)
            shared_block_weight[tile.thread_rank()] +=
               graph_edgeList_weight[
               e0[blockIdx.x*blockDim.x+tileIdx*tile_sz+wii]+thread_indx];
        }
      }
      tile.sync();
      GraphWeight tile_sum_Weight = weight_reduce_sum_tile_shfl<tile_sz>
         (tile, shared_block_weight[tile.thread_rank()]);
      if (tile.thread_rank() == 0) 
#if __cuda_arch__ >= 600
         atomicAdd(&selfLoopVec[blockIdx.x*blockDim.x+tileIdx*tile_sz+wii], tile_sum_Weight);
#else
#ifndef USE_32_BIT_GRAPH
         my_func_atomicAdd(&selfLoopVec[blockIdx.x*blockDim.x+tileIdx*tile_sz+wii], tile_sum_Weight);
#else
         atomicAdd(&selfLoopVec[blockIdx.x*blockDim.x+tileIdx*tile_sz+wii], tile_sum_Weight);
#endif
#endif
    }
  }
  /// Cater to only vertices with num of edges >= 16
  if(i >= nv) return;
  GraphWeight selfLoop = 0;
  if(shared_num_edges[tile.thread_rank()] < (GraphElem)tile_sz) { 
    for (GraphElem j = e0[i]; j < e1[i]; j++) {
      if(graph_edgeList_tail[j] == i + base)
         selfLoop += graph_edgeList_weight[j];
    }
    selfLoopVec[i] = selfLoop;
  }
}

template<int tile_sz>
__global__
void count_size_clmap (GraphElem nv, GraphElem* NumClusters, 
   GraphElem* clmap_loc, GraphElem* size_clmap,
   GraphElem* size_lt_ts, GraphElem* list_lt_ts, 
   GraphElem* size_lt_cs1, GraphElem* list_lt_cs1, 
   GraphElem* size_lt_cs2, GraphElem* list_lt_cs2, 
   GraphElem* e0, GraphElem* e1, GraphElem* List_numEdges) {

  int ii = threadIdx.x;
  GraphElem i = blockIdx.x * blockDim.x + threadIdx.x ;

#ifndef USE_32_BIT_GRAPH
  __shared__ long shared_mem_size[MS_THREADBLOCK_SIZE];
  __shared__ long shared_begin_loc[MS_BLOCK_TILE];
#else
  __shared__ int shared_mem_size[MS_THREADBLOCK_SIZE];
  __shared__ int shared_begin_loc[MS_BLOCK_TILE];
#endif
  shared_mem_size[ii] = 0; 

  GraphElem numEdges = 0;
#ifdef DEBUG_CUVITE
  int my_mem_block = 0;
#endif
  // GraphElem numEdges;
  if(i < nv) {
    numEdges = e1[i] - e0[i];
    List_numEdges[i] = numEdges;
// if(numEdges >= 90 && numEdges < 96)
//  printf("vertex[%ld]; numEdges[%ld]\n",i, numEdges);
#ifdef DEBUG_CUVITE
    if(numEdges > 0) my_mem_block = 1;
#endif
    // printf("vertex[%ld]; num_edges[%d] \n", i, numEdges);

    // shared_mem_size[ii] = numEdges * sizeof(GraphElem);
    shared_mem_size[ii] = (GraphElem) numEdges;
    // printf("vertex[%ld]; shared_mem_size[%d] \n", i, shared_mem_size[ii]);
  }
  auto g = cg::this_thread_block();
#if __cuda_arch__ >= 700
  auto tile = cg::partition<tile_sz>(cg::this_thread_block());
#else
  auto tile = cg::tiled_partition<tile_sz>(cg::this_thread_block());
#endif
  auto tileIdx = g.thread_rank()/tile_sz;
  GraphElem *t_shared_mem_size = &shared_mem_size[tileIdx * tile_sz];
  GraphElem *t_shared_begin_loc = &shared_begin_loc[tileIdx];

  tile.sync();
  /// Mem loc inside the block
  GraphElem my_mem_loc = 0;
  for(int s = 0; s < tile.size(); s++) {
    if(tile.thread_rank() > s) my_mem_loc += t_shared_mem_size[s]; 
  }

  tile.sync();
#if 1
  /// Accumulate number of clusters and mem requirement 
  for(int s = 1; s < tile.size(); s *= 2) {
    int indx = 2 * s * tile.thread_rank();

    if(indx < tile.size()) {
      t_shared_mem_size[indx] += t_shared_mem_size[indx+s];
    }
    tile.sync();
  }
#else
  GraphElem tile_sum_size = reduce_sum_tile_shfl<tile_sz>
      (tile, t_shared_mem_size[tile.thread_rank()]);
#endif
#ifdef DEBUG_CUVITE
  int tile_sum_mem = reduce_sum_tile_shfl<tile_sz>(tile, my_mem_block);
#endif
  if(tile.thread_rank() == 0) { 
       // printf("shared_mem_block[%d]; shared_mem_size[%ld]; blockIdx.x[%d] \n", 
       //    tile_sum_mem, shared_mem_size[0], blockIdx.x);
       // printf("shared_mem_block[%d]; shared_mem_size[%ld]; blockIdx.x[%d] \n", 
       //    tile_sum_mem, tile_sum_size, blockIdx.x);
#if __cuda_arch__ >= 600
#ifdef DEBUG_CUVITE
      atomicAdd(&NumClusters[0], tile_sum_mem);
#endif
      t_shared_begin_loc[0] = atomicAdd(&size_clmap[0], t_shared_mem_size[0]); 
      // t_shared_begin_loc[0] = atomicAdd(&size_clmap[0], tile_sum_size); 
#else
#ifdef DEBUG_CUVITE
      my_func_atomicAdd(&NumClusters[0], tile_sum_mem);
#endif
#ifndef USE_32_BIT_GRAPH
      t_shared_begin_loc[0] = my_func_atomicAdd(&size_clmap[0], t_shared_mem_size[0]); 
#else
      t_shared_begin_loc[0] = atomicAdd(&size_clmap[0], t_shared_mem_size[0]); 
#endif
#endif
  }
  tile.sync();
#ifdef DEBUG_CUVITE
  // if(i == 0) printf("Number of edges in a block[%ld]\n", NumClusters[0]);
#endif
  if(i >= nv) return;
  if(numEdges > 0) {
    clmap_loc[i] = t_shared_begin_loc[0] + my_mem_loc; 
    // printf("vertex[%ld]; shared_begin_loc0[%d]; numEdges[%ld]; my_mem_loc[%ld]; clmap_loc[%ld] \n", 
    //     i, t_shared_begin_loc[0], numEdges, my_mem_loc, clmap_loc[i]);
  } else {
    clmap_loc[i] = -1;
  }
  /// Group vertices based on degree
  tile.sync();
  if(numEdges <= (GraphElem)tile_sz) {
    cg::coalesced_group active = cg::coalesced_threads();
#if __cuda_arch__ >= 600
    if(active.thread_rank() == 0)
        t_shared_begin_loc[0] = atomicAdd(&size_lt_ts[0], active.size()); 
#else
#ifndef USE_32_BIT_GRAPH
    if(active.thread_rank() == 0)
        t_shared_begin_loc[0] = my_func_atomicAdd(&size_lt_ts[0], active.size()); 
#else
    if(active.thread_rank() == 0)
        t_shared_begin_loc[0] = atomicAdd(&size_lt_ts[0], active.size()); 
#endif
#endif
    active.sync();
    list_lt_ts[t_shared_begin_loc[0]+active.thread_rank()] = i; 
  }
  tile.sync();
  if(numEdges > (GraphElem)tile_sz && numEdges <= CUT_SIZE_NUM_EDGES1) {
    cg::coalesced_group active = cg::coalesced_threads();
    GraphElem index_loc;
#if __cuda_arch__ >= 600
    if(active.thread_rank() == 0)
        index_loc = atomicAdd(&size_lt_cs1[0], active.size()); 
#else
    if(active.thread_rank() == 0)
#ifndef USE_32_BIT_GRAPH
        index_loc = my_func_atomicAdd(&size_lt_cs1[0], active.size()); 
#else
        index_loc = atomicAdd(&size_lt_cs1[0], active.size()); 
#endif
#endif
    // active.sync();
    list_lt_cs1[active.shfl(index_loc, 0)+active.thread_rank()] = i; 
  }
  tile.sync();
  if(numEdges > CUT_SIZE_NUM_EDGES1) {
    // printf("vertex[%ld]; lt_cs2_num_edges[%ld] \n", i, numEdges);
    cg::coalesced_group active = cg::coalesced_threads();
    GraphElem index_loc;
#if __cuda_arch__ >= 600
    if(active.thread_rank() == 0)
        index_loc = atomicAdd(&size_lt_cs2[0], active.size()); 
#else
#ifndef USE_32_BIT_GRAPH
    if(active.thread_rank() == 0)
        index_loc = my_func_atomicAdd(&size_lt_cs2[0], active.size()); 
#else
    if(active.thread_rank() == 0)
        index_loc = atomicAdd(&size_lt_cs2[0], active.size()); 
#endif
#endif
    // active.sync();
    list_lt_cs2[active.shfl(index_loc, 0)+active.thread_rank()] = i; 
  }
}

__global__ 
    void gpu_distExecuteLouvainIteration(
    const GraphElem nv, 
    GraphElem* graph_edgeListIndexes,
    GraphElem* GraphEdge_low, GraphElem* GraphEdge_high, 
    int me, const GraphElem base, const GraphElem bound
) {

  // int ii = threadIdx.x;
  GraphElem i = blockIdx.x * blockDim.x + threadIdx.x ;
  if(i >= nv) return;

  GraphElem e0, e1; 
  e0 = graph_edgeListIndexes[i];
  e1 = graph_edgeListIndexes[i+1];
  /// Store variables to global memory
  GraphEdge_low[i] = e0;
  GraphEdge_high[i] = e1;
}

template<class T>
__global__ void print_device_vector(T *given_vec, GraphElem size_vec) 
{
  GraphElem ii = threadIdx.x;
  GraphElem i = blockIdx.x * blockDim.x + threadIdx.x ;

  if(i >= size_vec) return;
  printf("i[%ld]; VEC_VALUE[%f]\n", i, given_vec[i]);
}

__global__ void print_device_vector2(GraphWeight* given_vec, 
    GraphElem size_vec) 
{
  // GraphElem ii = threadIdx.x;
  GraphElem i = blockIdx.x * blockDim.x + threadIdx.x ;

  if(i >= size_vec) return;
  printf("i[%ld]; VEC_VALUE[%f]\n", i, given_vec[i]);
}

void set_gpuDevices(int *me) 
{
  int num_gpuDevices;
  cudaGetDeviceCount(&num_gpuDevices);
#if 1
  /// split MPI comm to get local node rank
  /// cudaSetDevice to local node rank
  MPI_Comm loc_comm;
  MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, *me,
     MPI_INFO_NULL, &loc_comm); 
  int node_rank = -1;
  MPI_Comm_rank(loc_comm,&node_rank);
  // std::cout << "me:" << *me << "; node rank:" << node_rank << std::endl;
  cudaError_t cudaStat;
  cudaStat = cudaSetDevice(node_rank);
  for (int dev_id = 0; dev_id < num_gpuDevices; dev_id++) {
    if( node_rank%num_gpuDevices == dev_id) {
      cudaStat = cudaSetDevice(dev_id);
      // std::cout << "me[" << *me << "]; node rank[" << node_rank 
      //   << "]; dev_id[" << dev_id << "]" << std::endl;
    }
  }
  if(cudaStat != cudaSuccess)
     printf("Process %d; ERROR DEVICE FAILED\n", *me);
  MPI_Comm_free(&loc_comm);
#else
  /// cudaSetDevice to MPI rank
  cudaError_t cudaStat;
  for (int dev_id = 0; dev_id < num_gpuDevices; dev_id++) {
    if( *me%num_gpuDevices == dev_id) cudaStat = cudaSetDevice(dev_id);
  }
  // cudaStat = cudaSetDevice(2);
  if(cudaStat != cudaSuccess)
     printf("Process %d; ERROR DEVICE FAILED\n", *me);
#endif
}

int gpu_for_louvain_iteration(
    const GraphElem nv, const DistGraph &dg,
    CommunityVector &currComm,
    CommunityVector &targetComm,
    GraphWeightVector &vDegree,
    CommVector &localCinfo, 
    CommVector &localCupdate,
    VertexCommMap &remoteComm,
    const CommMap &remoteCinfo,
    CommMap &remoteCupdate,
    const double constantForSecondTerm,
    GraphWeightVector &clusterWeight, 
    int me, int numIters, GpuGraph &gpu_graph)
{
  if(nv <= 0) return 1;
  const GraphElem base = dg.getBase(me), bound = dg.getBound(me);
#ifdef USE_HYBRID_CPU_GPU  // Run hybrid CPU-GPU code

  // create a temporary target buffer
  std::vector<GraphElem> temp_targetComm_cpu = targetComm;
  std::vector<GraphElem> temp_targetComm_gpu = targetComm;

  static GraphElem num_vertex_cpu, num_vertex_gpu;

  static double time_cpu, time_gpu;

  if(numIters == 1)
  {
    time_cpu = 1.e0;
    time_gpu = 1.e0;
  }

  if(time_cpu >= time_gpu){
    num_vertex_cpu = num_vertex_cpu  - 
      nv * (time_cpu - time_gpu) / (time_cpu + time_gpu) / 3;
  } 

  if(time_cpu < time_gpu){
    num_vertex_cpu = num_vertex_cpu  + 
      nv * (time_gpu - time_cpu) / (time_cpu + time_gpu) / 3;
  } 

  if(num_vertex_cpu <= 0) num_vertex_cpu = nv * 1/80;
  if(num_vertex_cpu > nv ) num_vertex_cpu = nv * 9/10;

  // if(numIters == 1) num_vertex_cpu =  nv * 1/20; 
  if(numIters == 1) num_vertex_cpu =  nv * 1/3; 
  // num_vertex_cpu = 0;

  num_vertex_gpu = nv - num_vertex_cpu;

#ifdef PRINT_HYBRID
  std::cout << "me[" << me << "]; nv: " << nv << "; num_vertex_gpu: " << 
               num_vertex_gpu << "; num_vertex_cpu: " << num_vertex_cpu << std::endl;
#endif
 
  int num_avail_threads = omp_get_num_threads();
  const int maxNumThreads = omp_get_max_threads();
  omp_set_num_threads(2);
  omp_set_nested(1);

  double t0 = timer();
#pragma omp parallel sections
  {
#pragma omp section
  { //call CPU function
  omp_set_num_threads(8);

#pragma omp parallel default(none), shared(clusterWeight, localCupdate, currComm, targetComm, \
        vDegree, localCinfo, remoteCinfo, remoteComm, dg, remoteCupdate, me, \
        temp_targetComm_cpu, num_vertex_gpu), \
        firstprivate(constantForSecondTerm)
      {
          // distCleanCWandCU(nv, clusterWeight, localCupdate);
#ifdef OMP_SCHEDULE_RUNTIME
#pragma omp for schedule(runtime)
#else
#pragma omp for schedule(guided)
#endif
          for (GraphElem i = num_vertex_gpu; i < nv; i++) {
              distExecuteLouvainIteration_hybrid(i, dg, currComm, targetComm, vDegree, localCinfo,
                      localCupdate, remoteComm, remoteCinfo, remoteCupdate,
                      constantForSecondTerm, clusterWeight, me, temp_targetComm_cpu);
          }
      }
    double t1 = timer();
    time_cpu = t1 - t0;
#ifdef PRINT_HYBRID
    std::cout << "me[" << me << "]; Time CPU: " << time_cpu << std::endl;
#endif
  }  // close cpu section 
#pragma omp section
  { /// call GPU function
  omp_set_num_threads(4);
  if (num_vertex_gpu > 0) {

  /// size equal to nv X GraphElem
  int mem_size_GraphElem_nv = sizeof(GraphElem) * nv;
  /// size equal to nv X GraphWeight
  int mem_size_GraphWeight_nv = sizeof(GraphWeight) * nv;
 
  /// All following have size = nv
  GraphElem size_localCinfo = localCinfo.size();
  GraphElem* temp_ModlocalCinfo_size = gpu_graph.getPinned_ModlocalCinfo_size();
  GraphWeight* temp_ModlocalCinfo_degree = gpu_graph.getPinned_ModlocalCinfo_degree(); 

#pragma omp parallel default(none), \
       shared(localCinfo, temp_ModlocalCinfo_size, temp_ModlocalCinfo_degree)
#pragma omp for schedule(guided)
  for(int ii=0; ii<localCinfo.size(); ii++) {
    temp_ModlocalCinfo_size[ii] = localCinfo[ii].size;
    temp_ModlocalCinfo_degree[ii] = localCinfo[ii].degree;
  }

  /// Remote Community Info
  /// First get the keys of remoteCinfo map
  std::vector<GraphElem> temp_remoteCinfo_key = extract_keys_CommMap(remoteCinfo);

  GraphElem size_remoteCinfo = remoteCinfo.size();
  /// split RemoteCinfo into vectors for different struct elements
  std::vector<GraphElem>temp_remoteCinfo_size = 
     extract_value_CommMap_size(remoteCinfo);

  std::vector<GraphWeight>temp_remoteCinfo_degree = 
     extract_value_CommMap_degree(remoteCinfo);

  /// now modify currComm to include remoteComm
  GraphElem* temp_ModlocalCinfo_oComm = gpu_graph.getPinned_ModlocalCinfo_oComm();
  ClusterLocalMap localCinfo_to_remoteCinfo_map;
  ClusterLocalMap::const_iterator storedAlready;
  GraphElem temp_counter_01 = 0; 
  std::vector<GraphElem> ModcurrComm = currComm;
  for(int ii=0; ii<temp_remoteCinfo_key.size(); ii++) {
    GraphElem temp_Comm = temp_remoteCinfo_key[ii];
    if(temp_Comm < base || temp_Comm >= bound) 
    {
      storedAlready = localCinfo_to_remoteCinfo_map.find(temp_Comm);
      if(storedAlready == localCinfo_to_remoteCinfo_map.end()) {
        localCinfo_to_remoteCinfo_map.insert(std::make_pair(
            temp_Comm, (temp_counter_01+bound)));
        temp_ModlocalCinfo_size[size_localCinfo+temp_counter_01] = temp_remoteCinfo_size[ii];
        temp_ModlocalCinfo_degree[size_localCinfo+temp_counter_01] = temp_remoteCinfo_degree[ii];
        temp_ModlocalCinfo_oComm[temp_counter_01] = temp_Comm;
        temp_counter_01++;
      }
    } 
  }
  GraphElem size_ModlocalCinfo = size_localCinfo+temp_counter_01;
  GraphElem size_ModlocalCinfo_oComm = temp_counter_01;
  std::vector<GraphElem>().swap(temp_remoteCinfo_key);
  CommunityVector().swap(temp_remoteCinfo_size);
  GraphWeightVector().swap(temp_remoteCinfo_degree);

  // remoteComm is broken into 2 arrays
  std::vector<GraphElem> temp_remoteComm_v;
  temp_remoteComm_v = extract_vertex_VertexCommMap(remoteComm);
  std::vector<GraphElem> temp_remoteComm_comm;
  temp_remoteComm_comm = extract_comm_VertexCommMap(remoteComm);

  /// Create map for remoteComm tail mapped to currComm
  ClusterLocalMap remoteComm_to_currComm_map_v;
  ClusterLocalMap::const_iterator storedAlready_v;
  ClusterLocalMap::const_iterator storedAlready_comm;
  GraphElem temp_counter_02 = 0; 
  GraphElem temp_tail;
  GraphElem temp_comm, temp_comm_mapped;
  // First modify currComm
#pragma omp parallel default(none), \
       shared(ModcurrComm, localCinfo_to_remoteCinfo_map), \
       private(temp_comm, storedAlready_comm)
#pragma omp for schedule(guided)
  for(int ii = 0; ii < ModcurrComm.size(); ii++) {
    temp_comm = ModcurrComm[ii];
    if(temp_comm < base || temp_comm >= bound) {
      storedAlready_comm = localCinfo_to_remoteCinfo_map.find(temp_comm);
      ModcurrComm[ii] = storedAlready_comm->second;
    }
  }
  // Next modify currComm to include remoteComm
  for(int ii=0; ii<temp_remoteComm_comm.size(); ii++) {
    temp_comm = temp_remoteComm_comm[ii];
    temp_tail = temp_remoteComm_v[ii];
    if(temp_comm < base || temp_comm >= bound) {
      storedAlready_comm = localCinfo_to_remoteCinfo_map.find(temp_comm);

      temp_comm_mapped = storedAlready_comm->second;
      temp_remoteComm_comm[ii] = temp_comm_mapped; 
      ModcurrComm.push_back(temp_comm_mapped);

    } else {
      ModcurrComm.push_back(temp_comm);
      /// check line below
    }
    if(temp_tail < base || temp_tail >= bound) {
      storedAlready_v = remoteComm_to_currComm_map_v.find(temp_tail);
      if(storedAlready_v == remoteComm_to_currComm_map_v.end()) {
        if(temp_tail < base || temp_tail >= bound) {
          remoteComm_to_currComm_map_v.insert(std::make_pair(
              temp_tail, (bound + temp_counter_02) ));
              temp_remoteComm_v[ii] = bound + temp_counter_02;
        } 
        temp_counter_02++;
      }
    } 
  }
  // }

  std::vector<GraphElem>().swap(temp_remoteComm_v);
  std::vector<GraphElem>().swap(temp_remoteComm_comm);

  // comm_node_info remote_comm_info;
  const Graph &g = dg.getLocalGraph();

  GraphElem size_edgeListIndexes = g.edgeListIndexes.size();

  GraphElem* temp_graph_edgeList_tail = gpu_graph.getPinned_edgeList_tail();
  GraphWeight* temp_graph_edgeList_weight = gpu_graph.getPinned_edgeList_weight();

#pragma omp parallel default(none), shared(g, temp_graph_edgeList_tail, \
        temp_graph_edgeList_weight, remoteComm_to_currComm_map_v), \
        private(storedAlready)
#pragma omp for schedule(guided)
  for(int ii=0; ii<g.edgeList.size(); ii++) {
    ClusterLocalMap edgeList_tail_map;
    GraphElem temp_tail = g.edgeList[ii].tail;
    temp_graph_edgeList_tail[ii] = temp_tail;
    temp_graph_edgeList_weight[ii] = g.edgeList[ii].weight;

    if(temp_tail < base || temp_tail >= bound) {
      /// use remoteComm_to_currComm_map_v map instead 
      storedAlready = remoteComm_to_currComm_map_v.find(temp_tail);
      if(storedAlready != edgeList_tail_map.end()) {
        temp_graph_edgeList_tail[ii] = storedAlready->second;
      }
    }
  }

#ifdef PRINT_TIMEDS
  double t_remap = timer();
  double time_remap = t_remap - t0;
  std::cout << "me[" << me << "]; Time gpu_remap: " << time_remap << std::endl;
#endif

  /// Get pointers to memory of device arrays
  GraphElem* dev_currComm = gpu_graph.get_currComm();
  GraphElem* dev_ModlocalTarget = gpu_graph.get_ModlocalTarget();
  GraphWeight* dev_vDegree = gpu_graph.get_vDegree();
  GraphWeight* dev_clusterWeight = gpu_graph.get_clusterWeight();
  GraphElem* dev_edgeListIndexes = gpu_graph.get_edgeListIndexes();

  GraphElem* dev_ModcurrComm = gpu_graph.get_ModcurrComm();
  GraphElem* dev_localCinfo_size = gpu_graph.get_ModlocalCinfo_size();
  GraphWeight* dev_localCinfo_degree = gpu_graph.get_ModlocalCinfo_degree();
  GraphElem* dev_localCinfo_oComm = gpu_graph.get_ModlocalCinfo_oComm();
  GraphElem* dev_graph_edgeList_tail = gpu_graph.get_edgeList_tail();
  GraphWeight* dev_graph_edgeList_weight = gpu_graph.get_edgeList_weight();

  GraphElem* dev_unique_comm_array = gpu_graph.get_unique_comm_array();
  GraphWeight* dev_unique_weight_array = gpu_graph.get_unique_weight_array();

  gpu_graph.cpyVecTodev(currComm, dev_currComm);
  gpu_graph.cpyVecTodev(vDegree, dev_vDegree);
  gpu_graph.cpyVecTodev(clusterWeight, dev_clusterWeight);
  gpu_graph.cpyVecTodev(g.edgeListIndexes, dev_edgeListIndexes);

  bool check_ModlocalCinfo_memory = gpu_graph.checkModCommMemory(size_ModlocalCinfo);
  assert(check_ModlocalCinfo_memory);
  bool check_ModlocalCinfoComm_memory = gpu_graph.checkModCommMemory(size_ModlocalCinfo_oComm);
  assert(check_ModlocalCinfoComm_memory);
  gpu_graph.cpyArrayTodev(temp_ModlocalCinfo_size, dev_localCinfo_size, size_ModlocalCinfo);
  gpu_graph.cpyArrayTodev(temp_ModlocalCinfo_degree, dev_localCinfo_degree, size_ModlocalCinfo);
  gpu_graph.cpyArrayTodev(temp_ModlocalCinfo_oComm, dev_localCinfo_oComm, size_ModlocalCinfo_oComm);

  gpu_graph.cpyArrayTodev(temp_graph_edgeList_tail, dev_graph_edgeList_tail, 
                      (GraphElem)g.edgeList.size());
  gpu_graph.cpyArrayTodev(temp_graph_edgeList_weight, dev_graph_edgeList_weight, 
                      (GraphElem)g.edgeList.size());

  bool check_ModcurrComm_memory = gpu_graph.checkModCommMemory(
       (GraphElem)ModcurrComm.size());
  assert(check_ModcurrComm_memory);
  gpu_graph.cpyVecTodev(ModcurrComm, dev_ModcurrComm);

  GraphElem* dev_GraphEdge_low = gpu_graph.get_GraphEdge_low();
  GraphElem* dev_GraphEdge_high = gpu_graph.get_GraphEdge_high();

  /// allocate device memory for filling in comm and weights
  GraphElem* dev_clmap_comm = gpu_graph.get_clmap_comm();
  GraphWeight* dev_clmap_weight = gpu_graph.get_clmap_weight();
  GraphElem clmapSize;

  GraphElem* dev_clmap_loc = gpu_graph.get_clmap_loc();

  GraphElem* dev_List_numEdges = gpu_graph.get_List_numEdges();

  GraphElem* dev_list_lt_ts = gpu_graph.get_dev_list_lt_ts();
  GraphElem* dev_list_lt_cs1 = gpu_graph.get_dev_list_lt_cs1();
  GraphElem* dev_list_lt_cs2 = gpu_graph.get_dev_list_lt_cs2();

#ifdef PRINT_TIMEDS
  cudaDeviceSynchronize();
  double t_dtrans = timer();
  double time_dtrans = t_dtrans - t0;
  std::cout << "me[" << me << "]; Time gpu_dtrans: " << time_dtrans << std::endl;
#endif

if(numIters == 1) 
{
  CUDA_SAFE(cudaMemset(dev_GraphEdge_low, 0, mem_size_GraphElem_nv));
  CUDA_SAFE(cudaMemset(dev_GraphEdge_high, 0, mem_size_GraphElem_nv));

  dim3 numBlocks01( (nv-1) / L_THREADBLOCK_SIZE + 1);
  dim3 Block_dim01(L_THREADBLOCK_SIZE);


  gpu_distExecuteLouvainIteration<<<numBlocks01,Block_dim01>>>(
                    nv,
                    dev_edgeListIndexes,
                    dev_GraphEdge_low, dev_GraphEdge_high, 
                    me, base, bound);

  CUDA_SAFE(cudaMemset(dev_List_numEdges, 0, mem_size_GraphElem_nv));

  CUDA_SAFE(cudaMemset(dev_clmap_loc, 0, mem_size_GraphElem_nv));

  GraphElem* dev_NumClusters = gpu_graph.get_NumClusters();
  CUDA_SAFE(cudaMemset(dev_NumClusters, 0, sizeof(GraphElem)));

  GraphElem* dev_size_clmap = gpu_graph.get_size_clmap();
  CUDA_SAFE(cudaMemset(dev_size_clmap, 0, sizeof(GraphElem)));

  GraphElem* dev_size_lt_ts = gpu_graph.get_dev_size_lt_ts();
  CUDA_SAFE(cudaMemset(dev_size_lt_ts, 0, sizeof(GraphElem)));
  GraphElem* dev_size_lt_cs1 = gpu_graph.get_dev_size_lt_cs1();
  CUDA_SAFE(cudaMemset(dev_size_lt_cs1, 0, sizeof(GraphElem)));
  GraphElem* dev_size_lt_cs2 = gpu_graph.get_dev_size_lt_cs2();
  CUDA_SAFE(cudaMemset(dev_size_lt_cs2, 0, sizeof(GraphElem)));

  CUDA_SAFE(cudaMemset(dev_list_lt_ts, 0, mem_size_GraphElem_nv));
  CUDA_SAFE(cudaMemset(dev_list_lt_cs1, 0, mem_size_GraphElem_nv));
  CUDA_SAFE(cudaMemset(dev_list_lt_cs2, 0, mem_size_GraphElem_nv));

  dim3 numBlocks02( (nv-1) / MS_THREADBLOCK_SIZE + 1);
  dim3 Block_dim02(MS_THREADBLOCK_SIZE);

  count_size_clmap<PHY_WRP_SZ><<<numBlocks02,Block_dim02>>>(nv, dev_NumClusters, 
     dev_clmap_loc, dev_size_clmap, 
     dev_size_lt_ts, dev_list_lt_ts, 
     dev_size_lt_cs1, dev_list_lt_cs1, 
     dev_size_lt_cs2, dev_list_lt_cs2, 
     dev_GraphEdge_low, dev_GraphEdge_high, dev_List_numEdges);  

  /// copy to host number of clusters and size of cluster map memory
#ifdef DEBUG_CUVITE
  GraphElem NumClusters = 0;
  CUDA_SAFE(cudaMemcpy(&NumClusters, dev_NumClusters, 
     sizeof(GraphElem), cudaMemcpyDeviceToHost));
  std::cout << "me[" << me << "]; NumClusters[ " << NumClusters << "]" << std::endl;
#endif
  CUDA_SAFE(cudaMemcpy(&clmapSize, dev_size_clmap, 
     sizeof(GraphElem), cudaMemcpyDeviceToHost));
  gpu_graph.set_clmapSize(clmapSize);

  dev_clmap_comm = gpu_graph.getDevMem_clmapComm(clmapSize);
  dev_clmap_weight = gpu_graph.getDevMem_clmapWeight(clmapSize);

  gpu_graph.set_size_lt_ts();
  gpu_graph.set_size_lt_cs1();
  gpu_graph.set_size_lt_cs2();

}

#ifdef PRINT_TIMEDS
  cudaDeviceSynchronize();
  double t_iter1 = timer();
  double time_iter1 = t_iter1 - t0;
  std::cout << "me[" << me << "]; Time gpu_iter1: " << time_iter1 << std::endl;
#endif

  GraphElem size_lt_ts = gpu_graph.get_size_lt_ts();
  GraphElem size_lt_cs1 = gpu_graph.get_size_lt_cs1();
  GraphElem size_lt_cs2 = gpu_graph.get_size_lt_cs2();

  clmapSize = gpu_graph.get_clmapSize();
  CUDA_SAFE(cudaMemset(dev_clmap_comm, 0, 
     clmapSize * sizeof(GraphElem)));
  CUDA_SAFE(cudaMemset(dev_clmap_weight, 0, 
     clmapSize * sizeof(GraphWeight)));

  GraphWeight* dev_selfLoopVec = gpu_graph.get_selfLoopVec();
  CUDA_SAFE(cudaMemset(dev_selfLoopVec, 0, mem_size_GraphWeight_nv));

  dim3 numBlocks03( (num_vertex_gpu-1) / S_THREADBLOCK_SIZE + 1);
  dim3 Block_dim03(S_THREADBLOCK_SIZE);
  distBuildLocalMapCounter<PHY_WRP_SZ><<<numBlocks03,Block_dim03>>>(
              num_vertex_gpu, dev_GraphEdge_low, dev_GraphEdge_high,
              dev_graph_edgeList_tail, 
              dev_graph_edgeList_weight,
              dev_List_numEdges, dev_selfLoopVec, 
              base); // , bound); 

  GraphElem* dev_uniq_clus_vec = gpu_graph.get_uniq_clus_vec();
  CUDA_SAFE(cudaMemset(dev_uniq_clus_vec, 0, mem_size_GraphElem_nv));
  GraphWeight* dev_counter = gpu_graph.get_counter();
  CUDA_SAFE(cudaMemset(dev_counter, 0, mem_size_GraphWeight_nv));

  const int num_streams = 2;
  cudaStream_t streams[num_streams];
  for(auto i_streams = 0; i_streams < num_streams; i_streams++) {
    CUDA_SAFE(cudaStreamCreate(&streams[i_streams]) );
  }

  dim3 numBlocks05( (num_vertex_gpu-1) / S_THREADBLOCK_SIZE + 1);
  dim3 Block_dim05(S_THREADBLOCK_SIZE);
  distGetMaxIndex<PHY_WRP_SZ><<<numBlocks05,Block_dim05, 0, streams[0]>>>(
              num_vertex_gpu, 
              dev_GraphEdge_low, // dev_GraphEdge_high,
              dev_graph_edgeList_tail,
              dev_graph_edgeList_weight,
              dev_ModcurrComm,
              dev_clmap_loc, 
              dev_clmap_comm, dev_clmap_weight, 
              dev_List_numEdges,
              dev_uniq_clus_vec, dev_counter,
              base); // , bound);

#if 0
  distGetMaxIndex_large<PHY_WRP_SZ><<<numBlocks05,Block_dim05, 0, streams[0]>>>(
              num_vertex_gpu, 
              dev_GraphEdge_low, dev_GraphEdge_high,
              dev_graph_edgeList_tail,
              dev_graph_edgeList_weight,
              dev_ModcurrComm,
              dev_clmap_loc, 
              dev_clmap_comm, dev_clmap_weight, 
              dev_List_numEdges,
              dev_uniq_clus_vec, dev_counter,
              base, bound);
#else
  if(size_lt_cs2 > 0) {
    dim3 numBlocks052(FINDING_UNIQCOMM_NUM_BLOCKS);
    dim3 Block_dim052(FINDING_UNIQCOMM_BLOCK_SIZE);
    GraphElem nv_chunk_size;
    nv_chunk_size = (size_lt_cs2 - 1) / FINDING_UNIQCOMM_NUM_BLOCKS + 1;

    assert(ModcurrComm.size() <= FINDING_UNIQCOMM_ARRAY_SIZE);
    distGetMaxIndex_large_new<PHY_WRP_SZ><<<numBlocks052,Block_dim052, 0, streams[0]>>>(
                me, numIters,
                num_vertex_gpu, nv_chunk_size,
                size_lt_cs2, dev_list_lt_cs2,
                ModcurrComm.size(), // size_ModlocalCinfo,
                dev_unique_comm_array,
                dev_unique_weight_array,
                dev_GraphEdge_low, // dev_GraphEdge_high,
                dev_graph_edgeList_tail,
                dev_graph_edgeList_weight,
                dev_ModcurrComm,
                dev_clmap_loc,
                dev_clmap_comm, dev_clmap_weight,
                dev_List_numEdges,
                dev_uniq_clus_vec, dev_counter,
                base); // , bound);
  }
#endif
  CUDA_SAFE(cudaMemcpy(dev_ModlocalTarget, dev_ModcurrComm, 
     sizeof(GraphElem)*ModcurrComm.size(), cudaMemcpyDeviceToDevice));

  dim3 numBlocks06( (num_vertex_gpu-1) / S_THREADBLOCK_SIZE + 1);
  dim3 Block_dim06(S_THREADBLOCK_SIZE);
  computeMaxIndex<PHY_WRP_SZ><<<numBlocks06,Block_dim06, 0, streams[0]>>>(
              // nv, 
              num_vertex_gpu,
              dev_currComm,
              dev_ModcurrComm,
              dev_localCinfo_size, 
              dev_localCinfo_degree, 
              dev_localCinfo_oComm, 
              dev_selfLoopVec, 
              dev_uniq_clus_vec, dev_counter,
              dev_clmap_loc, 
              dev_clmap_comm, dev_clmap_weight, 
              dev_vDegree,
              dev_ModlocalTarget,
              dev_clusterWeight,
              constantForSecondTerm,
              base, bound);

  computeMaxIndex_large<PHY_WRP_SZ><<<numBlocks06,Block_dim06, 0, streams[0]>>>(
              // nv, 
              num_vertex_gpu,
              dev_currComm,
              dev_ModcurrComm,
              dev_localCinfo_size, 
              dev_localCinfo_degree, 
              dev_localCinfo_oComm, 
              dev_selfLoopVec, 
              dev_uniq_clus_vec, dev_counter,
              dev_clmap_loc, 
              dev_clmap_comm, dev_clmap_weight, 
              dev_vDegree,
              dev_ModlocalTarget,
              dev_clusterWeight,
              constantForSecondTerm,
              base, bound);

#ifdef PRINT_TIMEDS
  cudaDeviceSynchronize();
  double t_kernels = timer();
  double time_kernels = t_kernels - t0;
  std::cout << "me[" << me << "]; Time gpu_kernels: " << time_iter1 << std::endl;
#endif

  for(auto i_streams = 0; i_streams < num_streams; i_streams++) {
    cudaStreamSynchronize(streams[i_streams]);
  }

  for(auto i_streams = 0; i_streams < num_streams; i_streams++) {
    cudaStreamDestroy(streams[i_streams]);
  }

  /// Copy Targets to Host
  CUDA_SAFE(cudaMemcpy(&temp_targetComm_gpu[0],
     dev_ModlocalTarget, 
     (num_vertex_gpu*sizeof(GraphElem)), cudaMemcpyDeviceToHost));
     /// Copy clusterWeight to Host
  CUDA_SAFE(cudaMemcpy(&clusterWeight[0],
     dev_clusterWeight, 
     (num_vertex_gpu*sizeof(GraphWeight)), cudaMemcpyDeviceToHost));

#ifdef PRINT_TIMEDS
  cudaDeviceSynchronize();
  double t_kd2h = timer();
  double time_kd2h = t_kd2h - t0;
  std::cout << "me[" << me << "]; Time gpu_kd2h: " << time_kd2h << std::endl;
#endif

  std::vector<GraphElem>().swap(ModcurrComm);

    double t1 = timer();
    time_gpu = t1 - t0;
#ifdef PRINT_HYBRID
    std::cout << "me[" << me << "]; Time GPU: " << time_gpu << std::endl;
#endif
    }  // if (num_vertex_gpu > 0) condition
    }  // close gpu secton 
  }  // close parallel 

  memcpy(&temp_targetComm_gpu[0+num_vertex_gpu], 
         &temp_targetComm_cpu[0+num_vertex_gpu], 
         num_vertex_cpu*sizeof(GraphElem));

  omp_set_num_threads(14);
  updateLocalTarget_gpu (
    nv, 
    currComm,
    targetComm,
    vDegree,
    remoteCupdate,
    temp_targetComm_gpu,
    localCupdate,
    base, bound, numIters); 

  std::vector<GraphElem>().swap(temp_targetComm_cpu);
  std::vector<GraphElem>().swap(temp_targetComm_gpu);
#else  // else option runs GPU code below
  omp_set_num_threads(14);
  double t0 = timer();
  // GPU only code
  /// size equal to nv X GraphElem
  int mem_size_GraphElem_nv = sizeof(GraphElem) * nv;
  /// size equal to nv X GraphWeight
  int mem_size_GraphWeight_nv = sizeof(GraphWeight) * nv;
 
  GraphElem size_localCinfo = localCinfo.size();
  /// split localCinfo into vectors for different struct elements
  GraphElem* temp_ModlocalCinfo_size = gpu_graph.getPinned_ModlocalCinfo_size();
  GraphWeight* temp_ModlocalCinfo_degree = gpu_graph.getPinned_ModlocalCinfo_degree(); 


#pragma omp parallel default(none), \
       shared(localCinfo, temp_ModlocalCinfo_size, temp_ModlocalCinfo_degree)
#pragma omp for schedule(guided)
  for(int ii=0; ii<localCinfo.size(); ii++) {
    temp_ModlocalCinfo_size[ii] = localCinfo[ii].size;
    temp_ModlocalCinfo_degree[ii] = localCinfo[ii].degree;
  }

  /// Remote Community Info
  /// First get the keys of remoteCinfo map
  std::vector<GraphElem> temp_remoteCinfo_key = extract_keys_CommMap(remoteCinfo);

  GraphElem size_remoteCinfo = remoteCinfo.size();
  /// split RemoteCinfo into vectors for different struct elements
  std::vector<GraphElem>temp_remoteCinfo_size = 
     extract_value_CommMap_size(remoteCinfo);

  std::vector<GraphWeight>temp_remoteCinfo_degree = 
     extract_value_CommMap_degree(remoteCinfo);

  /// now modify currComm to include remoteComm
  GraphElem* temp_ModlocalCinfo_oComm = gpu_graph.getPinned_ModlocalCinfo_oComm();
  ClusterLocalMap localCinfo_to_remoteCinfo_map;
  ClusterLocalMap::const_iterator storedAlready;
  GraphElem temp_counter_01 = 0; 
  std::vector<GraphElem> ModcurrComm = currComm;
  for(int ii=0; ii<temp_remoteCinfo_key.size(); ii++) {
    GraphElem temp_Comm = temp_remoteCinfo_key[ii];
    if(temp_Comm < base || temp_Comm >= bound) 
    {
      storedAlready = localCinfo_to_remoteCinfo_map.find(temp_Comm);
      if(storedAlready == localCinfo_to_remoteCinfo_map.end()) {
        localCinfo_to_remoteCinfo_map.insert(std::make_pair(
            temp_Comm, (temp_counter_01+bound)));
        temp_ModlocalCinfo_size[size_localCinfo+temp_counter_01] = temp_remoteCinfo_size[ii];
        temp_ModlocalCinfo_degree[size_localCinfo+temp_counter_01] = temp_remoteCinfo_degree[ii];
        temp_ModlocalCinfo_oComm[temp_counter_01] = temp_Comm;
        temp_counter_01++;
      }
    } 
  }
  GraphElem size_ModlocalCinfo = size_localCinfo+temp_counter_01;
  GraphElem size_ModlocalCinfo_oComm = temp_counter_01;
  std::vector<GraphElem>().swap(temp_remoteCinfo_key);
  CommunityVector().swap(temp_remoteCinfo_size);
  GraphWeightVector().swap(temp_remoteCinfo_degree);

  // remoteComm is broken into 2 arrays
  std::vector<GraphElem> temp_remoteComm_v;
  temp_remoteComm_v = extract_vertex_VertexCommMap(remoteComm);
  std::vector<GraphElem> temp_remoteComm_comm;
  temp_remoteComm_comm = extract_comm_VertexCommMap(remoteComm);

  /// Create map for remoteComm tail mapped to currComm
  ClusterLocalMap remoteComm_to_currComm_map_v;
  ClusterLocalMap::const_iterator storedAlready_v;
  ClusterLocalMap::const_iterator storedAlready_comm;
  GraphElem temp_counter_02 = 0; 
  GraphElem temp_tail;
  GraphElem temp_comm, temp_comm_mapped;
  // First modify currComm
#pragma omp parallel default(none), \
       shared(ModcurrComm, localCinfo_to_remoteCinfo_map), \
       private(temp_comm, storedAlready_comm)
#pragma omp for schedule(guided)
  for(int ii = 0; ii < ModcurrComm.size(); ii++) {
    temp_comm = ModcurrComm[ii];
    if(temp_comm < base || temp_comm >= bound) {
      storedAlready_comm = localCinfo_to_remoteCinfo_map.find(temp_comm);
      ModcurrComm[ii] = storedAlready_comm->second;
    }
  }
  // Next modify currComm to include remoteComm
  for(int ii=0; ii<temp_remoteComm_comm.size(); ii++) {
    temp_comm = temp_remoteComm_comm[ii];
    temp_tail = temp_remoteComm_v[ii];
    if(temp_comm < base || temp_comm >= bound) {
      storedAlready_comm = localCinfo_to_remoteCinfo_map.find(temp_comm);

      temp_comm_mapped = storedAlready_comm->second;
      temp_remoteComm_comm[ii] = temp_comm_mapped; 
      ModcurrComm.push_back(temp_comm_mapped);

    } else {
      ModcurrComm.push_back(temp_comm);
      /// check line below
    }
    if(temp_tail < base || temp_tail >= bound) {
      storedAlready_v = remoteComm_to_currComm_map_v.find(temp_tail);
      if(storedAlready_v == remoteComm_to_currComm_map_v.end()) {
        if(temp_tail < base || temp_tail >= bound) {
          remoteComm_to_currComm_map_v.insert(std::make_pair(
              temp_tail, (bound + temp_counter_02) ));
              temp_remoteComm_v[ii] = bound + temp_counter_02;
        } 
        temp_counter_02++;
      }
    } 
  }

  std::vector<GraphElem>().swap(temp_remoteComm_v);
  std::vector<GraphElem>().swap(temp_remoteComm_comm);

  // comm_node_info remote_comm_info;
  const Graph &g = dg.getLocalGraph();

  GraphElem size_edgeListIndexes = g.edgeListIndexes.size();

  GraphElem* temp_graph_edgeList_tail = gpu_graph.getPinned_edgeList_tail();
  GraphWeight* temp_graph_edgeList_weight = gpu_graph.getPinned_edgeList_weight();

#pragma omp parallel default(none), shared(g, temp_graph_edgeList_tail, \
        temp_graph_edgeList_weight, remoteComm_to_currComm_map_v), \
        private(storedAlready)
#pragma omp for schedule(guided)
  for(int ii=0; ii<g.edgeList.size(); ii++) {
    ClusterLocalMap edgeList_tail_map;
    GraphElem temp_tail = g.edgeList[ii].tail;
    temp_graph_edgeList_tail[ii] = temp_tail;
    temp_graph_edgeList_weight[ii] = g.edgeList[ii].weight;

    if(temp_tail < base || temp_tail >= bound) {
      /// use remoteComm_to_currComm_map_v map instead 
      storedAlready = remoteComm_to_currComm_map_v.find(temp_tail);
      if(storedAlready != edgeList_tail_map.end()) {
        temp_graph_edgeList_tail[ii] = storedAlready->second;
      }
    }
  }

#ifdef PRINT_TIMEDS
  double t_remap = timer();
  double time_remap = t_remap - t0;
  std::cout << "me[" << me << "]; Time GPU_remap: " << time_remap << std::endl;
#endif

  /// Get pointers to memory of device arrays 
  GraphElem* dev_currComm = gpu_graph.get_currComm();
  GraphElem* dev_ModlocalTarget = gpu_graph.get_ModlocalTarget();
  GraphWeight* dev_vDegree = gpu_graph.get_vDegree();
  GraphWeight* dev_clusterWeight = gpu_graph.get_clusterWeight();
  GraphElem* dev_edgeListIndexes = gpu_graph.get_edgeListIndexes();

  GraphElem* dev_ModcurrComm = gpu_graph.get_ModcurrComm();
  GraphElem* dev_localCinfo_size = gpu_graph.get_ModlocalCinfo_size();
  GraphWeight* dev_localCinfo_degree = gpu_graph.get_ModlocalCinfo_degree();
  GraphElem* dev_localCinfo_oComm = gpu_graph.get_ModlocalCinfo_oComm();
  GraphElem* dev_graph_edgeList_tail = gpu_graph.get_edgeList_tail();
  GraphWeight* dev_graph_edgeList_weight = gpu_graph.get_edgeList_weight();

  GraphElem* dev_unique_comm_array = gpu_graph.get_unique_comm_array();
  GraphWeight* dev_unique_weight_array = gpu_graph.get_unique_weight_array();

  gpu_graph.cpyVecTodev(currComm, dev_currComm);
  gpu_graph.cpyVecTodev(vDegree, dev_vDegree);
  gpu_graph.cpyVecTodev(clusterWeight, dev_clusterWeight);
#ifdef DEBUG_CUVITE
std::cout << "nv[" << nv << "]; size_edgeListIndexes["
     << g.edgeListIndexes.size() << "]" << std::endl;
#endif
  gpu_graph.cpyVecTodev(g.edgeListIndexes, dev_edgeListIndexes);

#ifdef DEBUG_CUVITE
std::cout << "nv[" << nv << "]; size_ModlocalCinfo["
     << size_ModlocalCinfo << "]; size_ModlocalCinfo_oComm[" 
     << size_ModlocalCinfo_oComm << "]" << std::endl;
#endif
  bool check_ModlocalCinfo_memory = gpu_graph.checkModCommMemory(size_ModlocalCinfo);
  assert(check_ModlocalCinfo_memory);
  bool check_ModlocalCinfoComm_memory = gpu_graph.checkModCommMemory(size_ModlocalCinfo_oComm);
  assert(check_ModlocalCinfoComm_memory);
  gpu_graph.cpyArrayTodev(temp_ModlocalCinfo_size, dev_localCinfo_size, size_ModlocalCinfo);
  gpu_graph.cpyArrayTodev(temp_ModlocalCinfo_degree, dev_localCinfo_degree, size_ModlocalCinfo);
  gpu_graph.cpyArrayTodev(temp_ModlocalCinfo_oComm, dev_localCinfo_oComm, size_ModlocalCinfo_oComm);

  gpu_graph.cpyArrayTodev(temp_graph_edgeList_tail, dev_graph_edgeList_tail, 
                        (GraphElem)g.edgeList.size());

  gpu_graph.cpyArrayTodev(temp_graph_edgeList_weight, dev_graph_edgeList_weight, 
                        (GraphElem)g.edgeList.size());

#ifdef DEBUG_CUVITE
std::cout << "nv[" << nv << "]; size_ModcurrComm["
     << ModcurrComm.size() << "]" << std::endl;
#endif
  bool check_ModcurrComm_memory = gpu_graph.checkModCommMemory(
       (GraphElem)ModcurrComm.size());
  assert(check_ModcurrComm_memory);
  gpu_graph.cpyVecTodev(ModcurrComm, dev_ModcurrComm);

  GraphElem* dev_GraphEdge_low = gpu_graph.get_GraphEdge_low();
  GraphElem* dev_GraphEdge_high = gpu_graph.get_GraphEdge_high();

  /// allocate device memory for filling in comm and weights
  GraphElem* dev_clmap_comm = gpu_graph.get_clmap_comm();
  GraphWeight* dev_clmap_weight = gpu_graph.get_clmap_weight();
  GraphElem clmapSize;

  GraphElem* dev_clmap_loc = gpu_graph.get_clmap_loc();

  GraphElem* dev_List_numEdges = gpu_graph.get_List_numEdges();

  GraphElem* dev_list_lt_ts = gpu_graph.get_dev_list_lt_ts();
  GraphElem* dev_list_lt_cs1 = gpu_graph.get_dev_list_lt_cs1();
  GraphElem* dev_list_lt_cs2 = gpu_graph.get_dev_list_lt_cs2();

#ifdef PRINT_TIMEDS
  cudaDeviceSynchronize();
  double t_dtrans = timer();
  double time_dtrans = t_dtrans - t0;
  std::cout << "me[" << me << "]; Time GPU_dtrans: " << time_dtrans << std::endl;
#endif

if(numIters == 1) 
{
  CUDA_SAFE(cudaMemset(dev_GraphEdge_low, 0, mem_size_GraphElem_nv));
  CUDA_SAFE(cudaMemset(dev_GraphEdge_high, 0, mem_size_GraphElem_nv));

  dim3 numBlocks01( (nv-1) / L_THREADBLOCK_SIZE + 1);
  dim3 Block_dim01(L_THREADBLOCK_SIZE);

  gpu_distExecuteLouvainIteration<<<numBlocks01,Block_dim01>>>(
                    nv, 
                    dev_edgeListIndexes,
                    dev_GraphEdge_low, dev_GraphEdge_high, 
                    me, base, bound);

  CUDA_SAFE(cudaMemset(dev_List_numEdges, 0, mem_size_GraphElem_nv));

  CUDA_SAFE(cudaMemset(dev_clmap_loc, 0, mem_size_GraphElem_nv));

  GraphElem* dev_NumClusters = gpu_graph.get_NumClusters();
  CUDA_SAFE(cudaMemset(dev_NumClusters, 0, sizeof(GraphElem)));

  GraphElem* dev_size_clmap = gpu_graph.get_size_clmap();
  CUDA_SAFE(cudaMemset(dev_size_clmap, 0, sizeof(GraphElem)));

  GraphElem* dev_size_lt_ts = gpu_graph.get_dev_size_lt_ts();
  CUDA_SAFE(cudaMemset(dev_size_lt_ts, 0, sizeof(GraphElem)));
  GraphElem* dev_size_lt_cs1 = gpu_graph.get_dev_size_lt_cs1();
  CUDA_SAFE(cudaMemset(dev_size_lt_cs1, 0, sizeof(GraphElem)));
  GraphElem* dev_size_lt_cs2 = gpu_graph.get_dev_size_lt_cs2();
  CUDA_SAFE(cudaMemset(dev_size_lt_cs2, 0, sizeof(GraphElem)));

  CUDA_SAFE(cudaMemset(dev_list_lt_ts, 0, mem_size_GraphElem_nv));
  CUDA_SAFE(cudaMemset(dev_list_lt_cs1, 0, mem_size_GraphElem_nv));
  CUDA_SAFE(cudaMemset(dev_list_lt_cs2, 0, mem_size_GraphElem_nv));

  dim3 numBlocks02( (nv-1) / MS_THREADBLOCK_SIZE + 1);
  dim3 Block_dim02(MS_THREADBLOCK_SIZE);

  count_size_clmap<PHY_WRP_SZ><<<numBlocks02,Block_dim02>>>(nv, dev_NumClusters, 
     dev_clmap_loc, dev_size_clmap, 
     dev_size_lt_ts, dev_list_lt_ts, 
     dev_size_lt_cs1, dev_list_lt_cs1, 
     dev_size_lt_cs2, dev_list_lt_cs2, 
     dev_GraphEdge_low, dev_GraphEdge_high, dev_List_numEdges);  

  /// copy to host number of clusters and size of cluster map memory
#ifdef DEBUG_CUVITE
  GraphElem NumClusters = 0;
  CUDA_SAFE(cudaMemcpy(&NumClusters, dev_NumClusters, 
     sizeof(GraphElem), cudaMemcpyDeviceToHost));
#endif
  CUDA_SAFE(cudaMemcpy(&clmapSize, dev_size_clmap, 
     sizeof(GraphElem), cudaMemcpyDeviceToHost));
  gpu_graph.set_clmapSize(clmapSize);

  dev_clmap_comm = gpu_graph.getDevMem_clmapComm(clmapSize);
  dev_clmap_weight = gpu_graph.getDevMem_clmapWeight(clmapSize);

  gpu_graph.set_size_lt_ts();
  gpu_graph.set_size_lt_cs1();
  gpu_graph.set_size_lt_cs2();

}

#ifdef PRINT_TIMEDS
  cudaDeviceSynchronize();
  double t_iter1 = timer();
  double time_iter1 = t_iter1 - t0;
  std::cout << "me[" << me << "]; Time GPU_iter1: " << time_iter1 << std::endl;
#endif

  GraphElem size_lt_ts = gpu_graph.get_size_lt_ts();
  GraphElem size_lt_cs1 = gpu_graph.get_size_lt_cs1();
  GraphElem size_lt_cs2 = gpu_graph.get_size_lt_cs2();

  clmapSize = gpu_graph.get_clmapSize();
  CUDA_SAFE(cudaMemset(dev_clmap_comm, 0, 
     clmapSize * sizeof(GraphElem)));
  CUDA_SAFE(cudaMemset(dev_clmap_weight, 0, 
     clmapSize * sizeof(GraphWeight)));

  GraphWeight* dev_selfLoopVec = gpu_graph.get_selfLoopVec();
  CUDA_SAFE(cudaMemset(dev_selfLoopVec, 0, mem_size_GraphWeight_nv));

  dim3 numBlocks03( (nv-1) / S_THREADBLOCK_SIZE + 1);
  dim3 Block_dim03(S_THREADBLOCK_SIZE);
  distBuildLocalMapCounter<PHY_WRP_SZ><<<numBlocks03,Block_dim03>>>(
              nv, dev_GraphEdge_low, dev_GraphEdge_high,
              dev_graph_edgeList_tail, 
              dev_graph_edgeList_weight,
              dev_List_numEdges, dev_selfLoopVec, 
              base); // , bound); 

  GraphElem* dev_uniq_clus_vec = gpu_graph.get_uniq_clus_vec();
  CUDA_SAFE(cudaMemset(dev_uniq_clus_vec, 0, mem_size_GraphElem_nv));
  GraphWeight* dev_counter = gpu_graph.get_counter();
  CUDA_SAFE(cudaMemset(dev_counter, 0, mem_size_GraphWeight_nv));

  const int num_streams = 2;
  cudaStream_t streams[num_streams];
  for(auto i_streams = 0; i_streams < num_streams; i_streams++) {
    CUDA_SAFE(cudaStreamCreate(&streams[i_streams]) );
  }

  dim3 numBlocks05( (nv-1) / S_THREADBLOCK_SIZE + 1);
  dim3 Block_dim05(S_THREADBLOCK_SIZE);
  distGetMaxIndex<PHY_WRP_SZ><<<numBlocks05,Block_dim05, 0>>>(
              nv, 
              dev_GraphEdge_low, // dev_GraphEdge_high,
              dev_graph_edgeList_tail,
              dev_graph_edgeList_weight,
              dev_ModcurrComm,
              dev_clmap_loc, 
              dev_clmap_comm, dev_clmap_weight, 
              dev_List_numEdges,
              dev_uniq_clus_vec, dev_counter,
              base); // , bound);

#ifdef PRINT_TIMEDS
  cudaDeviceSynchronize();
  double t_kernel11 = timer();
  double time_kernel11 = t_kernel11 - t0;
  std::cout << "me[" << me << "]; Time GPU_kernel11: " << time_kernel11 << std::endl;
#endif

#if 0
  distGetMaxIndex_large<PHY_WRP_SZ><<<numBlocks05,Block_dim05, 0, streams[1]>>>(
              nv, 
              dev_GraphEdge_low, dev_GraphEdge_high,
              dev_graph_edgeList_tail,
              dev_graph_edgeList_weight,
              dev_ModcurrComm,
              dev_clmap_loc, 
              dev_clmap_comm, dev_clmap_weight, 
              dev_List_numEdges,
              dev_uniq_clus_vec, dev_counter,
              base, bound);
#else
  if(size_lt_cs2 > 0) {
    dim3 numBlocks052(FINDING_UNIQCOMM_NUM_BLOCKS);
    dim3 Block_dim052(FINDING_UNIQCOMM_BLOCK_SIZE);
    GraphElem nv_chunk_size;
    nv_chunk_size = (size_lt_cs2 - 1) / FINDING_UNIQCOMM_NUM_BLOCKS + 1;
    assert(ModcurrComm.size() <= FINDING_UNIQCOMM_ARRAY_SIZE);
    distGetMaxIndex_large_new<PHY_WRP_SZ><<<numBlocks052,Block_dim052, 0, streams[0]>>>(
                me, numIters,
                nv, nv_chunk_size,
                size_lt_cs2, dev_list_lt_cs2,
                ModcurrComm.size(), // size_ModlocalCinfo,
                dev_unique_comm_array, 
                dev_unique_weight_array, 
                dev_GraphEdge_low, // dev_GraphEdge_high,
                dev_graph_edgeList_tail,
                dev_graph_edgeList_weight,
                dev_ModcurrComm,
                dev_clmap_loc, 
                dev_clmap_comm, dev_clmap_weight, 
                dev_List_numEdges,
                dev_uniq_clus_vec, dev_counter,
                base); // , bound);
  }
#endif
#ifdef PRINT_TIMEDS
  cudaDeviceSynchronize();
  double t_kernel1 = timer();
  double time_kernel1 = t_kernel1 - t0;
  std::cout << "me[" << me << "]; Time GPU_kernel1: " << time_kernel1 << std::endl;
#endif

  CUDA_SAFE(cudaMemcpy(dev_ModlocalTarget, dev_ModcurrComm, 
     sizeof(GraphElem)*ModcurrComm.size(), cudaMemcpyDeviceToDevice));

  dim3 numBlocks06( (nv-1) / S_THREADBLOCK_SIZE + 1);
  dim3 Block_dim06(S_THREADBLOCK_SIZE);
  computeMaxIndex<PHY_WRP_SZ><<<numBlocks06,Block_dim06, 0>>>(
              nv, 
              dev_currComm,
              dev_ModcurrComm,
              dev_localCinfo_size, 
              dev_localCinfo_degree, 
              dev_localCinfo_oComm, 
              dev_selfLoopVec, 
              dev_uniq_clus_vec, dev_counter,
              dev_clmap_loc, 
              dev_clmap_comm, dev_clmap_weight, 
              dev_vDegree,
              dev_ModlocalTarget,
              dev_clusterWeight,
              constantForSecondTerm,
              base, bound);

#ifdef PRINT_TIMEDS
  cudaDeviceSynchronize();
  double t_kernel21 = timer();
  double time_kernel21 = t_kernel21 - t0;
  std::cout << "me[" << me << "]; Time GPU_kernel21: " << time_kernel21 << std::endl;
#endif

  computeMaxIndex_large<PHY_WRP_SZ><<<numBlocks06,Block_dim06, 0, streams[0]>>>(
              nv, 
              dev_currComm,
              dev_ModcurrComm,
              dev_localCinfo_size, 
              dev_localCinfo_degree, 
              dev_localCinfo_oComm, 
              dev_selfLoopVec, 
              dev_uniq_clus_vec, dev_counter,
              dev_clmap_loc, 
              dev_clmap_comm, dev_clmap_weight, 
              dev_vDegree,
              dev_ModlocalTarget,
              dev_clusterWeight,
              constantForSecondTerm,
              base, bound);

#ifdef PRINT_TIMEDS
  cudaDeviceSynchronize();
  double t_kernels = timer();
  double time_kernels = t_kernels - t0;
  std::cout << "me[" << me << "]; Time GPU_kernels: " << time_kernels << std::endl;
#endif

  for(auto i_streams = 0; i_streams < num_streams; i_streams++) {
    cudaStreamSynchronize(streams[i_streams]);
  }

  for(auto i_streams = 0; i_streams < num_streams; i_streams++) {
    cudaStreamDestroy(streams[i_streams]);
  }

  /// Copy Targets to Host
  CUDA_SAFE(cudaMemcpy(&ModcurrComm[0], dev_ModlocalTarget, 
     (ModcurrComm.size()*sizeof(GraphElem)), cudaMemcpyDeviceToHost));
  /// Copy clusterWeight to Host
  CUDA_SAFE(cudaMemcpy(&clusterWeight[0],
     dev_clusterWeight, 
     (clusterWeight.size()*sizeof(GraphWeight)), cudaMemcpyDeviceToHost));

#ifdef PRINT_TIMEDS
  cudaDeviceSynchronize();
  double t_kd2h = timer();
  double time_kd2h = t_kd2h - t0;
  std::cout << "me[" << me << "]; Time GPU_kd2h: " << time_kd2h << std::endl;
#endif

  updateLocalTarget_gpu (
    nv, 
    currComm,
    targetComm,
    vDegree,
    remoteCupdate,
    ModcurrComm,
    localCupdate,
    base, bound, numIters); 

#ifdef PRINT_TIMEDS
  double t_locupd = timer();
  double time_locupd = t_locupd - t0;
  std::cout << "me[" << me << "]; Time GPU_locupd: " << time_locupd << std::endl;
#endif

  std::vector<GraphElem>().swap(ModcurrComm);

#ifdef PRINT_TIMEDS
  double t_all = timer();
  double time_all = t_all - t0;
  std::cout << "me[" << me << "]; Time GPU_all: " << time_all << std::endl;
#endif

#endif  // end of option to run hybrid or GPU-only code

  return 1;
}

