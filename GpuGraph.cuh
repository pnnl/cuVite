#ifndef __GPUGRAPH_CUH__
#define __GPUGRAPH_CUH__

#include <iostream>
#include <vector>

#include "edge.hpp"
#include "louvain_cuda_constants.cuh"

// =====================================================================
// Class GpuGraph
// 
// * This class includes data structure for graph on GPU-device for 
//   all graph parameters 
// * All the graph parameters become some of the the class member variables
// * The working memory for all the graph parameters is allocated through 
//   the memory manager
//
// =====================================================================

#define ASSERT(x)    if (!(x))  { printf("Assert Failed! <%s:%d>\n", __FILE__, __LINE__);  }

#define CUDA_SAFE(x) if ( cudaSuccess != (x) ) { printf("CUDA CALL FAILED AT %d\n", __LINE__ ); exit(1);}
#define CUDA_SAFE_MALLOC(DP, SIZE)  (cudaMalloc((void**)&DP, SIZE))

// #define ONE_RANK_RUN

namespace CuVite
{

class GpuGraph
{

  /* set nm_ = max(5*nv_, ng_)
   * This will take care of graph that has ng_ = 0
   * in the first phase */
public:
  GpuGraph(GraphElem nv, GraphElem ne, GraphElem ng): 
           nv_(nv), ne_(ne), ng_(ng),
#ifndef ONE_RANK_RUN
           nm_(std::max(MODCURRCOMM_SIZE_FACTOR*nv, (nv+ng)))
#else
           nm_(3/2*nv)
#endif
  {}

  // GpuGraph(){}

  /// Allocate pinned memory on host
  template <typename T>
  T *pinned_memory_ofSize(GraphElem size);

  /// Allocate device memory
  template <typename T>
  T * device_memory_ofSize(GraphElem size);

  void devMemAllocate();

  template <typename T>
  T* DevMemAllocate(T*& device_buf, size_t size);

  void pinnedAndDevMemAllocate();

  GraphElem* getDevMem_clmapComm(GraphElem size);
  GraphWeight* getDevMem_clmapWeight(GraphElem size);

  inline  GraphElem getRefVecSize()
  {
    // GraphElem size = m_currComm_.getRefVecSize();
    // std::cout << "Size of RefVector[" << size << "]" << std::endl;
    return nv_;
  }

  template <typename T>
  void DevMemFree(T*& device_buf);
 
  void DevMemCopy(GraphElem* host_buf, GraphElem* dev_buf, GraphElem size);
  void DevMemCopy(GraphWeight* host_buf, GraphWeight* dev_buf, GraphElem size);
  void DevMemCopy(const std::vector<GraphElem>& host_buf, GraphElem* dev_buf);
  void DevMemCopy(const std::vector<GraphWeight>& host_buf, GraphWeight* dev_buf);
  void freeDev_ModcurrComm();

  template <typename T>
  void cpyArrayTodev(T* host_buf, T* dev_buf, GraphElem size) {
   DevMemCopy(host_buf, dev_buf, size);
  }

  template <typename T>
  void cpyVecTodev(std::vector<T>& host_buf, T* dev_buf) {
   DevMemCopy(host_buf, dev_buf);
  }

  template <typename T>
  void cpyVecTodev(const std::vector<T>& host_buf, T* dev_buf) {
   DevMemCopy(host_buf, dev_buf);
  }

  bool checkModCommMemory(const GraphElem size_ModcurrComm) {
    bool check_passed = true;
    if(size_ModcurrComm > nm_) {
      check_passed = false;
      std::cout << "Insufficient memory \n"
                << "Increase the size of MODCURRCOMM_SIZE_FACTOR" << std::endl;
    }
    return check_passed;
  }

  // void cpyTodev_edgeList_tail(GraphElem dev_buf, std::vector<GraphElem>& host_buf);

  GraphElem* get_currComm();
  GraphElem* get_ModlocalTarget();
  GraphWeight* get_vDegree();
  GraphWeight* get_clusterWeight();
  GraphElem* get_edgeListIndexes();

  GraphElem* get_GraphEdge_low();
  GraphElem* get_GraphEdge_high();
  GraphWeight* get_selfLoopVec();
  GraphElem* get_List_numEdges();
  GraphElem* get_clmap_loc();
  GraphElem* get_NumClusters();
  GraphElem* get_size_clmap();
  GraphElem* get_uniq_clus_vec();
  GraphWeight* get_counter();

  GraphElem* get_dev_size_lt_ts();
  GraphElem* get_dev_list_lt_ts();
  GraphElem* get_dev_size_lt_cs1();
  GraphElem* get_dev_list_lt_cs1();
  GraphElem* get_dev_size_lt_cs2();
  GraphElem* get_dev_list_lt_cs2();

  GraphElem get_clmapSize();
  void set_clmapSize(GraphElem size);
  GraphElem* get_clmap_comm();
  GraphWeight* get_clmap_weight();

  GraphElem get_size_lt_ts();
  GraphElem get_size_lt_cs1();
  GraphElem get_size_lt_cs2();
  void set_size_lt_ts();
  void set_size_lt_cs1();
  void set_size_lt_cs2();

  GraphElem* get_ModcurrComm();
  GraphElem* getPinned_ModlocalCinfo_size();
  GraphElem* get_ModlocalCinfo_size();
  GraphWeight* getPinned_ModlocalCinfo_degree();
  GraphWeight* get_ModlocalCinfo_degree();
  GraphElem* getPinned_ModlocalCinfo_oComm();
  GraphElem* get_ModlocalCinfo_oComm();
  GraphElem* getPinned_edgeList_tail();
  GraphElem* get_edgeList_tail();
  GraphWeight* getPinned_edgeList_weight();
  GraphWeight* get_edgeList_weight();

  GraphElem* get_unique_comm_array();
  GraphWeight* get_unique_weight_array();

  GraphWeight* get_sum_weight();
  GraphWeight* get_sum_degree();

private:
  GraphElem nv_; 
  GraphElem ne_; 
  GraphElem ng_; 
  GraphElem nm_; // Ideally, nm_ = nv_ + ng_

  /// Device arrays corresponding to Vite Vectors
  GraphElem* dev_currComm_;  // implemented
  GraphElem* dev_ModlocalTarget_;  // implemented
  GraphWeight* dev_vDegree_;  // implemented
  GraphWeight* dev_clusterWeight_;  // implemented
  GraphElem* dev_edgeListIndexes_;  // implemented

  /// Device arrays for GPU kernels only 
  GraphElem* dev_GraphEdge_low_;  // implemented
  GraphElem* dev_GraphEdge_high_;  // implemented
  GraphWeight* dev_selfLoopVec_;  // implemented
  GraphElem* dev_List_numEdges_;  // implemented
  GraphElem* dev_clmap_loc_;  // implemented
  GraphElem* dev_NumClusters_;  // implemented
  GraphElem* dev_size_clmap_;  // implemented
  GraphElem* dev_uniq_clus_vec_;  // implemented
  GraphWeight* dev_counter_;  // implemented

  GraphElem size_lt_ts_;
  GraphElem* dev_size_lt_ts_;
  GraphElem* dev_list_lt_ts_;
  GraphElem size_lt_cs1_;
  GraphElem* dev_size_lt_cs1_;
  GraphElem* dev_list_lt_cs1_;
  GraphElem size_lt_cs2_;
  GraphElem* dev_size_lt_cs2_;
  GraphElem* dev_list_lt_cs2_;

  /// Device arrays computed only once in every phase. 
  GraphElem clmapSize_;
  bool dev_clmap_comm_alloc_;
  bool dev_clmap_weight_alloc_;
  GraphElem* dev_clmap_comm_;
  GraphWeight* dev_clmap_weight_;

  /// Arrays on pinned memory
  GraphElem* pinned_ModcurrComm_;
  GraphElem* pinned_ModlocalCinfo_size_;
  GraphWeight* pinned_ModlocalCinfo_degree_;
  GraphElem* pinned_ModlocalCinfo_oComm_;
  GraphElem* pinned_edgeList_tail_;
  GraphWeight* pinned_edgeList_weight_;

  /// Device arrays corresponding to arrays on pinned memory
  GraphElem* dev_ModcurrComm_;  // implemented on dev
  GraphElem* dev_ModlocalCinfo_size_;
  GraphWeight* dev_ModlocalCinfo_degree_;
  GraphElem* dev_ModlocalCinfo_oComm_;
  GraphElem* dev_edgeList_tail_;  //implemented on dev
  GraphWeight* dev_edgeList_weight_;  // implemented on dev

  GraphElem* dev_unique_comm_array_;  // working array to find unique comm
  GraphWeight* dev_unique_weight_array_;  // working array to find unique comm-weights

  GraphWeight* dev_sum_weight_;
  GraphWeight* dev_sum_degree_;
};

}  // namespace CuVite
#endif /* __GPUGRAPH_CUH__ */
