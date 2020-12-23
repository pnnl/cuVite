#include "GpuGraph.cuh"

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

using namespace CuVite;

/// Allocate pinned memory on host
template <typename T>
T *GpuGraph::pinned_memory_ofSize(GraphElem size)
{
  T* hostP_vec;
  CUDA_SAFE( cudaMallocHost((void**)&hostP_vec, size*sizeof(T)) );
  // std::cout << "memory_size[" << size*sizeof(T) << "]" << std::endl;
  return hostP_vec;
}

/// Allocate device memory
template <typename T>
T *GpuGraph::device_memory_ofSize(GraphElem size)
{
  T* dev_vec;
  CUDA_SAFE( cudaMalloc((void**)&dev_vec, size*sizeof(T)) );
  // std::cout << "memory_size[" << size*sizeof(T) << "]" << std::endl;
  return dev_vec;
}

void GpuGraph::devMemAllocate() 
{
  dev_currComm_ = device_memory_ofSize<GraphElem>(nv_);
  dev_ModlocalTarget_ = device_memory_ofSize<GraphElem>(nm_);
  dev_vDegree_ = device_memory_ofSize<GraphWeight>(nv_);
  dev_clusterWeight_ = device_memory_ofSize<GraphWeight>(nv_);
  dev_edgeListIndexes_ = device_memory_ofSize<GraphElem>(nv_+1);

  dev_GraphEdge_low_ = device_memory_ofSize<GraphElem>(nv_);
  dev_GraphEdge_high_ = device_memory_ofSize<GraphElem>(nv_);
  dev_selfLoopVec_ = device_memory_ofSize<GraphWeight>(nv_);
  dev_List_numEdges_ = device_memory_ofSize<GraphElem>(nv_);
  dev_clmap_loc_ = device_memory_ofSize<GraphElem>(nv_);
  dev_NumClusters_ = device_memory_ofSize<GraphElem>(sizeof(GraphElem));
  dev_size_clmap_ = device_memory_ofSize<GraphElem>(sizeof(GraphElem));
  dev_uniq_clus_vec_ = device_memory_ofSize<GraphElem>(nv_);
  dev_counter_ = device_memory_ofSize<GraphWeight>(nv_);

  dev_size_lt_ts_ = device_memory_ofSize<GraphElem>(1);
  dev_list_lt_ts_ = device_memory_ofSize<GraphElem>(nv_);
  dev_size_lt_cs1_ = device_memory_ofSize<GraphElem>(1);
  dev_list_lt_cs1_ = device_memory_ofSize<GraphElem>(nv_);
  dev_size_lt_cs2_ = device_memory_ofSize<GraphElem>(1);
  dev_list_lt_cs2_ = device_memory_ofSize<GraphElem>(nv_);

  dev_unique_comm_array_ = device_memory_ofSize<GraphElem>(FINDING_UNIQCOMM_NUM_BLOCKS
                            * FINDING_UNIQCOMM_ARRAY_SIZE);
  dev_unique_weight_array_ = device_memory_ofSize<GraphWeight>(FINDING_UNIQCOMM_NUM_BLOCKS
                            * FINDING_UNIQCOMM_ARRAY_SIZE);

  dev_clmap_comm_alloc_ = false;
  dev_clmap_weight_alloc_ = false;

  dev_sum_weight_ = device_memory_ofSize<GraphWeight>(1);;
  dev_sum_degree_ = device_memory_ofSize<GraphWeight>(1);
}

template <typename T>
T* GpuGraph::DevMemAllocate(T*& device_buf, size_t size)
{
  CUDA_SAFE( cudaMalloc((void**)&device_buf, size*sizeof(T)) );
  CUDA_SAFE(cudaMemset(device_buf, 0, size*sizeof(T)));
}

void GpuGraph::pinnedAndDevMemAllocate() 
{
  /// currently we use std::vector<GraphElem> for ModcurrComm on host
  // pinned_ModcurrComm_ = pinned_memory_ofSize<GraphElem>(nm_);
  dev_ModcurrComm_ = device_memory_ofSize<GraphElem>(nm_); 

  pinned_ModlocalCinfo_size_ = pinned_memory_ofSize<GraphElem>(nm_);
  dev_ModlocalCinfo_size_ = device_memory_ofSize<GraphElem>(nm_); 

  pinned_ModlocalCinfo_degree_ = pinned_memory_ofSize<GraphWeight>(nm_);
  dev_ModlocalCinfo_degree_ = device_memory_ofSize<GraphWeight>(nm_);

  pinned_ModlocalCinfo_oComm_ = pinned_memory_ofSize<GraphElem>(nm_);
  dev_ModlocalCinfo_oComm_= device_memory_ofSize<GraphElem>(nm_); 


  pinned_edgeList_tail_ = pinned_memory_ofSize<GraphElem>(ne_);
  dev_edgeList_tail_ = device_memory_ofSize<GraphElem>(ne_); 

  pinned_edgeList_weight_ = pinned_memory_ofSize<GraphWeight>(ne_);
  dev_edgeList_weight_ = device_memory_ofSize<GraphWeight>(ne_);
}

GraphElem* GpuGraph::getDevMem_clmapComm(GraphElem size)
{
  // if(dev_clmap_comm_ == nullptr) 
  if(dev_clmap_comm_alloc_ == false) {
    CUDA_SAFE( cudaMalloc((void**)&dev_clmap_comm_, size*sizeof(GraphElem)) );
    dev_clmap_comm_alloc_ = true;
    return dev_clmap_comm_;
  } else {
    return dev_clmap_comm_;
    // std::cerr << "Failed to allocate memory for dev_clmap_comm_ " << std::endl;
  }
}

GraphWeight* GpuGraph::getDevMem_clmapWeight(GraphElem size)
{
  // if(dev_clmap_weight_ == nullptr)
  if(dev_clmap_weight_alloc_ == false) {
    CUDA_SAFE( cudaMalloc((void**)&dev_clmap_weight_, size*sizeof(GraphElem)) );
    dev_clmap_weight_alloc_ = true;
    return dev_clmap_weight_;
  } else {
    return dev_clmap_weight_;
    // std::cerr << "Failed to allocate memory for dev_clmap_weight_ " << std::endl;
  }
}

  /// Free memory
template <typename T>
void GpuGraph::DevMemFree(T*& device_buf) {
  if(device_buf != nullptr) {
  cudaError_t err = cudaFree(device_buf);
  if (err != cudaSuccess) {
    std::cerr << "Failed to free device memory for dev_vec_ " << std::endl;
  }
  device_buf = nullptr;
  }
}

void GpuGraph::DevMemCopy(GraphElem* host_buf, GraphElem* dev_buf, GraphElem size) {
  if(dev_buf!= nullptr) {
    CUDA_SAFE(cudaMemcpy(dev_buf, host_buf, sizeof(GraphElem)*size, cudaMemcpyHostToDevice));
  } else {
    std::cerr << "Failed to to copy to device memory " << std::endl;
  }
}

void GpuGraph::DevMemCopy(GraphWeight* host_buf, GraphWeight* dev_buf, GraphElem size) {
  if(dev_buf!= nullptr) {
    CUDA_SAFE(cudaMemcpy(dev_buf, host_buf, sizeof(GraphWeight)*size, cudaMemcpyHostToDevice));
  } else {
    std::cerr << "Failed to to copy to device memory " << std::endl;
  }
}

void GpuGraph::DevMemCopy(const std::vector<GraphElem>& host_buf, GraphElem* dev_buf) {
  if(dev_buf!= nullptr) {
    CUDA_SAFE(cudaMemcpy(dev_buf, host_buf.data(), sizeof(GraphElem)*host_buf.size(), cudaMemcpyHostToDevice));
  } else {
    std::cerr << "Failed to to copy to device memory " << std::endl;
  }
}

void GpuGraph::DevMemCopy(const std::vector<GraphWeight>& host_buf, GraphWeight* dev_buf) {
  if(dev_buf!= nullptr) {
    CUDA_SAFE(cudaMemcpy(dev_buf, host_buf.data(), sizeof(GraphWeight)*host_buf.size(), cudaMemcpyHostToDevice));
  } else {
    std::cerr << "Failed to to copy to device memory " << std::endl;
  }
}

void GpuGraph::freeDev_ModcurrComm()
{
  DevMemFree(dev_ModcurrComm_);    
}

GraphElem* GpuGraph::get_currComm() {
  if(dev_currComm_!= nullptr) {
    return dev_currComm_;
  } else {
    std::cerr << "Failed to provide device memory for dev_currComm_" << std::endl;
    return nullptr;
  }
}

GraphElem* GpuGraph::get_ModlocalTarget() {
  if(dev_ModlocalTarget_!= nullptr) {
    return dev_ModlocalTarget_;
  } else {
    std::cerr << "Failed to provide device memory for dev_ModlocalTarget_" << std::endl;
    return nullptr;
  }
}

GraphWeight* GpuGraph::get_vDegree() {
  if(dev_vDegree_ != nullptr) {
    return dev_vDegree_;
  } else {
    std::cerr << "Failed to provide device memory for dev_vDegree_ " << std::endl;
    return nullptr;
  }
}

GraphWeight* GpuGraph::get_clusterWeight() {
  if(dev_clusterWeight_ != nullptr) {
    return dev_clusterWeight_;
  } else {
    std::cerr << "Failed to provide device memory for dev_clusterWeight_ " << std::endl;
    return nullptr;
  }
}

GraphElem* GpuGraph::get_edgeListIndexes() {
  if(dev_edgeListIndexes_ != nullptr) {
    return dev_edgeListIndexes_;
  } else {
    std::cerr << "Failed to provide device memory for dev_edgeListIndexes_ " << std::endl;
    return nullptr;
  }
}

GraphElem* GpuGraph::get_GraphEdge_low() {
  if(dev_GraphEdge_low_!= nullptr) {
    return dev_GraphEdge_low_;
  } else {
    std::cerr << "Failed to provide device memory for dev_GraphEdge_low_" << std::endl;
    return nullptr;
  }
}

GraphElem* GpuGraph::get_GraphEdge_high() {
  if(dev_GraphEdge_high_!= nullptr) {
    return dev_GraphEdge_high_;
  } else {
    std::cerr << "Failed to provide device memory for dev_GraphEdge_high_" << std::endl;
    return nullptr;
  }
}

GraphWeight* GpuGraph::get_selfLoopVec() {
  if(dev_selfLoopVec_ != nullptr) {
    return dev_selfLoopVec_;
  } else {
    std::cerr << "Failed to provide device memory for dev_selfLoopVec_" << std::endl;
    return nullptr;
  }
}

GraphElem* GpuGraph::get_List_numEdges() {
  if(dev_List_numEdges_ != nullptr) {
    return dev_List_numEdges_;
  } else {
    std::cerr << "Failed to provide device memory for dev_List_numEdges_" << std::endl;
    return nullptr;
  }
}

GraphElem* GpuGraph::get_clmap_loc() {
  if(dev_clmap_loc_ != nullptr) {
    return dev_clmap_loc_;
  } else {
    std::cerr << "Failed to provide device memory for dev_clmap_loc_" << std::endl;
    return nullptr;
  }
}

GraphElem* GpuGraph::get_NumClusters() {
  if(dev_NumClusters_!= nullptr) {
    return dev_NumClusters_;
  } else {
    std::cerr << "Failed to provide device memory for dev_NumClusters_" << std::endl;
    return nullptr;
  }
}

GraphElem* GpuGraph::get_size_clmap() {
  if(dev_size_clmap_!= nullptr) {
    return dev_size_clmap_;
  } else {
    std::cerr << "Failed to provide device memory for dev_size_clmap_" << std::endl;
    return nullptr;
  }
}

GraphElem* GpuGraph::get_uniq_clus_vec() {
  if(dev_uniq_clus_vec_!= nullptr) {
    return dev_uniq_clus_vec_;
  } else {
    std::cerr << "Failed to provide device memory for dev_uniq_clus_vec_" << std::endl;
    return nullptr;
  }
}

GraphWeight* GpuGraph::get_counter() {
  if(dev_counter_!= nullptr) {
    return dev_counter_;
  } else {
    std::cerr << "Failed to provide device memory for dev_counter_" << std::endl;
    return nullptr;
  }
}

GraphElem* GpuGraph::get_dev_size_lt_ts() {
  if(dev_size_lt_ts_!= nullptr) {
    return dev_size_lt_ts_;
  } else {
    std::cerr << "Failed to provide device memory for dev_size_lt_ts_" << std::endl;
    return nullptr;
  }
}

GraphElem* GpuGraph::get_dev_list_lt_ts() {
  if(dev_list_lt_ts_!= nullptr) {
    return dev_list_lt_ts_;
  } else {
    std::cerr << "Failed to provide device memory for dev_list_lt_ts_" << std::endl;
    return nullptr;
  }
}

GraphElem* GpuGraph::get_dev_size_lt_cs1() {
  if(dev_size_lt_cs1_!= nullptr) {
    return dev_size_lt_cs1_;
  } else {
    std::cerr << "Failed to provide device memory for dev_size_lt_cs1_" << std::endl;
    return nullptr;
  }
}

GraphElem* GpuGraph::get_dev_list_lt_cs1() {
  if(dev_list_lt_cs1_!= nullptr) {
    return dev_list_lt_cs1_;
  } else {
    std::cerr << "Failed to provide device memory for dev_list_lt_cs1_" << std::endl;
    return nullptr;
  }
}

GraphElem* GpuGraph::get_dev_size_lt_cs2() {
  if(dev_size_lt_cs2_!= nullptr) {
    return dev_size_lt_cs2_;
  } else {
    std::cerr << "Failed to provide device memory for dev_size_lt_cs2_" << std::endl;
    return nullptr;
  }
}

GraphElem* GpuGraph::get_dev_list_lt_cs2() {
  if(dev_list_lt_cs2_!= nullptr) {
    return dev_list_lt_cs2_;
  } else {
    std::cerr << "Failed to provide device memory for dev_list_lt_cs2_" << std::endl;
    return nullptr;
  }
}

GraphElem GpuGraph::get_clmapSize() {
    return clmapSize_;
}

void GpuGraph::set_clmapSize(GraphElem size) {
    clmapSize_ = size;
}

GraphElem* GpuGraph::get_clmap_comm() {
  // if(dev_clmap_comm_ != nullptr) {
    return dev_clmap_comm_;
  // } else {
  //   std::cerr << "Failed to provide device memory for dev_clmap_comm_" << std::endl;
  //   return nullptr;
  // }
}

GraphWeight* GpuGraph::get_clmap_weight() {
  // if(dev_clmap_weight_ != nullptr) {
    return dev_clmap_weight_;
  // } else {
  //   std::cerr << "Failed to provide device memory for dev_clmap_weight_" << std::endl;
  //   return nullptr;
  // }
}

GraphElem GpuGraph::get_size_lt_ts() {
    return size_lt_ts_;
}

GraphElem GpuGraph::get_size_lt_cs1() {
    return size_lt_cs1_;
}

GraphElem GpuGraph::get_size_lt_cs2() {
    return size_lt_cs2_;
}

void GpuGraph::set_size_lt_ts() {
  CUDA_SAFE(cudaMemcpy(&size_lt_ts_, dev_size_lt_ts_,
     sizeof(GraphElem), cudaMemcpyDeviceToHost));
}

void GpuGraph::set_size_lt_cs1() {
  CUDA_SAFE(cudaMemcpy(&size_lt_cs1_, dev_size_lt_cs1_,
     sizeof(GraphElem), cudaMemcpyDeviceToHost));
}

void GpuGraph::set_size_lt_cs2() {
  CUDA_SAFE(cudaMemcpy(&size_lt_cs2_, dev_size_lt_cs2_,
     sizeof(GraphElem), cudaMemcpyDeviceToHost));
}

GraphElem* GpuGraph::get_ModcurrComm() {
  if(dev_ModcurrComm_ != nullptr) {
    return dev_ModcurrComm_;
  } else {
    std::cerr << "Failed to provide device memory for dev_ModcurrComm_ " << std::endl;
    return nullptr;
  }
}

GraphElem* GpuGraph::getPinned_ModlocalCinfo_size() {
  if(pinned_ModlocalCinfo_size_ != nullptr) {
    return pinned_ModlocalCinfo_size_;
  } else {
    std::cerr << "Failed to provide memory for pinned_ModlocalCinfo_size_ " << std::endl;
    return nullptr;
  }
}

GraphElem* GpuGraph::get_ModlocalCinfo_size() {
  if(dev_ModlocalCinfo_size_ != nullptr) {
    return dev_ModlocalCinfo_size_;
  } else {
    std::cerr << "Failed to provide device memory for dev_ModlocalCinfo_size_ " << std::endl;
    return nullptr;
  }
}

GraphWeight* GpuGraph::getPinned_ModlocalCinfo_degree() {
  if(pinned_ModlocalCinfo_degree_ != nullptr) {
    return pinned_ModlocalCinfo_degree_;
  } else {
    std::cerr << "Failed to provide memory for pinned_ModlocalCinfo_degree_ " << std::endl;
    return nullptr;
  }
}

GraphWeight* GpuGraph::get_ModlocalCinfo_degree() {
  if(dev_ModlocalCinfo_degree_ != nullptr) {
    return dev_ModlocalCinfo_degree_;
  } else {
    std::cerr << "Failed to provide device memory for dev_ModlocalCinfo_degree_ " << std::endl;
    return nullptr;
  }
}

GraphElem* GpuGraph::getPinned_ModlocalCinfo_oComm() {
  if(pinned_ModlocalCinfo_oComm_ != nullptr) {
    return pinned_ModlocalCinfo_oComm_;
  } else {
    std::cerr << "Failed to provide memory for pinned_ModlocalCinfo_oComm_ " << std::endl;
    return nullptr;
  }
}

GraphElem* GpuGraph::get_ModlocalCinfo_oComm() {
  if(dev_ModlocalCinfo_oComm_ != nullptr) {
    return dev_ModlocalCinfo_oComm_;
  } else {
    std::cerr << "Failed to provide device memory for dev_ModlocalCinfo_oComm_ " << std::endl;
    return nullptr;
  }
}

GraphElem* GpuGraph::getPinned_edgeList_tail() {
  if(pinned_edgeList_tail_ != nullptr) {
    return pinned_edgeList_tail_;
  } else {
    std::cerr << "Failed to provide memory for pinned_edgeList_tail_ " << std::endl;
    return nullptr;
  }
}

GraphElem* GpuGraph::get_edgeList_tail() {
  if(dev_edgeList_tail_ != nullptr) {
    return dev_edgeList_tail_;
  } else {
    std::cerr << "Failed to provide device memory for dev_edgeList_tail_ " << std::endl;
    return nullptr;
  }
}

GraphWeight* GpuGraph::getPinned_edgeList_weight() {
  if(pinned_edgeList_weight_ != nullptr) {
    return pinned_edgeList_weight_;
  } else {
    std::cerr << "Failed to provide memory for pinned_edgeList_weight_ " << std::endl;
    return nullptr;
  }
}

GraphWeight* GpuGraph::get_edgeList_weight() {
  if(dev_edgeList_weight_ != nullptr) {
    return dev_edgeList_weight_;
  } else {
    std::cerr << "Failed to provide device memory for dev_edgeList_weight_ " << std::endl;
    return nullptr;
  }
}

GraphElem* GpuGraph::get_unique_comm_array() {
  if(dev_unique_comm_array_ != nullptr) {
    return dev_unique_comm_array_;
  } else {
    std::cerr << "Failed to provide device memory for dev_unique_comm_array_" << std::endl;
    return nullptr;
  }
}

GraphWeight* GpuGraph::get_unique_weight_array() {
  if(dev_unique_weight_array_ != nullptr) {
    return dev_unique_weight_array_;
  } else {
    std::cerr << "Failed to provide device memory for dev_unique_weight_array_" << std::endl;
    return nullptr;
  }
}

GraphWeight* GpuGraph::get_sum_weight() {
  if(dev_sum_weight_ != nullptr) {
    return dev_sum_weight_;
  } else {
    std::cerr << "Failed to provide device memory for dev_sum_weight_" << std::endl;
    return nullptr;
  }
}

GraphWeight* GpuGraph::get_sum_degree() {
  if(dev_sum_degree_ != nullptr) {
    return dev_sum_degree_;
  } else {
    std::cerr << "Failed to provide device memory for dev_sum_degree_" << std::endl;
    return nullptr;
  }
}

