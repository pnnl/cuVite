#ifndef __BUILD_NEXT_PHASE_H
#define __BUILD_NEXT_PHASE_H

#include <cassert>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>

#include <iostream>
#include <numeric>

#include <omp.h>

#include "edge.hpp"
#include "louvain.hpp"

typedef struct edgeInfo{
  GraphElem s;
  GraphElem t;
  GraphWeight w;
}EdgeInfo;

static MPI_Datatype edgeType;

#if defined(__CRAY_MIC_KNL) && defined(USE_AUTOHBW_MEMALLOC)
typedef std::vector<EdgeInfo, hbw::allocator<EdgeInfo> > EdgeVector;
typedef std::unordered_set<GraphElem, std::hash<GraphElem>, std::equal_to<GraphElem>, 
	hbw::allocator<GraphElem> > RemoteCommList;
typedef std::vector<RemoteCommList, hbw::allocator<RemoteCommList>> PartArray;
typedef std::map<GraphElem, GraphWeight, std::less<GraphElem>, 
	hbw::allocator< std::pair< const GraphElem, GraphWeight > > > NewEdge;
typedef std::unordered_map<GraphElem, NewEdge, std::hash<GraphElem>, std::equal_to<GraphElem>, 
	hbw::allocator< std::pair< const GraphElem, NewEdge > > > NewEdgesMap;
#else
typedef std::vector<EdgeInfo> EdgeVector;
typedef std::unordered_set<GraphElem> RemoteCommList;
typedef std::vector<RemoteCommList> PartArray;
typedef std::map<GraphElem, GraphWeight> NewEdge;
typedef std::unordered_map<GraphElem,NewEdge> NewEdgesMap;
#endif

void createEdgeMPIType();
void destroyEdgeMPIType();

static GraphElem distReNumber(int nprocs, ClusterLocalMap& lookUp, int me, 
        DistGraph &dg, const size_t &ssz, const size_t &rsz, 
        const std::vector<GraphElem> &ssizes, const std::vector<GraphElem> &rsizes, 
        const std::vector<GraphElem> &svdata, const std::vector<GraphElem> &rvdata,
        CommunityVector &cvect, VertexCommMap &remoteComm);

void fill_newEdgesMap(int me, NewEdgesMap &newEdgesMap, 
        DistGraph& dg, CommunityVector &cvect, VertexCommMap &remoteComm, 
        ClusterLocalMap &lookUp);

void send_newEdges(int me, int nprocs, DistGraph* &dg, GraphElem newGlobalNumVertices,
  NewEdgesMap& newEdgesMap);

void distbuildNextLevelGraph(int nprocs, int me, DistGraph*& dg, 
        const size_t &ssz, const size_t &rsz, const std::vector<GraphElem> &ssizes, 
        const std::vector<GraphElem> &rsizes, const std::vector<GraphElem> &svdata, 
        const std::vector<GraphElem> &rvdata, CommunityVector &cvect);
#endif
