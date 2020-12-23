#ifndef __EDGE_H
#define __EDGE_H

#include <cstdint>

#include <vector>

#include <mpi.h>

#ifdef USE_32_BIT_GRAPH
typedef int32_t GraphElem;
typedef float GraphWeight;
const MPI_Datatype MPI_GRAPH_TYPE = MPI_INT32_T;
const MPI_Datatype MPI_WEIGHT_TYPE = MPI_FLOAT;
#else
typedef int64_t GraphElem;
typedef double GraphWeight;
const MPI_Datatype MPI_GRAPH_TYPE = MPI_INT64_T;
const MPI_Datatype MPI_WEIGHT_TYPE = MPI_DOUBLE;
#endif

struct Edge {
  GraphElem tail;
  GraphWeight weight;

  Edge();
  Edge(GraphElem t, GraphWeight w): 
        tail(t), weight(w) 
  {}
};

struct EdgeTuple
{
    GraphElem ij_[2];
    GraphWeight w_;

    EdgeTuple(GraphElem i, GraphElem j, GraphWeight w): 
        ij_{i, j}, w_(w)
    {}
    EdgeTuple(GraphElem i, GraphElem j): 
        ij_{i, j}, w_(1.0) 
    {}
    EdgeTuple(): 
        ij_{-1, -1}, w_(0.0)
    {}
};

typedef std::vector<GraphElem> EdgeIndexes;

inline Edge::Edge()
  : tail(-1), weight(0.0)
{
} // Edge

#endif // __EDGE_DECL_H
