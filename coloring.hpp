#ifndef __COLORING_H
#define __COLORING_H

#include <algorithm>
#include <fstream>
#include <functional>
#include <iostream>
#include <list>
#include <numeric>
#include <vector>
#include <unordered_set>
#include <map>

#include <mpi.h>
#include <omp.h>

#include "graph.hpp"
#include "distgraph.hpp"

#ifdef USE_32_BIT_GRAPH
typedef int32_t ColorElem;
#else
typedef int64_t ColorElem;
#endif

const int ColoringSizeTag = 6;
const int ColoringDataTag = 7;

typedef std::unordered_set<GraphElem> ColoredVertexSet; 
typedef std::vector<ColorElem> ColorVector;

ColorElem distColoringMultiHashMinMax(const int me, const int nprocs, const DistGraph &dg, ColorVector &vertexColor, const ColorElem nHash, const int target_percent, const bool singleIteration);

static unsigned int hash(unsigned int a, unsigned int seed);

void distColoringIteration(const int me, const DistGraph &dg, ColorVector &vertexColor, ColoredVertexSet &remoteColoredVertices, const ColorElem nHash, const ColorElem nextColor, const unsigned int seed);

void setUpGhostVertices(const int me, const int nprocs, const DistGraph &dg, std::vector<GraphElem> &ghostVertices, std::vector<GraphElem> &ghostSizes);

void sendColoredRemoteVertices(const int me, const int nprocs, const DistGraph &dg, ColoredVertexSet &remoteColoredVertices, const ColorVector &vertexColor, const std::vector<GraphElem> &ghostVertices, const std::vector<GraphElem> &ghostSizes);

GraphElem countUnassigned(const ColorVector &vertexColor);

GraphElem distCheckColoring(const int me, const int nprocs, const DistGraph &dg, const ColorVector &vertexColor, const std::vector<GraphElem> &ghostVertices, const std::vector<GraphElem> &ghostSizes);	

#endif 
