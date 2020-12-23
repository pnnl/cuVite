#ifndef __CLUSTER_COMPARE_H
#define __CLUSTER_COMPARE_H

#include "louvain.hpp"

void compare_communities(std::vector<GraphElem> const& C1, std::vector<GraphElem> const& C2);
double compute_gini_coeff(GraphElem *colorSize, int numColors);

#endif
