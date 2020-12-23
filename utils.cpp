#include <sys/resource.h>
#include <sys/time.h>

#include <cstdlib>
#include <iostream>
#include <numeric>

#include "utils.hpp"

double mytimer(void)
{
  static long int start = 0L, startu;

  const double million = 1000000.0;

  timeval tp;

  if (start == 0L) {
    gettimeofday(&tp, NULL);

    start = tp.tv_sec;
    startu = tp.tv_usec;
  }

  gettimeofday(&tp, NULL);

  return (static_cast<double>(tp.tv_sec - start) + (static_cast<double>(tp.tv_usec - startu) /
						    million));
}

// Random number generator from B. Stroustrup: 
// http://www.stroustrup.com/C++11FAQ.html#std-random
GraphWeight genRandom(GraphWeight low, GraphWeight high)
{
    static std::default_random_engine re {};
    using Dist = std::uniform_real_distribution<GraphWeight>;
    static Dist uid {};
    return uid(re, Dist::param_type{low,high});
}

void processGraphData(Graph &g, std::vector<GraphElem> &edgeCount,
		      std::vector<GraphElemTuple> &edgeList,
		      const GraphElem nv, const GraphElem ne)
{
  std::vector<GraphElem> ecTmp(nv + 1);

  std::partial_sum(edgeCount.begin(), edgeCount.end(), ecTmp.begin());
  edgeCount = ecTmp;

  g.setEdgeStartForVertex(0, 0);

  for (GraphElem i = 0; i < nv; i++)
    g.setEdgeStartForVertex(i + 1, edgeCount[i + 1]);

  edgeCount.clear();

  auto ecmp = [] (GraphElemTuple const& e0, GraphElemTuple const& e1)
  { return ((e0.i_ < e1.i_) || ((e0.i_ == e1.i_) && (e0.j_ < e1.j_))); };
  
  if (!std::is_sorted(edgeList.begin(), edgeList.end(), ecmp)) {
    std::cout << "Edge list is not sorted" << std::endl;
    std::sort(edgeList.begin(), edgeList.end(), ecmp);
  }
  else
    std::cout << "Edge list is sorted!" << std::endl;

  GraphElem ePos = 0;
  for (GraphElem i = 0; i < nv; i++) {
    GraphElem e0, e1;

    g.getEdgeRangeForVertex(i, e0, e1);
    if ((i % 100000) == 0)
      std::cout << "Processing edges for vertex: " << i << ", range(" << e0 << ", " << e1 <<
	")" << std::endl;

    for (GraphElem j = e0; j < e1; j++) {
      Edge &edge = g.getEdge(j);

      assert(ePos == j);
      assert(i == edgeList[ePos].i_);
      edge.tail = edgeList[ePos].j_;
      edge.weight = edgeList[ePos].w_;

      ePos++;
    }
  }
} // processGraphData


