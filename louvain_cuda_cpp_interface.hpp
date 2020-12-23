#ifndef __LOUVAIN_CUDA_CPP_INTERFACE_H
#define __LOUVAIN_CUDA_CPP_INTERFACE_H

std::vector<GraphElem> extract_keys_CommMap(std::map<GraphElem, Comm>
    const& input_map) {
  std::vector<GraphElem> ret_val;
  for (auto const& element : input_map) {
    ret_val.push_back(element.first);
  }
  return ret_val;
}

std::vector<GraphElem> extract_value_CommMap_size(std::map<GraphElem, Comm>
    const& input_map) {
  std::vector<GraphElem> ret_val;
  for (auto const& element : input_map) {
    ret_val.push_back(element.second.size);
  }
  return ret_val;
}

std::vector<GraphWeight> extract_value_CommMap_degree(std::map<GraphElem, Comm>
    const& input_map) {
  std::vector<GraphWeight> ret_val;
  for (auto const& element : input_map) {
    ret_val.push_back(element.second.degree);
  }
  return ret_val;
}

std::vector<GraphElem>
extract_vertex_VertexCommMap(
    std::unordered_map<GraphElem, GraphElem> const& input_map) {
  std::vector<GraphElem> ret_val;
  for (auto const& element : input_map) {
    ret_val.push_back(element.first);
  }
  return ret_val;
}

std::vector<GraphElem> extract_comm_VertexCommMap(
    std::unordered_map<GraphElem, GraphElem> const& input_map) {
  std::vector<GraphElem> ret_val;
  for (auto const& element : input_map) {
    ret_val.push_back(element.second);
  }
  return ret_val;
}

GraphElem distGetMaxIndex_h(const ClusterLocalMap &clmap, const GraphWeightVector &counter,
        const GraphWeight selfLoop, const CommVector &localCinfo,
        const CommMap &remoteCinfo,
        const GraphWeight vDegree,
                          const GraphElem currSize,
                          const GraphWeight currDegree,
        const GraphElem currComm,
        const GraphElem base,
        const GraphElem bound,
        const GraphWeight constant)
{
  ClusterLocalMap::const_iterator storedAlready;
  GraphElem maxIndex = currComm;
  GraphWeight curGain = 0.0, maxGain = 0.0;
  GraphWeight eix = static_cast<GraphWeight>(counter[0]) - static_cast<GraphWeight>(selfLoop);

  GraphWeight ax = currDegree - vDegree;
  GraphWeight eiy = 0.0, ay = 0.0;

  GraphElem maxSize = currSize;
  GraphElem size = 0;

  storedAlready = clmap.begin();
#ifdef DEBUG_PRINTF
  assert(storedAlready != clmap.end());
#endif
  do {
      if (currComm != storedAlready->first) {

          // is_local, direct access local info
          if ((storedAlready->first >= base) && (storedAlready->first < bound)) {
              ay = localCinfo[storedAlready->first-base].degree;
              size = localCinfo[storedAlready->first - base].size;
          }
          else {
              // is_remote, lookup map
              CommMap::const_iterator citer = remoteCinfo.find(storedAlready->first);
              ay = citer->second.degree;
              size = citer->second.size;
          }

          eiy = counter[storedAlready->second];

          curGain = 2.0 * (eiy - eix) - 2.0 * vDegree * (ay - ax) * constant;

          if ((curGain > maxGain) ||
                  ((curGain == maxGain) && (curGain != 0.0) && (storedAlready->first < maxIndex))) {
              maxGain = curGain;
              maxIndex = storedAlready->first;
              maxSize = size;
          }
      }
      storedAlready++;
  } while (storedAlready != clmap.end());

  if ((maxSize == 1) && (currSize == 1) && (maxIndex > currComm))
    maxIndex = currComm;

  return maxIndex;
} // distGetMaxIndex_h

GraphWeight distBuildLocalMapCounter_h(const GraphElem e0, const GraphElem e1,
           ClusterLocalMap &clmap,
           GraphWeightVector &counter,
           const Graph &g, const CommunityVector &currComm,
           const VertexCommMap &remoteComm,
           const GraphElem vertex,
           const GraphElem base, const GraphElem bound)
{
  GraphElem numUniqueClusters = 1L;
  GraphWeight selfLoop = 0.0;
  ClusterLocalMap::const_iterator storedAlready;

  for (GraphElem j = e0; j < e1; j++) {

    const Edge &edge = g.getEdge(j);
    const GraphElem &tail = edge.tail;
    const GraphWeight &weight = edge.weight;
    GraphElem tcomm;

    if (tail == vertex + base)
      selfLoop += weight;


    // is_local, direct access local CommunityVector
    if ((tail >= base) && (tail < bound))
      tcomm = currComm[tail - base];
    else { // is_remote, lookup map
      VertexCommMap::const_iterator iter = remoteComm.find(tail);

#ifdef DEBUG_PRINTF
      assert(iter != remoteComm.end());
#endif
      tcomm = iter->second;
    }

    storedAlready = clmap.find(tcomm);

    if (storedAlready != clmap.end())
      counter[storedAlready->second] += weight;
    else {
        clmap.insert(ClusterLocalMap::value_type(tcomm, numUniqueClusters));
        counter.push_back(weight);
        numUniqueClusters++;
    }
  }

  return selfLoop;
} // distBuildLocalMapCounter_h


void distExecuteLouvainIteration_hybrid(const GraphElem i, const DistGraph &dg,
         const CommunityVector &currComm,
         CommunityVector &targetComm,
               const GraphWeightVector &vDegree,
                                 CommVector &localCinfo,
                                 CommVector &localCupdate,
         const VertexCommMap &remoteComm,
                                 const CommMap &remoteCinfo,
                                 CommMap &remoteCupdate,
                                 const GraphWeight constantForSecondTerm,
                                 GraphWeightVector &clusterWeight,
         const int me, CommunityVector &temp_targetComm)
{
  GraphElem localTarget = -1;
  GraphElem e0, e1;
  GraphWeight selfLoop = 0.0;
  ClusterLocalMap clmap;
  GraphWeightVector counter;

  const GraphElem base = dg.getBase(me), bound = dg.getBound(me);
  const Graph &g = dg.getLocalGraph();
  const GraphElem cc = currComm[i];
  GraphWeight ccDegree;
  GraphElem ccSize;
  bool currCommIsLocal=false;
  bool targetCommIsLocal=false;

  // Current Community is local
  if (cc >= base && cc < bound) {
  ccDegree=localCinfo[cc-base].degree;
        ccSize=localCinfo[cc-base].size;
        currCommIsLocal=true;
  } else {
  // is remote
        CommMap::const_iterator citer = remoteCinfo.find(cc);
  ccDegree = citer->second.degree;
  ccSize = citer->second.size;
  currCommIsLocal=false;
  }

  g.getEdgeRangeForVertex(i, e0, e1);

  if (e0 != e1) {
    clmap.insert(ClusterLocalMap::value_type(cc, 0));
    counter.push_back(0.0);

    selfLoop =  distBuildLocalMapCounter_h(e0, e1, clmap, counter, g, currComm, remoteComm, i, base, bound);

    clusterWeight[i] += counter[0];

    temp_targetComm[i] = distGetMaxIndex_h(clmap, counter, selfLoop, localCinfo, remoteCinfo, vDegree[i], ccSize, ccDegree, cc, base, bound, constantForSecondTerm);

  }
  else
    temp_targetComm[i] = cc;

#if 0
   // is the Target Local?
   if (localTarget >= base && localTarget < bound) {
      targetCommIsLocal = true;
   }

  // current and target comm are local - atomic updates to vectors
  if ((localTarget != cc) && (localTarget != -1) && currCommIsLocal && targetCommIsLocal) {

#ifdef DEBUG_PRINTF
        assert( base < localTarget < bound);
        assert( base < cc < bound);
  assert( cc - base < localCupdate.size());
  assert( localTarget - base < localCupdate.size());
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

  // current is local, target is not - do atomic on local, accumulate in Maps for remote
  if ((localTarget != cc) && (localTarget != -1) && currCommIsLocal && !targetCommIsLocal) {
        #pragma omp atomic update
        localCupdate[cc-base].degree -= vDegree[i];
        #pragma omp atomic update
        localCupdate[cc-base].size--;

        // search target!
        CommMap::iterator iter=remoteCupdate.find(localTarget);

        #pragma omp atomic update
        iter->second.degree += vDegree[i];
        #pragma omp atomic update
        iter->second.size++;
  }

   // current is remote, target is local - accumulate for current, atomic on local
   if ((localTarget != cc) && (localTarget != -1) && !currCommIsLocal && targetCommIsLocal) {
        #pragma omp atomic update
        localCupdate[localTarget-base].degree += vDegree[i];
        #pragma omp atomic update
        localCupdate[localTarget-base].size++;

        // search current
        CommMap::iterator iter=remoteCupdate.find(cc);

        #pragma omp atomic update
        iter->second.degree -= vDegree[i];
        #pragma omp atomic update
        iter->second.size--;
   }

   // current and target are remote - accumulate for both
   if ((localTarget != cc) && (localTarget != -1) && !currCommIsLocal && !targetCommIsLocal) {

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
#endif  // comment out
} // distExecuteLouvainIteration_hybrid

#endif
