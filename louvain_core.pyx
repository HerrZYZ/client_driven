# distutils: language = c++
# cython: language_level=3
# cython: linetrace=True
# distutils: define_macros=CYTHON_TRACE_NOGIL=1
from libcpp.set cimport set
from libcpp.vector cimport vector
cimport cython
import time
import random
import pickle
import pandas as pd
from scipy.sparse import csr_matrix
from random import sample

ctypedef fused int_or_long:
    int
    long

@cython.boundscheck(False)
@cython.wraparound(False)

def Allocate(unfull_ratio, unfull_cluster):
    randomnum=random.uniform(0, 1)
    sumvalue=0
    allocated_cluster = -1
    for index in range(len(unfull_ratio)):
        if randomnum < sumvalue:
            allocated_cluster=unfull_cluster[index]
            break
        else:
            sumvalue =sumvalue +unfull_ratio[index]

    return allocated_cluster


def fit_core(float resolution, float action_ratio, float future_leak_ratio, float localepochlength, float bigepochlength, float threshold, float difficulty, float num_shards, float loop_thres,  float tol, float[:] ou_node_probs, float[:] in_node_probs, float[:] self_loops,
             float[:] data, int_or_long[:] indices, int_or_long[:] indptr):  # pragma: no cover
    """Fit the clusters to the objective function.

    Parameters
    ----------
    resolution :
        Resolution parameter (positive).
    tol :
        Minimum increase in modularity to enter a new optimization pass.
    ou_node_probs :
        Distribution of node weights based on their out-edges (sums to 1).
    in_node_probs :
        Distribution of node weights based on their in-edges (sums to 1).
    self_loops :
        Weights of self loops.
    data :
        CSR format data array of the normalized adjacency matrix.
    indices :
        CSR format index array of the normalized adjacency matrix.
    indptr :
        CSR format index pointer array of the normalized adjacency matrix.

    Returns
    -------
    labels :
        Cluster index of each node.
    total_increase :
        Score of the clustering (total increase in modularity).
    """

    cdef int_or_long n = indptr.shape[0] - 1
    cdef int_or_long increase = 1
    cdef int_or_long cluster
    cdef int_or_long cluster_best
    cdef int_or_long cluster_node
    cdef int_or_long i
    cdef int_or_long j
    cdef int_or_long j1
    cdef int_or_long j2
    cdef int_or_long label
    cdef int_or_long element_unique_clusters
    cdef int_or_long ti
    cdef int_or_long globalepochlength
    cdef int_or_long epochlength

    cdef int_or_long num_of_tx_total

    cdef float increase_total = 0
    cdef float increase_pass
    cdef float delta
    cdef float delta_best
    cdef float delta_exit
    cdef float delta_local
    cdef float node_prob_in
    cdef float node_prob_ou
    cdef float ratio_in
    cdef float ratio_ou
    cdef float workload_afterjoin_clusternode
    cdef float workload_delta_clusternode
    cdef float workload_beforejoin_clusternode
    cdef float group_size_afterjoin_clusternode
    cdef float group_size_beforejoin_clusternode
    cdef float workload_delta_cluster
    cdef float workload_afterjoin_cluster
    cdef float workload_beforejoin_cluster
    cdef float group_size_afterjoin_cluster
    cdef float group_size_beforejoin_cluster

    cdef vector[float] neighbor_clusters_weights

    cdef vector[float] in_clusters_weights
    cdef vector[float] workload
    cdef vector[float] process_cost
    cdef vector[int_or_long] labels

    cdef set[int_or_long]  biggroup=()
    cdef set[int_or_long]  allnodes=()
    cdef set[int_or_long]  biggroup_ele=()
    cdef set[int_or_long]  smallgroup_ele=()

    cdef set[int_or_long] unique_clusters = ()
    cdef set[int_or_long] nodelistset = ()


    ti = long(resolution)

    globalepochlength=int(bigepochlength)
    epochlength=int(localepochlength)

    df_incremental = pd.read_csv('data/incremental_undirected_test_graph_numeric_'+str(ti-epochlength)+'~'+str(ti)+'h.csv')

    nodelist=list(df_incremental['from_address'])
    for nodeelement in nodelist:
        nodelistset.insert(nodeelement)
    nodelist=list(nodelistset)

    print('len(nodelist)'+str(len(nodelist)))

    f_num_of_tx_total = open('num_of_tx_incremental_ti=' + str(ti-1) + '.txt', "r")
    f_num_of_tx_total_ins = f_num_of_tx_total.read()
    num_of_tx_total=int(f_num_of_tx_total_ins)
    f_num_of_tx_total.close()

    print(num_of_tx_total)

    if ti%globalepochlength==epochlength:
        with open('train_workloadlist'+str(long(num_shards))+'d='+str(long(difficulty))+'ti='+str(ti-epochlength), 'rb') as f:
            workload_raw = pickle.load(f)

        with open('train_inclusterweight'+str(long(num_shards))+'d='+str(long(difficulty))+'ti='+str(ti-epochlength), 'rb') as f:
            in_clusters_weights_raw = pickle.load(f)

        with open('train_allocationlist' + str(long(num_shards))+'d='+str(long(difficulty))+'ti='+str(ti-epochlength), 'rb') as f:
            labels_list = pickle.load(f)
    else:
        with open('client_allo_workloadlist'+str(long(num_shards))+'d='+str(long(difficulty))+'ti='+str(ti-epochlength), 'rb') as f:
            workload_raw = pickle.load(f)

        with open('client_allo_degreelist'+str(long(num_shards))+'d='+str(long(difficulty))+'ti='+str(ti-epochlength), 'rb') as f:
            in_clusters_weights_raw = pickle.load(f)

        with open('client_allo_allocationlist' + str(long(num_shards))+'d='+str(long(difficulty))+'ti='+str(ti-epochlength), 'rb') as f:
            labels_list = pickle.load(f)

    labels_list=list(labels_list)

    for ele_labels_list in labels_list:
        labels.push_back(ele_labels_list)

    label_lengthpre=len(labels)
    label_lengthend=label_lengthpre+long(loop_thres)

    in_clusters_weights_list=[]

    for x in in_clusters_weights_raw:
        in_clusters_weights_list.append(x / num_of_tx_total)

    for ele_in_clusters_weights_list in in_clusters_weights_list:
        in_clusters_weights.push_back(ele_in_clusters_weights_list)

    workload_real=[]
    for x in workload_raw:
        workload_real.append(x/ num_of_tx_total)

    for i in range(n):
        neighbor_clusters_weights.push_back(0.)

    workloadtop100index=range(long(num_shards))

    for elements_workloadtop100index in workloadtop100index:
        biggroup.insert(elements_workloadtop100index)

    for i in nodelist:
        allnodes.insert(i)
        if labels[i] in workloadtop100index:
            biggroup_ele.insert(i)

    for i in range(label_lengthpre,label_lengthend):
        smallgroup_ele.insert(i)

    unfull_cluster = []
    unfull_ratio = []

    workload_counter=int(0)

    for ele_workload_real in workload_real:
        workload.push_back(ele_workload_real)
        if ele_workload_real < threshold:
            unfull_cluster.append(workload_counter)
            unfull_ratio.append((threshold-ele_workload_real)/(threshold * num_shards))
        workload_counter =workload_counter+int(1)

    print(sum(workload))
    print(0.3*difficulty+0.5*0.7)
    print(min(workload))
    print(unfull_cluster)
    print(unfull_ratio)

    label= 0
    workloadmin=workload[0]
    for i in range(len(workload)):
        if workload[i] < workloadmin:
            workloadmin = workload[i]
            label= i


#--------- New nodes allocation

    for i in smallgroup_ele:
        labels.push_back(label)

    labels_new=labels

    start_time = time.time()

    increase_pass = 0

    for i in allnodes:
        label = Allocate(unfull_ratio,unfull_cluster)
        if label < 0:
            randomnum=random.uniform(0, 1)
            if randomnum < action_ratio:
                unique_clusters.clear()
                cluster_node = labels[i]
                j1 = indptr[i]
                j2 = indptr[i + 1]
                for j in range(j1, j2):
                    label = labels[indices[j]]
                    randomnum=random.uniform(0, 1)
                    if randomnum < future_leak_ratio:
                        neighbor_clusters_weights[label] += data[j]
                        unique_clusters.insert(label)
                unique_clusters.erase(cluster_node)

                if not unique_clusters.empty():
                    node_prob_ou = ou_node_probs[i]
                    node_prob_in = in_node_probs[i]

                    delta_exit =  (neighbor_clusters_weights[cluster_node]-0.5 * self_loops[i]+ difficulty * (node_prob_in  -  neighbor_clusters_weights[cluster_node]))

                    delta_best = 0
                    cluster_best = cluster_node

                    for cluster in unique_clusters:
                        delta = (neighbor_clusters_weights[cluster]+ difficulty * (node_prob_in -  neighbor_clusters_weights[cluster]))

                        delta_local = delta - delta_exit
                        if delta_local < delta_best:
                            delta_best = delta_local
                            cluster_best = cluster

                    if delta_best < 0:
                        increase_pass += delta_best
                        labels_new[i] = cluster_best

                    for cluster in unique_clusters:
                        neighbor_clusters_weights[cluster] = 0

                neighbor_clusters_weights[cluster_node] = 0
        else:
            labels_new[i] = label


    print('increase_pass='+str(increase_pass))

    unique_clusters.clear()

    runningtime=time.time() - start_time
    print('runningtime='+str(runningtime))
    f_junktime = open('incrementalJunk_'+str(long(num_shards))+'Louvain_results_time_d='+str(long(difficulty))+'.txt', "w")
    f_junktime.write(str(runningtime))
    f_junktime.close()

    increase_total = 1
    return labels_new, increase_total
