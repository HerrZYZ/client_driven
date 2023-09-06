import pandas as pd
import numpy as np
from sknetwork.clustering import Louvain, modularity
from scipy.sparse import csr_matrix
import pickle
import time
from operator import itemgetter

difficulty_parameter=2
action_ratio_list=[0.3, 0.8]
epochlength = 1
future_leak_ratio_list = [0.9,0.5,0.1]
# bigepochlengthlist=[40,100,200,20]
bigepochlength = 200

aware_history_length = 1
def checkifclusterempty(x):
    for i in range(16):
       if x.item((0, i)) ==0:
           print('error!!empty cluster')

for future_leak_ratio in future_leak_ratio_list:
    for action_ratio in action_ratio_list:
        bigloopindex = 0
        f_k = open('action_ratio'+str(action_ratio)+'future_leak_ratio'+str(future_leak_ratio) + str(16)+'client_allo'+str(epochlength)+'timefluiding_incremental_Louvain_results_shardnum_k_d=' + str(difficulty_parameter) +'bigloopindex='+str(bigloopindex) +'bigepochlength='+str(bigepochlength)+'.txt', "w")
        f_time = open('action_ratio'+str(action_ratio)+'future_leak_ratio'+str(future_leak_ratio) + str(16)+'client_allo'+str(epochlength)+'timefluiding_incremental_Louvain_results_time_d=' + str(difficulty_parameter) +'bigloopindex='+str(bigloopindex) + 'bigepochlength='+str(bigepochlength)+'.txt', "w")
        f_ratio = open('action_ratio'+str(action_ratio)+'future_leak_ratio'+str(future_leak_ratio) + str(16)+'client_allo'+str(epochlength)+'timefluiding_incremental_Louvain_results_ratio_d=' + str(difficulty_parameter) +'bigloopindex='+str(bigloopindex) + 'bigepochlength='+str(bigepochlength)+'.txt', "w")
        f_workload = open('action_ratio'+str(action_ratio) + 'future_leak_ratio'+str(future_leak_ratio) + str(16)+'client_allo'+str(epochlength)+'timefluiding_incremental_Louvain_results_workload_d=' + str(difficulty_parameter) +'bigloopindex='+str(bigloopindex) + 'bigepochlength='+str(bigepochlength)+'.txt', "w")
        f_throuput = open('action_ratio'+str(action_ratio) + 'future_leak_ratio'+str(future_leak_ratio) + str(16)+'client_allo'+str(epochlength)+'timefluiding_incremental_Louvain_results_throughput_d=' + str(difficulty_parameter) +'bigloopindex='+str(bigloopindex) + 'bigepochlength='+str(bigepochlength)+'.txt', "w")

        tilist= list(range(epochlength + bigloopindex*bigepochlength,bigepochlength + bigloopindex*bigepochlength,epochlength))

        df_previous = pd.read_csv('data/undirected_training_graph_numeric_1h.csv')

        num_of_tx_incremental = sum(df_previous['size'])
        f_num_of_tx_incremental = open('num_of_tx_incremental_ti=' + str(0) + '.txt', "w")
        f_num_of_tx_incremental.write(str(num_of_tx_incremental))
        f_num_of_tx_incremental.close()

        for ti in tilist:
            df_incremental = pd.read_csv('data/incremental_undirected_test_graph_numeric_'+str(ti-epochlength)+'~'+str(ti)+'h.csv')
            new_address_num= max(df_incremental['from_address'])-max(df_previous['from_address'])
            df_nodeedge = pd.concat([df_incremental, df_previous])
            df_nodeedge = df_nodeedge.groupby(['from_address', 'to_address'])['size'].agg('sum').reset_index(name='size')

            num_of_tx_incremental=sum(df_incremental['size'])
            f_num_of_tx_incremental = open('num_of_tx_incremental_ti=' + str(ti) + '.txt',"w")
            f_num_of_tx_incremental.write(str(num_of_tx_incremental))
            f_num_of_tx_incremental.close()

            df_previous=df_nodeedge

            df_from=df_incremental['from_address']
            df_to=df_incremental['to_address']
            weights=df_incremental['size']

            row = df_from.to_numpy()
            col = df_to.to_numpy()
            data = weights.to_numpy()

            adjacency = csr_matrix((data, (row, col)))

            klist=[16]
            for k in klist:
                thresholdvalue = 1 / k

                louvain = Louvain(resolution=ti, action_ratio = action_ratio, future_leak_ratio = future_leak_ratio, localepochlength=epochlength, bigepochlength=aware_history_length, threshold=thresholdvalue/2, difficulty = difficulty_parameter, num_shards=k, loop_thres = new_address_num, modularity='newman', tol_aggregation=k*(1e-5), tol_optimization = k*(1e-5), n_aggregations=1, shuffle_nodes= False, verbose=True, return_aggregate = True)

                res = louvain.fit(adjacency)

                groupmatrix = res.aggregate_

                summ = res.aggregate_.sum()

                sum0_res = (res.aggregate_.sum(axis=0))

                diag = groupmatrix.diagonal()

                trace = diag.sum()

                labels=res.labels_
                labels_unique, counts = np.unique(res.labels_, return_counts=True)
                print(labels_unique, counts)

                ratiolist=[]

                checkifclusterempty(sum0_res)

                for i in range(k):
                    if sum0_res.item((0, i)) == 0:
                        ratiolist.append(0)
                    else:
                        ratiolist.append(1 - diag[i] / sum0_res.item((0, i)))


                allgroupworkload = []
                allgroupindex = []

                for i in range(len(diag)):
                    allgroupworkload.append(difficulty_parameter * (sum0_res.item((0, i)) - diag[i]) + 0.5 * diag[i])
                    allgroupindex.append(i)


                intra = trace

                throughputlist=[]

                valuelist = list(allgroupworkload)

                for i in range(k):
                    if allgroupworkload[i] > thresholdvalue * (0.5*summ):
                        throughput = (0.5 * sum0_res.item((0, i))) * thresholdvalue / allgroupworkload[i]

                    else:
                        throughput = (0.5 * sum0_res.item((0, i)))/(0.5*summ)

                    throughputlist.append(throughput)

                f_latency = open(str(0.01) + "incre_latency" + str(k) + 'd=' + str(
                    difficulty_parameter)  + 'bigepochlength='+str(bigepochlength)+".txt", "w")

                for i in range(k):
                    f_latency.write(str(valuelist[i]))
                    f_latency.write(',')

                balance_metric = np.std(valuelist) / np.mean(valuelist)

                f_k.write(str(len(valuelist)))
                f_k.write(',')

                f_junktime_out = open('incrementalJunk_' + str(k) + 'Louvain_results_time_d=' + str(difficulty_parameter) + '.txt', "r")
                f_junktime_out_ins = f_junktime_out.read()


                f_time.write(str(f_junktime_out_ins))
                f_time.write(',')
                f_junktime_out.close()


                f_ratio.write(str(1-trace/summ))
                f_ratio.write(',')


                f_workload.write(str(balance_metric))
                f_workload.write(',')

                f_throuput.write(str(sum(throughputlist)))
                f_throuput.write(',')

                f = open('client_allo'+str(epochlength)+"timefluiding_incremental_Louvain_results_crossratio" +str(k)+'d='+str(difficulty_parameter)+'bigepochlength='+str(bigepochlength)+".txt", "w")

                f.write('trace')
                f.write(str(intra))
                f.write('\n')

                f.write('ratio')
                f.write(str(1-intra/summ))
                f.write('\n')

                workload_check=difficulty_parameter*(summ-intra)+0.5*intra

                f.write('sum_workload')

                f.write(str(sum(valuelist)))

                f.write('workload_check'+str(workload_check))




                f.write('throughputsum')
                f.write(str(sum(throughputlist)))
                f.write('\n')

                f.write('throughputlist')

                for i in range(k):
                    f.write(str(throughputlist[i]))
                    f.write(',')
                f.write('\n')


                f.write('degree')

                for i in range(k):
                    f.write(str(sum0_res.item((0, i))))
                    f.write(',')
                f.write('\n')

                f.write('workload')

                for i in range(k):
                    f.write(str(difficulty_parameter *  (sum0_res.item((0, i)) - diag[i]) + 0.5 * diag[i]))
                    f.write(',')
                f.write('\n')

                f.write('degree:workload')


                f.write('ratiolist')

                for i in range(k):
                    f.write(str(ratiolist[i]))
                    f.write(',')
                f.write('\n')


                f.write('lenlist')
                f.write(str(len(valuelist)))
                f.write('\n')


                f.write('num_nodes')

                for i in range(len(valuelist)):
                    f.write(str(counts[i]))
                    f.write(',')

                f.close()

                print(sum(valuelist))

                with open('client_allo_allocationlist'+str(k)+'d='+str(difficulty_parameter)+'ti='+str(ti), 'wb') as f:
                    pickle.dump(labels, f)

                with open('client_allo_workloadlist'+str(k)+'d='+str(difficulty_parameter)+'ti='+str(ti), 'wb') as f:
                    pickle.dump(valuelist, f)

                sum0_res_list=sum0_res.tolist()

                with open('client_allo_degreelist'+str(k)+'d='+str(difficulty_parameter)+'ti='+str(ti), 'wb') as f:
                    pickle.dump(sum0_res_list[0], f)

        f_k.close()
        f_time.close()
        f_ratio.close()
        f_workload.close()
        f_throuput.close()

        # df_incremental = pd.read_csv(
        #     'data/incremental_undirected_test_graph_numeric_' + str(bigepochlength + bigloopindex*bigepochlength - epochlength) + '~' + str(
        #         bigepochlength + bigloopindex*bigepochlength) + 'h.csv')
        # df_nodeedge = pd.concat([df_incremental, df_previous])
        # df_nodeedge = df_nodeedge.groupby(['from_address', 'to_address'])['size'].agg('sum').reset_index(name='size')
        # df_previous = df_nodeedge

