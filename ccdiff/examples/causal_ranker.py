#  Copyright (c) 2018-present, Cruise LLC
#
#  This source code is licensed under the Apache License, Version 2.0,
#  found in the LICENSE file in the root directory of this source tree.
#  You may not use this file except in compliance with the License.

import os
import numpy as np
import networkx as nx
from matplotlib import pyplot as plt

def preprocess_sequence(data): 
    '''
    Preprocess the data to get the matrix
        data: [T_plan x N x N x T_sub] matrix
        return: [T_plan x N x N]
    '''
    print(data.shape)
    data = np.transpose(data, (0, 3, 1, 2))
    raw_data = data[0]
    for i in range(1, data.shape[0]):
        raw_data = np.concatenate((raw_data, data[i, -5:]), axis=0)
    raw_data = np.array(raw_data)
    return raw_data

def extract_initial_cond(data): 
    data = np.transpose(data, (0, 3, 1, 2))
    return data[0]

def detect_cliques(matrix): 
    # Function to calculate total weight of a strongly connected component (treated as a clique)
    def component_weight(component, graph):
        return sum(graph[u][v]['weight'] for u in component for v in component if u != v and graph.has_edge(u, v))

    G = nx.Graph()
    for i in range(len(matrix)):
        for j in range(i, len(matrix)):
            if matrix[i][j] > 0.5 or matrix[j][i] > 0.5: # ttc threshold for TTC graph
                G.add_edge(i, j, weight=matrix[i][j]+matrix[j][i])

    # Find all strongly connected components of size >= 3
    all_cliques = [list(component) for component in nx.enumerate_all_cliques(G) if len(component) >= 3]

    # Rank components based on total edge weights
    ranked_components = sorted(all_cliques, key=lambda component: component_weight(component, G), reverse=True)
    weight_sum = 0
    # Print the ranked components
    for rank, component in enumerate(ranked_components, 1):
        total_weight = component_weight(component, G)
        # print(f"Rank {rank}: Component {component}, Total Weight: {total_weight}")
        weight_sum += total_weight
    return len(all_cliques), ranked_components

def causal_ranking(causal_graph, topK = 10): 
    ranked_components = [detect_cliques(causal_graph[i])[1][:3] for i in range(causal_graph.shape[0])]
    results_freq = {}
    for ret in ranked_components: 
        for clique_id in ret: 
            for car_id in clique_id: 
                results_freq.setdefault(car_id, 0)
                results_freq[car_id] += 1
    # print(results_freq)
    sorted_by_values = dict(sorted(results_freq.items(), key=lambda item: item[1], reverse=True))
    print(sorted_by_values)
    print(idx, list(sorted_by_values.keys())[:topK])
    final_idx[idx] = list(sorted_by_values.keys())[:topK]
    return final_idx 

if __name__ == '__main__': 
    # parse the initial layout relationships
    import joblib
    scene_idx = joblib.load("0_scene_idx.pkl")
    filepath = "data/gt_results/scene_edit_eval/matrix/"
    topK = 10

    num_agents = []
    dist_sum_list, ttc_sum_list = [], []  
    final_idx = {}
    for idx in scene_idx: 
        file_dist = os.path.join(filepath, "dist_[\'{}\'].npy".format(str(idx)))
        file_ttc = os.path.join(filepath, "ttc_[\'{}\'].npy".format(str(idx)))
        # print(file_dist, file_ttc)
        dist = np.load(file_dist)
        ttc = np.load(file_ttc)
        # print(dist.shape, ttc.shape)
        num_agents.append(dist.shape[1])
        dist = extract_initial_cond(dist)
        ttc = extract_initial_cond(ttc)
        causal_ranking(ttc, topK=topK)
    import joblib
    joblib.dump(final_idx, "{}_control_idx_test_init.pkl".format(10))