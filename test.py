from utils import average_pairwise_distance, is_valid_network
from parse import read_input_file, read_output_file, write_output_file
import networkx as nx
import os
import sys
import matplotlib.pyplot as plt

if __name__ == "__main__":
    output_dir = "outputs"
    new_output_dir = "outputsRaghav"
    input_dir = "inputs subset"
    for input_path in os.listdir(input_dir):
        graph_name = input_path.split(".")[0]
        print(graph_name)
        G = read_input_file(f"{input_dir}/{input_path}")
        # print('Output Graph:', average_pairwise_distance(T))
        T = read_output_file(f"{output_dir}/{graph_name}.out", G)
        T_star = read_output_file(f"{output_dir}/{graph_name}.out", G)
        old = average_pairwise_distance(T)

        t_nodes = list(nx.nodes(T))
        for u, v, w in G.edges.data('weight'):
            cur = average_pairwise_distance(T)
            if u in nx.nodes(T) and v not in nx.nodes(T):
                T_star.add_node(v)
                T_star.add_edge(u, v, weight=w)
                if average_pairwise_distance(T_star) < cur:
                    T.add_node(v)
                    T.add_edge(u, v, weight=w)
                else:
                    T_star.remove_edge(u, v)
                    T_star.remove_node(v)
            elif v in nx.nodes(T) and u not in nx.nodes(T):
                T_star.add_node(u)
                T_star.add_edge(v, u, weight=w)
                if average_pairwise_distance(T_star) < cur:
                    T.add_node(u)
                    T.add_edge(v, u, weight=w)
                else:
                    T_star.remove_edge(v, u)
                    T_star.remove_node(u)

        new = average_pairwise_distance(T)
        if new < old:
            if is_valid_network(G, T):
                write_output_file(T, f"{new_output_dir}/{graph_name}.out")
            else:
                continue
        else:
            continue
