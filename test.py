from utils import average_pairwise_distance, is_valid_network
from parse import read_input_file, read_output_file, write_output_file
import networkx as nx
import os
import sys
import matplotlib.pyplot as plt

def level2(G, originalT, old):
    T = nx.Graph()
    T.add_nodes_from(originalT)
    T.add_weighted_edges_from(originalT.edges.data('weight'))
    T_star = nx.Graph()
    T_star.add_nodes_from(originalT)
    T_star.add_weighted_edges_from(originalT.edges.data('weight'))

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
    if new < old and is_valid_network(G, T):
        return T
    else:
        return originalT

if __name__ == "__main__":
    output_dir = "outputs_4"
    new_output_dir = "outputs_5"
    input_dir = "inputs subset"
    for input_path in os.listdir(input_dir):
        graph_name = input_path.split(".")[0]
        G = read_input_file(f"{input_dir}/{input_path}")
        # print('Output Graph:', average_pairwise_distance(T))
        T = read_output_file(f"{output_dir}/{graph_name}.out", G)
        T_star = read_output_file(f"{output_dir}/{graph_name}.out", G)
        old = average_pairwise_distance(T)

        subG = nx.subgraph(G, T.nodes)
        T = nx.minimum_spanning_tree(subG, weight='weight', algorithm='prim')
        T_star = nx.minimum_spanning_tree(subG, weight='weight', algorithm='prim')
        for solution_vertex in list(nx.nodes(T)):
            T_star.remove_node(solution_vertex)
            if nx.is_dominating_set(G, nx.nodes(T_star)) and nx.is_connected(T_star) and average_pairwise_distance(T_star) < average_pairwise_distance(T):
                T.remove_node(solution_vertex)
            else:
                T_star.add_node(solution_vertex)
                for (u, v, wt) in T.edges.data('weight'):
                    if u == solution_vertex or v == solution_vertex:
                        T_star.add_edge(u, v, weight=wt)

        for solution_vertex in list(nx.nodes(T)):
            T_star.remove_node(solution_vertex)
            if nx.is_dominating_set(G, nx.nodes(T_star)) and nx.is_connected(T_star) and average_pairwise_distance(T_star) < average_pairwise_distance(T):
                T.remove_node(solution_vertex)
            else:
                T_star.add_node(solution_vertex)
                for (u, v, wt) in T.edges.data('weight'):
                    if u == solution_vertex or v == solution_vertex:
                        T_star.add_edge(u, v, weight=wt)

        for solution_vertex in list(nx.nodes(T)):
            T_star.remove_node(solution_vertex)
            if nx.is_dominating_set(G, nx.nodes(T_star)) and nx.is_connected(T_star) and average_pairwise_distance(T_star) < average_pairwise_distance(T):
                T.remove_node(solution_vertex)
            else:
                T_star.add_node(solution_vertex)
                for (u, v, wt) in T.edges.data('weight'):
                    if u == solution_vertex or v == solution_vertex:
                        T_star.add_edge(u, v, weight=wt)
        for solution_vertex in list(nx.nodes(T)):
            T_star.remove_node(solution_vertex)
            if nx.is_dominating_set(G, nx.nodes(T_star)) and nx.is_connected(T_star) and average_pairwise_distance(T_star) < average_pairwise_distance(T):
                T.remove_node(solution_vertex)
            else:
                T_star.add_node(solution_vertex)
                for (u, v, wt) in T.edges.data('weight'):
                    if u == solution_vertex or v == solution_vertex:
                        T_star.add_edge(u, v, weight=wt)

        for solution_vertex in list(nx.nodes(T)):
            T_star.remove_node(solution_vertex)
            if nx.is_dominating_set(G, nx.nodes(T_star)) and nx.is_connected(T_star) and average_pairwise_distance(T_star) < average_pairwise_distance(T):
                T.remove_node(solution_vertex)
            else:
                T_star.add_node(solution_vertex)
                for (u, v, wt) in T.edges.data('weight'):
                    if u == solution_vertex or v == solution_vertex:
                        T_star.add_edge(u, v, weight=wt)

        for solution_vertex in list(nx.nodes(T)):
            T_star.remove_node(solution_vertex)
            if nx.is_dominating_set(G, nx.nodes(T_star)) and nx.is_connected(T_star) and average_pairwise_distance(T_star) < average_pairwise_distance(T):
                T.remove_node(solution_vertex)
            else:
                T_star.add_node(solution_vertex)
                for (u, v, wt) in T.edges.data('weight'):
                    if u == solution_vertex or v == solution_vertex:
                        T_star.add_edge(u, v, weight=wt)

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
                    T_lvl2 = level2(G, T_star, cur)
                    if T_lvl2 == T_star:
                        T_star.remove_edge(u, v)
                        T_star.remove_node(v)
                    else:
                        T = nx.Graph()
                        T.add_nodes_from(T_lvl2)
                        T.add_weighted_edges_from(T_lvl2.edges.data('weight'))
                        T_star = nx.Graph()
                        T_star.add_nodes_from(T_lvl2)
                        T_star.add_weighted_edges_from(T_lvl2.edges.data('weight'))
            elif v in nx.nodes(T) and u not in nx.nodes(T):
                T_star.add_node(u)
                T_star.add_edge(v, u, weight=w)
                if average_pairwise_distance(T_star) < cur:
                    T.add_node(u)
                    T.add_edge(v, u, weight=w)
                else:
                    T_lvl2 = level2(G, T_star, cur)
                    if T_lvl2 == T_star:
                        T_star.remove_edge(v, u)
                        T_star.remove_node(u)
                    else:
                        T = nx.Graph()
                        T.add_nodes_from(T_lvl2)
                        T.add_weighted_edges_from(T_lvl2.edges.data('weight'))
                        T_star = nx.Graph()
                        T_star.add_nodes_from(T_lvl2)
                        T_star.add_weighted_edges_from(T_lvl2.edges.data('weight'))

        for u, v, w in G.edges.data('weight'):
            cur = average_pairwise_distance(T)
            if u in nx.nodes(T) and v not in nx.nodes(T):
                T_star.add_node(v)
                T_star.add_edge(u, v, weight=w)
                if average_pairwise_distance(T_star) < cur:
                    T.add_node(v)
                    T.add_edge(u, v, weight=w)
                else:
                    T_lvl2 = level2(G, T_star, cur)
                    if T_lvl2 == T_star:
                        T_star.remove_edge(u, v)
                        T_star.remove_node(v)
                    else:
                        T = nx.Graph()
                        T.add_nodes_from(T_lvl2)
                        T.add_weighted_edges_from(T_lvl2.edges.data('weight'))
                        T_star = nx.Graph()
                        T_star.add_nodes_from(T_lvl2)
                        T_star.add_weighted_edges_from(T_lvl2.edges.data('weight'))
            elif v in nx.nodes(T) and u not in nx.nodes(T):
                T_star.add_node(u)
                T_star.add_edge(v, u, weight=w)
                if average_pairwise_distance(T_star) < cur:
                    T.add_node(u)
                    T.add_edge(v, u, weight=w)
                else:
                    T_lvl2 = level2(G, T_star, cur)
                    if T_lvl2 == T_star:
                        T_star.remove_edge(v, u)
                        T_star.remove_node(u)
                    else:
                        T = nx.Graph()
                        T.add_nodes_from(T_lvl2)
                        T.add_weighted_edges_from(T_lvl2.edges.data('weight'))
                        T_star = nx.Graph()
                        T_star.add_nodes_from(T_lvl2)
                        T_star.add_weighted_edges_from(T_lvl2.edges.data('weight'))

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

        for solution_vertex in list(nx.nodes(T)):
            T_star.remove_node(solution_vertex)
            if nx.is_dominating_set(G, nx.nodes(T_star)) and nx.is_connected(T_star) and average_pairwise_distance(T_star) < average_pairwise_distance(T):
                T.remove_node(solution_vertex)
            else:
                T_star.add_node(solution_vertex)
                for (u, v, wt) in T.edges.data('weight'):
                    if u == solution_vertex or v == solution_vertex:
                        T_star.add_edge(u, v, weight=wt)

        for solution_vertex in list(nx.nodes(T)):
            T_star.remove_node(solution_vertex)
            if nx.is_dominating_set(G, nx.nodes(T_star)) and nx.is_connected(T_star) and average_pairwise_distance(T_star) < average_pairwise_distance(T):
                T.remove_node(solution_vertex)
            else:
                T_star.add_node(solution_vertex)
                for (u, v, wt) in T.edges.data('weight'):
                    if u == solution_vertex or v == solution_vertex:
                        T_star.add_edge(u, v, weight=wt)

        new = average_pairwise_distance(T)
        if new < old:
            if is_valid_network(G, T):
                print(graph_name, old - new)
                write_output_file(T, f"{new_output_dir}/{graph_name}.out")
            else:
                print(graph_name, 'new solution invalid')
        else:
            print(graph_name)
