import networkx as nx
from parse import read_input_file, read_output_file, write_output_file
from utils import is_valid_network, average_pairwise_distance
import sys
import os
import random
from pqdict import pqdict
import matplotlib.pyplot as plt
# from multiprocessing import Process
# Here's an example of how to run your solver.
    # nx.draw(T_Output)  # networkx draw()
    # nx.draw(T_MST_1)
    # plt.draw()  # pyplot draw()
    # plt.show()
# Usage: python3 solver.py test.in

def RajivMishraAlgorithm(G):
    T = nx.Graph()
    G.remove_edges_from(nx.selfloop_edges(G))

    all_vertices = list(nx.nodes(G))
    noOfVertices = len(all_vertices)

    vertice_degrees = nx.degree(G, all_vertices)
    vertice_degrees = sorted(vertice_degrees, key = lambda x: x[1], reverse = True)

    #handle the case of graph with one vertex having degree 'n-1'
    if vertice_degrees[0][1] == (len(all_vertices) - 1):
        T.add_node(vertice_degrees[0][0])
        return T


    # add all the edges in both directions
    w = nx.get_edge_attributes(G, 'weight')
    temp = w.keys()
    for i in list(temp):
        w[(i[1], i[0])] = w[i]

    avg_outgoing_edge_cost = {}
    for v in all_vertices:
        total = 0.0
        count = 0.0
        for n in nx.neighbors(G, v):
            total += w.get((v, n))
            count += 1
        avg_outgoing_edge_cost[v] = total/count

    def score(vertex):
        return (vertice_degrees[vertex][1])/avg_outgoing_edge_cost[vertex]

    sorted_vertices = sorted(all_vertices, key = lambda x: score(x), reverse = True)
    # vertices_sorted_by_scores

    slice = min(30, noOfVertices)

    T_Output = nx.Graph()
    T_min_score = float('inf')

    for iter in range(slice):
        T = nx.Graph()
        T_star = nx.Graph()
        towers = set()
        covered_vertices_count = dict()
        remaining_vertices = set()
        remaining_vertices.update(list(nx.nodes(G)))

        starting_node = sorted_vertices[iter]
        towers.add(starting_node)
        covered_vertices_count[starting_node] = covered_vertices_count.get(starting_node, 0) + 1
        remaining_vertices.remove(starting_node)

        for n in nx.all_neighbors(G, starting_node):
            covered_vertices_count[n] = covered_vertices_count.get(n, 0) + 1
            remaining_vertices.remove(n)

        T.add_node(starting_node)
        T_star.add_node(starting_node)

        while(len(remaining_vertices) != 0):
            running_cost = pqdict()
            minimum_edge = dict()
            covered_vertices = list(covered_vertices_count.keys())
            for tree_node in nx.nodes(T):
                for n in nx.all_neighbors(G, tree_node):
                    if n not in towers:
                        T_star.add_node(n)
                        T_star.add_edge(tree_node, n, weight=w[(tree_node, n)])
                        new_neighbor = nx.neighbors(G, n)
                        unique_neighbor = [i for i in new_neighbor if i not in covered_vertices]
                        cost = average_pairwise_distance(T_star)/((1.0 + len(unique_neighbor))*score(n))
                        if cost < running_cost.get(n, float('inf')):
                            running_cost[n] = cost
                            minimum_edge[n] = (tree_node, n)
                        T_star.remove_edge(tree_node, n)
                        T_star.remove_node(n)
                    else:
                        for common_v in list(nx.common_neighbors(T, tree_node, n)):
                            if common_v in nx.nodes(T):
                                old = w.get((common_v, n))
                                new = w.get((tree_node, n))
                                if new < old:
                                    T.remove_edge(common_v, n)
                                    T.add_edge(tree_node, n, weight=new)
                                    T_star.remove_edge(common_v, n)
                                    T_star.add_edge(tree_node, n, weight=new)

            selected_node = running_cost.pop()
            selected_edge = minimum_edge[selected_node]

            towers.add(selected_node)
            for neigh in nx.all_neighbors(G, selected_node):
                covered_vertices_count[neigh] = covered_vertices_count.get(neigh, 0) + 1
                if neigh in remaining_vertices:
                    remaining_vertices.remove(neigh)

            T.add_node(selected_node)
            T.add_edge(selected_edge[0], selected_edge[1], weight=w.get(selected_edge))
            T_star.add_node(selected_node)
            T_star.add_edge(selected_edge[0], selected_edge[1], weight=w.get(selected_edge))

        # print('Before T:', T.nodes, T.edges)
        # print('Before T_star:', T_star.nodes, T_star.edges)
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

        T = test(G, T)
        T = test(G, T)
        # print('After T:', T.nodes, T.edges)
        # print('After T_star:', T_star.nodes, T_star.edges)
        avg_dist = average_pairwise_distance(T)
        if avg_dist < T_min_score:
            T_Output = nx.Graph()
            T_Output.add_nodes_from(T)
            T_Output.add_weighted_edges_from(T.edges.data('weight'))
            T_min_score = avg_dist

    # T_towers = nx.subgraph(G, nx.nodes(T_Output))
    # print([i for i in T_towers.edges.data('weight') if i not in T.edges.data('weight')])
    # print(T_towers.edges.data('weight'))
    # T_MST_1 = nx.minimum_spanning_tree(T_towers, algorithm='kruskal')
    # print('Kruskal:', [i for i in T_MST_1.edges.data('weight')])
    # T_MST_2 = nx.minimum_spanning_tree(T_towers, algorithm='prim')
    # T_MST_3 = nx.minimum_spanning_tree(T_towers, algorithm='boruvka')
    # min_MST = min([T_MST_1, T_MST_2, T_MST_3], key = lambda x : average_pairwise_distance(x))
    # print('MST:', average_pairwise_distance(min_MST))
    # print('Non MST:', T_min_score)
    # if average_pairwise_distance(min_MST) < T_min_score:
    #     T_Output = nx.Graph()
    #     T_Output.add_nodes_from(min_MST)
    #     T_Output.add_weighted_edges_from(min_MST.edges.data('weight'))
    return T_Output

def test(G, originalT):
    T = nx.Graph()
    T.add_nodes_from(originalT)
    T.add_weighted_edges_from(originalT.edges.data('weight'))
    T_star = nx.Graph()
    T_star.add_nodes_from(originalT)
    T_star.add_weighted_edges_from(originalT.edges.data('weight'))

    old = average_pairwise_distance(originalT)
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
    output_dir = "outputs_3"
    output_d = "outputs_3"
    input_dir = "inputs subset"
    for input_path in os.listdir(input_dir):
        graph_name = input_path.split(".")[0]
        G = read_input_file(f"{input_dir}/{input_path}")
        T = RajivMishraAlgorithm(G)
        # print('Output Graph:', average_pairwise_distance(T))
        old_T = read_output_file(f"{output_d}/{graph_name}.out", G)
        old = average_pairwise_distance(old_T)
        new = average_pairwise_distance(T)
        if new < old:
            if is_valid_network(G, T):
                print(graph_name, old - new)
                write_output_file(T, f"{output_dir}/{graph_name}.out")
            else:
                print(graph_name, 'new solution invalid')
        else:
            print(graph_name)
        break
# def combine_outputs():
#     output_dir = "outputs"
#     output_Avik = "outputsAvik"
#     output_Cassidy = "outputsCassudy"
#     output_Raghav = "outputsRaghav"
#     input_dir = "inputs"
#     for input_path in os.listdir(input_dir):
#         graph_name = input_path.split(".")[0]
#         G = read_input_file(f"{input_dir}/{input_path}")
#         Avik_T = read_output_file(f"{output_Avik}/{graph_name}.out", G)
#         Cassidy_T = read_output_file(f"{output_Cassidy}/{graph_name}.out", G)
#         Raghav_T = read_output_file(f"{output_Raghav}/{graph_name}.out", G)
#
#         T = min([Avik_T, Cassidy_T, Raghav_T], key = lambda x: average_pairwise_distance(x))
#         write_output_file(T, f"{output_dir}/{graph_name}.out")
