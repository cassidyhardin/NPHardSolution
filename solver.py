import networkx as nx
from parse import read_input_file, read_output_file, write_output_file
from utils import is_valid_network, average_pairwise_distance
import sys
import os
import random
from pqdict import pqdict


def dijkstraSet(G):
    """
    Args:
        G: networkx.Graph

    Returns:
        T: networkx.Graph
    """
    T = nx.Graph()
    towers = set()
    cities = set()
    vertexSet = set()
    vertexSet.update(G.nodes)
    degreeSort = sorted(G.degree, key=lambda x: x[1], reverse=True)g
    # TODO: your code here!
    return RajivMishraAlgorithm(G)
# Here's an example of how to run your solver.

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

    slice = min(10, noOfVertices)

    T_Output = nx.Graph()
    T_min_score = float('inf')

    for iter in range(slice):
        T = nx.Graph()
        T_star = nx.Graph()
        towers = set()
        covered_vertices_count = dict()
        remaining_vertices = set()
        remaining_vertices.update(list(nx.nodes(G)))

        starting_node = vertice_degrees[iter][0]
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
                        cost = average_pairwise_distance(T_star)/(1.0 + len(unique_neighbor))
                        if cost < running_cost.get(n, float('inf')):
                            running_cost[n] = cost
                            minimum_edge[n] = (tree_node, n)
                        T_star.remove_edge(tree_node, n)
                        T_star.remove_node(n)
            selected_node = running_cost.pop()
            selected_edge = minimum_edge[selected_node]

            towers.add(selected_node)
            for neigh in nx.all_neighbors(G, selected_node):
                covered_vertices_count[neigh] = covered_vertices_count.get(neigh, 0) + 1
                if neigh in remaining_vertices:
                    remaining_vertices.remove(neigh)

            T.add_node(selected_node)
            T.add_edges_from([selected_edge])
            T_star.add_node(selected_node)
            T_star.add_edges_from([selected_edge])

        avg_dist = average_pairwise_distance(T)
        if avg_dist < T_min_score:
            T_Output = nx.Graph()
            T_Output.add_nodes_from(T)
            T_Output.add_edges_from(T.edges)
            T_min_score = avg_dist
    return T_Output

if __name__ == "__main__":
    output_dir = "outputs"
    input_dir = "inputs"
    for input_path in os.listdir(input_dir):
        graph_name = input_path.split(".")[0]
        G = read_input_file(f"{input_dir}/{input_path}")
        T = solver(G)
        #print("Average  pairwise distance: {}".format(average_pairwise_distance(T)))
        write_output_file(T, f"{output_dir}/{graph_name}.out")
