import networkx as nx
from parse import read_input_file, read_output_file, write_output_file
from utils import is_valid_network, average_pairwise_distance
import sys
import random
from random import choice
import heapq
import math
import matplotlib.pyplot as plt
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
    degreeSort = sorted(G.degree, key=lambda x: x[1], reverse=True)
  

    maximum = degreeSort[0][0]
    T.add_node(maximum)
    towers.add(maximum)
    vertexSet.remove(maximum)
    for v in G.neighbors(maximum):
        cities.add(v)    

    while len(vertexSet) != 0:
        start = random.sample(vertexSet, 1)
        value = math.inf
        node = []
        for s, end, weight in G.edges(start, data=True):
            if end in cities:
                for city, tower, cost in G.edges(end, data=True):
                    if tower in towers:
                        temp = T.copy()
                        temp.add_edge(tower, city, weight=cost['weight'])
                        currCost = average_pairwise_distance(temp)
                        if currCost < value:
                            node = []
                            value = currCost
                            test = (city, tower, cost)
                            node.append(test)
            for c in cities:
                common = nx.common_neighbors(G, c, end)
                if common is not None:
                    for n in common:
                        edgeNode = G.get_edge_data(end, n)
                        for middle, city, cost1 in G.edges(n, data=True):
                            # find corresponding city
                            if city in cities:
                                for connection, tower, cost2 in G.edges(city, data=True):
                                    if tower in towers:
                                        temp = T.copy()
                                        temp.add_edge(n, end, weight=edgeNode['weight'])
                                        temp.add_edge(n, city, weight=cost1['weight'])
                                        temp.add_edge(tower, city, weight=cost2['weight'])
                                        currCost = average_pairwise_distance(temp)
                                        if currCost < value:
                                            value = currCost
                                            node = []
                                            test = (end, n, edgeNode)
                                            node.append(test)
                                            test1 = (n, city, cost1)
                                            node.append(test1)
                                            test2 = (city, tower, cost2)
                                            node.append(test2)
        if node is not None:  
            for pair in node:
                begin = pair[0]
                end = pair[1]
                weight = pair[2]
                cycleTest = T.copy()
                cycleTest.add_edge(begin, end, weight=weight['weight'])
                if not nx.is_tree(cycleTest):
                    break
                T.add_edge(begin, end, weight=weight['weight'])
                if begin in cities: 
                    cities.remove(begin)
                if end in cities:
                    cities.remove(end)
                towers.add(begin)
                towers.add(end)
                if begin in vertexSet: 
                    vertexSet.remove(begin)
                if end in vertexSet:
                    vertexSet.remove(end)
                for v in G.neighbors(begin):
                    if v not in towers: 
                        if v not in cities:
                            cities.add(v)
                for v in G.neighbors(end):
                    if v not in towers:
                        if v not in cities:
                            cities.add(v)
            if start[0] in vertexSet:
                vertexSet.remove(start[0])
                if start[0] not in cities:
                    cities.add(start[0])
            # print(vertexSet)
            # print(T.nodes)
    return T

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
        T = dijkstraSet(G)
        #print("Average  pairwise distance: {}".format(average_pairwise_distance(T)))
        write_output_file(T, f"{output_dir}/{graph_name}.out")
