import networkx as nx
from networkx import Graph
from parse import read_input_file, write_output_file
from utils import is_valid_network, average_pairwise_distance
import sys
import random
from random import choice
import heapq
import math
import matplotlib.pyplot as plt


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
    # G.nodes
    print("vertex set:")
    print(vertexSet)
    # print("")
    # place for later optimization with randomly selecting vertex from upper quartile
    degreeSort = sorted(G.degree, key=lambda x: x[1], reverse=True)
    # print("sorted degrees")
    # print(degreeSort)

    maximum = degreeSort[0][0]
    T.add_node(maximum)
    towers.add(maximum)
    vertexSet.remove(maximum)
    for v in G.neighbors(maximum):
        cities.add(v)
        # vertexSet.remove(v)
    

    while len(vertexSet) != 0:
        heap = []
        start = random.sample(vertexSet, 1)
        # print(start[0])
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

if __name__ == '__main__':
    assert len(sys.argv) == 2
    path = sys.argv[1]
    G = read_input_file(path)
    T = dijkstraSet(G)
    nx.draw(G)
    plt.draw()
    plt.show()
    nx.draw(T)
    plt.draw()
    plt.show()
    assert is_valid_network(G, T)
    print("Average  pairwise distance: {}".format(average_pairwise_distance(T)))
    write_output_file(T, 'outputs/test.out')
