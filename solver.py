import networkx as nx
from parse import read_input_file, write_output_file
from utils import is_valid_network, average_pairwise_distance,average_pairwise_distance_fast

import networkx as nx
from parse import read_input_file, read_output_file, write_output_file
from utils import is_valid_network, average_pairwise_distance
import os
import heapq
import math
import os
import random


def one_by_one(s, l):
    for i in l:
        s.add(i)


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
def solve(G):
    """
    Args:
        G: networkx.Graph

    Returns:
        T: networkx.Graph
    """
    #rm_solution= dijkstraSet(G)
    G.remove_edges_from(nx.selfloop_edges(G))
    for node in G.nodes():
        if G.degree(node)>=len(G.nodes)-1:
            print("This Edge is connected to all other edges")
            ret_graph=nx.Graph()
            ret_graph.add_node(node)
            return ret_graph
    mst_prim = modified_mst(G,'prim')
    mst_kruskal = modified_mst(G, 'kruskal')
    mst_boruvka= modified_mst(G, 'boruvka')
    solution=min_of_mst([mst_prim,mst_boruvka,mst_kruskal])
    print("THIS IS THE SOLUTION")
    for a, b, data in sorted(solution.edges(data=True), key=lambda x: x[2]['weight']):
        print('{a} {b} {w}'.format(a=a, b=b, w=data['weight']))

    #if average_pairwise_distance_fast(solution)<=average_pairwise_distance_fast(rm_solution):
    return solution







def is_leaf(graph,node):
    if graph.degree(node)==1:
        return True
    else:
        return False

    # TODO: your code here!

def modified_mst(graph,algorithm):
    mst = nx.minimum_spanning_tree(graph,algorithm=algorithm)
    mst_copy = mst.copy()
    mst_copy_2 = mst.copy()
    # Set of vertices that cannot will have to be connected directly to the graph
    # print("THIS IS THE MST")
    vertex_set = set()
    node_to_be_added = None
    first_loop = True
    count = 0
    removed_nodes=set()
    for i in range(2):
        first_loop = False
        count += 1
        print("Iteration %s " % (count))
        mst = mst_copy.copy()
        mst_copy_2 = mst_copy.copy()
        for a, b, data in sorted(mst.edges(data=True), key=lambda x: x[2]['weight'], reverse=True):
            #print("DEBUGGER")
            #print(mst_copy.nodes(data=True),mst_copy_2.nodes(data=True))
            #print(mst_copy.edges(data=True), mst_copy_2.edges(data=True))
            #print('{a} {b} {w}'.format(a=a, b=b, w=data['weight']))
            if a not in set(mst_copy_2.nodes()) or b not in set(mst_copy_2.nodes()):
                continue
            if is_leaf(mst_copy_2, a):
                print("a is the leaf")
                mst_copy_2.remove_edge(a, b)
                node_to_be_added = a
                mst_copy_2.remove_node(a)

            elif is_leaf(mst_copy_2, b):
                print("b is the leaf")
                mst_copy_2.remove_edge(a, b)
                node_to_be_added = b
                mst_copy_2.remove_node(b)
            else:
                print("No leafs in this edge")
                # Edges connecting Two non leaf nodes cannot be removed
                distance_to_beat=average_pairwise_distance_fast(mst_copy_2)
                mst_copy_2.remove_edge(a, b)
                small_sub_graph = min(nx.connected_component_subgraphs(mst_copy_2), key=len)
                big_sub_graph = max(nx.connected_component_subgraphs(mst_copy_2), key=len)
                if vertices_covered_by_tree(big_sub_graph,graph) and average_pairwise_distance_fast(big_sub_graph)<distance_to_beat:
                    one_by_one(removed_nodes,list(small_sub_graph.nodes()))
                    print("SMALL GRAPH HAS BEEN PRUNED")
                    mst_copy_2=big_sub_graph
                    mst_copy=big_sub_graph

                elif vertices_covered_by_tree(small_sub_graph,graph) and average_pairwise_distance_fast(small_sub_graph)<distance_to_beat:
                    one_by_one(removed_nodes, list(big_sub_graph.nodes()))
                    print("BIG GRAPH HAS BEEN PRUNED")
                    mst_copy_2 = small_sub_graph
                    mst_copy = small_sub_graph
                else:
                    mst_copy_2.add_edge(a, b,weight=data['weight'])




                continue

            if average_pairwise_distance_fast(mst_copy_2) < average_pairwise_distance_fast(mst_copy):
                # if is_leaf(mst_copy,a) or is_leaf(mst_copy,b):
                # print("Pairwise distance is improved by removing this edge")

                if is_leaf(mst_copy, a) and not (is_leaf(mst_copy, b)) and (a not in vertex_set) and vertices_covered_by_tree(mst_copy_2,graph) :
                    # print("Removing this edge and node part a")
                    check_copy=mst_copy.copy()
                    check_copy.remove_node(a)
                    check_copy.remove_node(b)

                    if not node_in_vertices_covered_by_tree(a,check_copy,graph):
                        vertex_set.add(b)
                    else:
                        print(" B DIDN'T GET ADDED")
                    mst_copy.remove_edge(a, b)
                    mst_copy.remove_node(a)
                elif is_leaf(mst_copy, b) and not (is_leaf(mst_copy, a)) and (b not in vertex_set) and vertices_covered_by_tree(mst_copy_2,graph):
                    # print("Removing this edge and node part b")
                    check_copy = mst_copy.copy()
                    check_copy.remove_node(a)
                    check_copy.remove_node(b)

                    if not node_in_vertices_covered_by_tree(b,check_copy, graph):
                        vertex_set.add(a)
                    else:
                        print(" A DIDN'T GET ADDED")
                    mst_copy.remove_edge(a, b)
                    mst_copy.remove_node(b)
                else:
                    mst_copy_2.add_node(node_to_be_added)
                    mst_copy_2.add_edge(a, b, weight=data['weight'])

            else:
                mst_copy_2.add_node(node_to_be_added)

                mst_copy_2.add_edge(a, b, weight=data['weight'])

    print("THIS IS THE SOLUTION")
    for a, b, data in sorted(mst_copy.edges(data=True), key=lambda x: x[2]['weight']):
        print('{a} {b} {w}'.format(a=a, b=b, w=data['weight']))
    return mst_copy

def min_of_mst(list_mst):
    index=[average_pairwise_distance_fast(i) for i in list_mst].index(min([average_pairwise_distance_fast(i) for i in list_mst]))
    print("INDEX NUMBER %s"%(index))
    return list_mst[index]

def vertices_covered_by_tree(tree,graph):
    vertice_set=set()
    for node in tree.nodes():
        for neighbor in graph.neighbors(node):
            vertice_set.add(neighbor)
            vertice_set.add(node)
    return len(vertice_set)==len(graph.nodes())

def node_in_vertices_covered_by_tree(node,tree,graph):
    vertice_set=set()
    for node in tree.nodes():
        for neighbor in graph.neighbors(node):
            vertice_set.add(neighbor)
            vertice_set.add(node)
    return node in vertice_set
# Here's an example of how to run your solver.

# Usage: python3 solver.py test.in

if __name__ == '__main__' :
    output_dir = "outputs"
    input_dir = "inputs"
    for input_path in os.listdir(input_dir):
        graph_name = input_path.split(".")[0]
        print(graph_name)
        G = read_input_file(f"{input_dir}/{input_path}")
        T = solve(G)
        write_output_file(T, f"{output_dir}/{graph_name}.out")



