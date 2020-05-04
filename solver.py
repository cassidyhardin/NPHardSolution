import networkx as nx
from parse import read_input_file, write_output_file
from utils import is_valid_network, average_pairwise_distance,average_pairwise_distance_fast
import sys
import random
from random import choice
import networkx as nx
from parse import read_input_file, read_output_file, write_output_file
from utils import is_valid_network, average_pairwise_distance
import os
import heapq
import math
import os
import random
from pqdict import pqdict
import matplotlib.pyplot as plt



def primMSTwithHeuristic(G):
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


    # add all the edges in both directions in a dictionary
    w = nx.get_edge_attributes(G, 'weight')
    temp = w.keys()
    for i in list(temp):
        w[(i[1], i[0])] = w[i]

    # calculate the average outgoing edge cost for every solution_vertex
    # to be used with the
    avg_outgoing_edge_cost = {}
    for v in all_vertices:
        total = 0.0
        count = 0.0
        for n in nx.neighbors(G, v):
            total += w.get((v, n))
            count += 1
        avg_outgoing_edge_cost[v] = total/count

    # Scoring function for nodes based on their degree and their outgoing edge cost
    def score(vertex):
        return (vertice_degrees[vertex][1])/avg_outgoing_edge_cost[vertex]

    sorted_vertices = sorted(all_vertices, key = lambda x: score(x), reverse = True)
    # vertices_sorted_by_scores

    noOfIterations = min(30, noOfVertices)

    T_Output = nx.Graph()
    T_min_score = float('inf')

    # Run multiple iterations to get minimum output
    for iter in range(noOfIterations):
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

        # Run solution till it is a dominating set
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
                        # Cost serves as a heuristic
                        # Can be improved by looking at more data, such as seeing more than 1 step ahead or only looking at outoing edge costs of edges to unique neighbors
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

        # Remove nodes from the solution as long as the solution stays valid
        # The remval should reduce average pairwise distance reduces
        # Keeps iterating till there is no improvement
        change = float('inf')
        while(change > 0):
            originalAvgPairDist = average_pairwise_distance(T)
            for solution_vertex in list(nx.nodes(T)):
                T_star.remove_node(solution_vertex)
                if nx.is_dominating_set(G, nx.nodes(T_star)) and nx.is_connected(T_star) and average_pairwise_distance(T_star) < average_pairwise_distance(T):
                    T.remove_node(solution_vertex)
                else:
                    T_star.add_node(solution_vertex)
                    for (u, v, wt) in T.edges.data('weight'):
                        if u == solution_vertex or v == solution_vertex:
                            T_star.add_edge(u, v, weight=wt)
            change = originalAvgPairDist - average_pairwise_distance(T)

        T = addNodes(G, T)

        avg_dist = average_pairwise_distance(T)

        # takes the final output as the minimum of all iterations
        if avg_dist < T_min_score:
            T_Output = nx.Graph()
            T_Output.add_nodes_from(T)
            T_Output.add_weighted_edges_from(T.edges.data('weight'))
            T_min_score = avg_dist

    return T_Output

def addNodes(G, originalT):
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
    new = average_pairwise_distance(T)
    if new < old and is_valid_network(G, T):
        return T
    else:
        return originalT

def level2(G, originalT, old):
    T = nx.Graph()
    T.add_nodes_from(originalT)
    T.add_weighted_edges_from(originalT.edges.data('weight'))
    T_star = nx.Graph()
    T_star.add_nodes_from(originalT)
    T_star.add_weighted_edges_from(originalT.edges.data('weight'))

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

def one_by_one(s, l):
    for i in l:
        s.add(i)

def MST(G):
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
#Takes a list of mst's ' and returns the one with the smallest average min pairwise distance
def min_of_mst(list_mst):
    index=[average_pairwise_distance_fast(i) for i in list_mst].index(min([average_pairwise_distance_fast(i) for i in list_mst]))
    print("INDEX NUMBER %s"%(index))
    return list_mst[index]
#Checks if the the nodes of tree and all nodes that are neighbours of tree cover all nodes in graph
def vertices_covered_by_tree(tree,graph):
    vertice_set=set()
    for node in tree.nodes():
        for neighbor in graph.neighbors(node):
            vertice_set.add(neighbor)
            vertice_set.add(node)
    return len(vertice_set)==len(graph.nodes())
#Checks if node is at most a degree of separation 1 away from any vertex present in tree in graoh
def node_in_vertices_covered_by_tree(node,tree,graph):
    vertice_set=set()
    for n in tree.nodes():
        for neighbor in graph.neighbors(n):
            vertice_set.add(neighbor)
            vertice_set.add(n)
    return node in vertice_set


def dijkstraSet(G):
    """
    Args:
        G: networkx.Graph
    Returns:
        T: networkx.Graph
    """
    minTree = nx.Graph()
    towers = set()
    cities = set()
    vertexSet = set()
    vertexSet.update(G.nodes)
    degreeSort = sorted(G.degree, key=lambda x: x[1], reverse=True)
    for x in range(1):
        finalTree = nx.Graph()
        for v in degreeSort:
            towers = set()
            cities = set()
            vertexSet = set()
            vertexSet.update(G.nodes)
            T = nx.Graph()
            maximum = v[0]
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
            if T.number_of_nodes() > 0 and is_valid_network(G, T):
                if finalTree.number_of_nodes() > 0:
                    if average_pairwise_distance(finalTree) > average_pairwise_distance(T):
                        finalTree = T.copy()
                else:
                    finalTree = T.copy()
    return finalTree

def combine_outputs():
    output_dir = "outputs"
    output_Avik = "outputsAvik"
    output_Raghav = "outputsRaghav"
    input_dir = "inputs"
    for input_path in os.listdir(input_dir):
        graph_name = input_path.split(".")[0]
        G = read_input_file(f"{input_dir}/{input_path}")
        Avik_T = read_output_file(f"{output_Avik}/{graph_name}.out", G)
        Raghav_T = read_output_file(f"{output_Raghav}/{graph_name}.out", G)
        T = min([Avik_T,Raghav_T], key = lambda x: average_pairwise_distance(x))
        write_output_file(T, f"{output_dir}/{graph_name}.out")
        print("%s Written"%(graph_name))


if __name__ == '__main__' :
    output_dir = "outputs"
    input_dir = "inputs"
    for input_path in os.listdir(input_dir):
        graph_name = input_path.split(".")[0]
        print(graph_name)
        G = read_input_file(f"{input_dir}/{input_path}")
        # Tmst = MST(G)
        Tmst = primMSTwithHeuristic(G)
        Tds = dijkstraSet(G)
        Tprim = primMSTwithHeuristic(G)
        Tmin = nx.Graph()
        Gfinal = read_input_file(f"{input_dir}/{input_path}")
        if Tds.number_of_nodes() > 0:
            if average_pairwise_distance(Tmst) < average_pairwise_distance(Tds):
                if average_pairwise_distance(Tmst) < average_pairwise_distance(Tprim): 
                    Tmin = Tmst.copy()
                else:
                    Tmin = Tprim.copy()
            else:
                if average_pairwise_distance(Tds) < average_pairwise_distance(Tprim): 
                    Tmin = Tds.copy()
                else:
                    Tmin = Tprim.copy()
        else:
            if average_pairwise_distance(Tmst) < average_pairwise_distance(Tprim): 
                Tmin = Tmst.copy()
            else:
                Tmin = Tprim.copy()
        write_output_file(Tmin, f"{output_dir}/{graph_name}.out")
        print("Minimum Average Pairwise Distance:")
        print(average_pairwise_distance(Tmin))


