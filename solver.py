import networkx as nx
from parse import read_input_file, write_output_file
from utils import is_valid_network, average_pairwise_distance,average_pairwise_distance_fast
import sys
import os
import matplotlib.pyplot as plt
def one_by_one(s, l):
    for i in l:
        s.add(i)

def solve(G):
    """
    Args:
        G: networkx.Graph

    Returns:
        T: networkx.Graph
    """
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
                    print("GRAPH HAS BEEN PRUNED")
                    mst_copy_2=big_sub_graph
                    mst_copy=big_sub_graph
                else:
                    mst_copy_2.add_edge(a, b,weight=data['weight'])




                continue

            if average_pairwise_distance_fast(mst_copy_2) < average_pairwise_distance_fast(mst_copy):
                # if is_leaf(mst_copy,a) or is_leaf(mst_copy,b):
                # print("Pairwise distance is improved by removing this edge")

                if is_leaf(mst_copy, a) and not (is_leaf(mst_copy, b)) and (a not in vertex_set) and vertices_covered_by_tree(mst_copy_2,graph) :
                    # print("Removing this edge and node part a")
                    vertex_set.add(b)
                    mst_copy.remove_edge(a, b)
                    mst_copy.remove_node(a)
                elif is_leaf(mst_copy, b) and not (is_leaf(mst_copy, a)) and (b not in vertex_set) and vertices_covered_by_tree(mst_copy_2,graph):
                    # print("Removing this edge and node part b")
                    vertex_set.add(a)
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
        nx.draw(T)
        write_output_file(T, f"{output_dir}/{graph_name}.out")



