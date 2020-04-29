import networkx as nx
from parse import read_input_file, write_output_file
from utils import is_valid_network, average_pairwise_distance
import sys
import os
import matplotlib.pyplot as plt


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


    mst = nx.minimum_spanning_tree(G)
    mst_copy = mst.copy()
    mst_copy_2= mst.copy()
    # Set of vertices that cannot will have to be connected directly to the graph
    #print("THIS IS THE MST")
    vertex_set= set()
    node_to_be_added= None
    for a, b, data in sorted(mst.edges(data=True), key=lambda x: x[2]['weight'],reverse=True):
        #print("DEBUGGER")
        #print(mst_copy.nodes(data=True),mst_copy_2.nodes(data=True))
        #print(mst_copy.edges(data=True), mst_copy_2.edges(data=True))
        #print('{a} {b} {w}'.format(a=a, b=b, w=data['weight']))
        if is_leaf(mst_copy_2,a):
            #print("a is the leaf")
            mst_copy_2.remove_edge(a, b)
            node_to_be_added =a
            mst_copy_2.remove_node(a)

        elif is_leaf(mst_copy_2,b):
            #print("b is the leaf")
            mst_copy_2.remove_edge(a, b)
            node_to_be_added =b
            mst_copy_2.remove_node(b)
        else: #Edges connecting Two non leaf nodes cannot be removed
            continue

        if average_pairwise_distance(mst_copy_2)<average_pairwise_distance(mst_copy):
            #if is_leaf(mst_copy,a) or is_leaf(mst_copy,b):
            #print("Pairwise distance is improved by removing this edge")

            if is_leaf(mst_copy,a) and not(is_leaf(mst_copy,b)) and (a not in vertex_set) :
                #print("Removing this edge and node part a")
                vertex_set.add(b)
                mst_copy.remove_edge(a, b)
                mst_copy.remove_node(a)
            elif is_leaf(mst_copy,b) and not(is_leaf(mst_copy,a))and (b not in vertex_set):
                #print("Removing this edge and node part b")
                vertex_set.add(a)
                mst_copy.remove_edge(a, b)
                mst_copy.remove_node(b)
            else:
                mst_copy_2.add_node(node_to_be_added)
                mst_copy_2.add_edge(a, b,weight=data['weight'])

        else:
            mst_copy_2.add_node(node_to_be_added)

            mst_copy_2.add_edge(a,b, weight=data['weight'])

    mst_copy_3=mst_copy.copy()
    mst_copy_4=mst_copy.copy()
    for a, b, data in sorted(mst_copy.edges(data=True), key=lambda x: x[2]['weight'],reverse=True):
        #print("DEBUGGER")
        #print(mst_copy_3.nodes(data=True),mst_copy_4.nodes(data=True))
        #print(mst_copy_3.edges(data=True), mst_copy_4.edges(data=True))
        #print('{a} {b} {w}'.format(a=a, b=b, w=data['weight']))
        if is_leaf(mst_copy_4,a):
            #print("a is the leaf")
            mst_copy_4.remove_edge(a, b)
            node_to_be_added =a
            mst_copy_4.remove_node(a)

        elif is_leaf(mst_copy_4,b):
            #print("b is the leaf")
            mst_copy_4.remove_edge(a, b)
            node_to_be_added =b
            mst_copy_4.remove_node(b)
        else: #Edges connecting Two non leaf nodes cannot be removed
            continue

        if average_pairwise_distance(mst_copy_2)<average_pairwise_distance(mst_copy):
            #if is_leaf(mst_copy,a) or is_leaf(mst_copy,b):
            #print("Pairwise distance is improved by removing this edge")

            if is_leaf(mst_copy,a) and not(is_leaf(mst_copy,b)) and (a not in vertex_set) :
                #print("Removing this edge and node part a")
                vertex_set.add(b)
                mst_copy_3.remove_edge(a, b)
                mst_copy_3.remove_node(a)
            elif is_leaf(mst_copy,b) and not(is_leaf(mst_copy,a))and (b not in vertex_set):
                #print("Removing this edge and node part b")
                vertex_set.add(a)
                mst_copy_3.remove_edge(a, b)
                mst_copy_3.remove_node(b)
            else:
                mst_copy_4.add_node(node_to_be_added)
                mst_copy_4.add_edge(a, b,weight=data['weight'])

        else:
            mst_copy_4.add_node(node_to_be_added)

            mst_copy_4.add_edge(a,b, weight=data['weight'])
    print("THIS IS THE SOLUTION")
    for a, b, data in sorted(mst_copy.edges(data=True), key=lambda x: x[2]['weight']):
        print('{a} {b} {w}'.format(a=a, b=b, w=data['weight']))
    return mst_copy_3



def is_leaf(graph,node):
    if graph.degree(node)==1:
        return True
    else:
        return False

    # TODO: your code here!



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



