# import numpy as np

def generateMatrix(file_name):
    f = open('inputs/' + file_name, 'r')
    size = int(f.readline())
    output = [[0 for i in range(size)] for j in range(size)]
    for line in f:
        edge = line.split(" ")
        output[int(edge[0])][int(edge[1])] = int(edge[2])
        output[int(edge[1])][int(edge[0])] = int(edge[2])
    return output

def greedy_network(file_name):
    cityMatrix = generateMatrix(file_name)
    return cityMatrix

def mst_network(cities):
    return