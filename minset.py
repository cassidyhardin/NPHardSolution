"""
CS 170 Project : Spring 2020
Avik Sethia, Cassidy Hardin, Raghav Singh
An attempt to solve an NP-Hard Problem of min-set cover with the added constraint that the cover must be connected.

The algorithm returns a valid solution that is not guaranteed to the be the optimum.
The randomness ensures that the algorithm is not guaranteed to return the same solution on each run.
Therefore, it can run repeatedly and the minimum solution is picked as the optimum.
This is the best approximation of the actual solution.
"""

import parse
import os

def minset(G):
    return None

def isValidSolution():
    """
    1. Check if the towers are connected. (The vertices in the min set cover are connected)
    2. Check if the towers cover all cities (The min set cover is valid)
    """
    return True

def main():
    for input in os.listdir('inputs'):
        if parse.validate_file('inputs/' + input):
            graph = parse.read_input_file('inputs/' + input)
            solution = minset(graph)
            solution = parse.read_output_file(solution, graph)
            outputFilename = 'outputs/' + os.path.splitext(input)[0] + '.out'
            parse.write_output_file(solution, outputFilename)
        else:
            print('Invalid File:', str(input))
