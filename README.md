# Network Connectivity Minimization

This collection of algorithms addresses the issue of connectivity within graphs in respect to building cell towers and minimizing the cost of constructing enough cell towers within the network, such that every city is either a cell-tower or is a direct neighbor of a cell tower. 

Each node in the graph represents a city with outgoing edges as the cost of the fiber cables between the two cities if they were both cell towers. 

Overall these algorithms are looking to minimize the cost of constructing a fiber network, the average pairwise distance between every cell tower constructed in our network. 

$$\frac{1}{{{T}\choose{2}}}\sum_{{u,v }  \in  {{T}\choose{2}}}^{} d(u,v)$$

Valid solutions are acyclic connected dominating sets, in other words trees such that every node in the graph is either in the solution tree or a direct neighbor. 

### Requirements

This program requires `python3` to execute, and to visualize any of the graphs or trees built you will need to have `networkx 2.4` installed.

 

```python
pip install networkx 2.4
```

### Inputs

A collection of sample inputs of varying sizes can be found in the `inputs` directory although you are welcome to construct your own. All valid inputs must follow the following form:

```
inputs/graph-name.in

3               (number of vertecies in the entire graph)
0 3 72.3        (city u and v and the distance between)
1 2 89.1
3 1 1.04    
```

All input weights must be positive and less than zero. 

### Execution

The entire algorithm can be run on all input graphs in the `inputs` directory  with the following 

```python
python3 solver.py test.in
```

This will output the minimum average pair wise distance for each of the graphs within the inputs directory and will write the solution tree to the outputs directory. 

The outputted solution is the minimum of three separate algorithms the input graph will be run with. These three solutions can be found in  `[solver.py](http://solver.py)` , and include two different MST approaches and a Set-Cover/Dijkstras hybrid solution.


### Contributors

This was our final project for Berkeley's Algorithims course CS170 Spring 2020. Our solution was ranked raltive to each input graph against every group within our class. The output rankings can be found on `[the class leader board] (https://berkeley-cs170.github.io/project-leaderboard/)`

- Cassidy Hardin
- Raghav Singh
- Avik Sethia
