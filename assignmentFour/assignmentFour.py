# Design & Analysis of Algorithms (CSCI 323)
# Winter Session 2022
# Assignment 4:
# @Andreea Ibanescu


import heapq
import sys # Library for INT_MAX
import time
from os import listdir
from os.path import isfile, join
from copy import deepcopy
from heapq import heappop, heappush


# Function to read the graph from a file


def read_graph(file_name):
    with open(file_name, 'r') as file:
        graph = []
        lines = file.readlines()
        for line in lines:
            costs = line.split(' ')
            row = []
            for cost in costs:
                row.append(int(cost))
            graph.append(row)
        return graph


# Describe the graph: Num of vertices, Num of edges, Symmetric or not


def desc_graph(graph):
    num_vertices = len(graph)
    message = ''
    message += 'Number of vertices = ' + str(num_vertices) + '\n'
    non_zero = 0
    for i in range(num_vertices):
        for j in range(num_vertices):
            if graph[i][j] > 0:
                non_zero += 1
    num_edges = int(non_zero / 2)
    message += 'Number of edges = ' + str(num_edges) + '\n'
    message += 'Symmetric = ' + str(is_symmetric(graph)) + '\n'
    return message


# Function to see if the graph is symmetric or not returns T/F


def is_symmetric(graph):
    num_vertices = len(graph)
    for i in range(num_vertices):
        for j in range(num_vertices):
            if graph[i][j] != graph[j][i]:
                return False
    return True


# this function prints out the graph as it is


def print_graph(graph, sep=' '):
    str_graph = ''
    for row in range(len(graph)):
        str_graph += sep.join([str(c) for c in graph[row]]) + '\n'
    return str_graph


# this function analyzes the graph by its contents. Then applies different algorithms to the graph & calcs its runtime


def analyze_graph(file_name):
    graph = read_graph(file_name)
    output_file_name = file_name[0:-4 + len(file_name)] + '_report.txt'
    with open(output_file_name, 'w') as output_file:

      #  output_file.write('Analysis of graph: ' + file_name + '\n\n')
        str_graph = print_graph(graph)
        output_file.write(str_graph + '\n')
        graph_descrip = desc_graph(graph)
        output_file.write(graph_descrip + '\n')

        start_time_1 = time.time()
        dfs_traversal = dfs(graph)
        end_time_1 = time.time()
        net_time_1 = (end_time_1 - start_time_1)*1000
        output_file.write('\n' +'Dfs Traversal: ' + str(dfs_traversal) +'\n' + "Runtime: " + str(net_time_1) + '\n')

        start_time_2 = time.time()
        bfs_tranversal = bfs(graph)
        end_time_2 = time.time()
        net_time_2 = (end_time_2 - start_time_2)*1000
        output_file.write('\n' +'Bfs Traversal: ' + str(bfs_tranversal) + '\n' + "Runtime: " + str(net_time_2)+'\n')

        start_time_3 = time.time()
        p = prims_mst(graph)
        end_time_3 = time.time()
        net_time_3 = (end_time_3 - start_time_3)*1000
        output_file.write('\n' + 'Prims MST: ' + str(p) +'\n' + "Runtime: " + str(net_time_3) + '\n')

        start_time_4 = time.time()
        vert = len(graph)
        n = graph_to_tuple(graph)
        k = kruskal_mst(n, vert)
        end_time_4 = time.time()
        net_time_4 = (end_time_4 - start_time_4) * 1000

        output_file.write('\n' + 'Kruskal MST: ' + str(k) +'\n' + "Runtime: " + str(net_time_4) + '\n')

        start_time_5 = time.time()
        d = dijkstra(graph)
        end_time_5 = time.time()
        net_time_5 = (end_time_5- start_time_5)*1000
        output_file.write('\n' 'Dijkstras SSSP: ' + '\n' + print_graph(d) + '\n' + "Runtime: " + str(net_time_5) + '\n')

        start_time_6 = time.time()
        f = floyd_pred(graph)
        end_time_6 = time.time()
        net_time_6 = (end_time_6 - start_time_6)*1000
        output_file.write('\n' +'Floyds APSP algorithm' + '\n' + print_graph(f)+'\n' + "Runtime: " + str(net_time_6) + '\n')

        start_time_7 = time.time()
        f_two = floyd_dist(graph)
        end_time_7 = time.time()
        net_time_7 = (end_time_7 - start_time_7)*1000
        output_file.write('\n' +'Floyds dist' + '\n' + print_graph(f_two)+'\n' + "Runtime: " + str(net_time_7) + '\n')


# Depth-First Search


def dfs_util(graph, v, visited):
    visited.append(v)
    for col in range(len(graph[v])):
        if graph[v][col] > 0 and col not in visited:
            dfs_util(graph, col, visited)


def dfs(graph):
    visited = []
    dfs_util(graph, 0, visited)
    return visited


# Breadth-First Search: https://www.geeksforgeeks.org/implementation-of-bfs-using-adjacency-matrix/


def bfs(graph):
    start = 0;
    new = []
    # Visited vector to so that a vertex is not visited more than
    # once Initializing the vector to false as no vertex is visited at the beginning
    visited = [False] * len(graph)
    q = [start]
    # Set source as visited
    visited[start] = True
    while q:
        vis = q[0]
        # current node
        new.append(vis)
        q.pop(0)
        # For every adjacent vertex to the current vertex
        for i in range(len(graph)):
            if (graph[vis][i] > 0 and
                    (not visited[i])):
                # Push the adjacent node in the queue
                q.append(i)
                # set
                visited[i] = True

    return new;


# PRIMS MST : https://www.geeksforgeeks.org/prims-minimum-spanning-tree-mst-greedy-algo-5/


def prims_mst(graph):
    # Prim's Algorithm in Python
    INF = 9999999
    # number of vertices in graph
    N = len(graph)
    selected_node = []
    for i in range(0, N):
        selected_node.append(0)
    no_edge = 0
    selected_node[0] = True
    # printing for edge and weight
    # print("Edge : Weight\n")
    count = 0
    edges = []
    while no_edge < N - 1:
        minimum = INF
        a = 0
        b = 0
        for m in range(N):
            if selected_node[m]:
                for n in range(N):
                    if ((not selected_node[n]) and graph[m][n]):
                        # not in selected and there is an edge
                        if minimum > graph[m][n]:
                            minimum = graph[m][n]
                            a = m
                            b = n
        edges.append((a, b)) # "("+ str(a) + "," + str(b) + ")"
        count += graph[a][b]
        selected_node[b] = True
        no_edge += 1
    return_string = ("edges:", edges, "total cost:", count)
    return edges


def prims_total_cost(graph):
    # Prim's Algorithm in Python
    INF = 9999999
    # number of vertices in graph
    N = len(graph)
    selected_node = []
    for i in range(0, N):
        selected_node.append(0)
    no_edge = 0
    selected_node[0] = True
    # printing for edge and weight
    # print("Edge : Weight\n")
    count = 0
    while (no_edge < N - 1):
        minimum = INF
        a = 0
        b = 0
        for m in range(N):
            if selected_node[m]:
                for n in range(N):
                    if ((not selected_node[n]) and graph[m][n]):
                        # not in selected and there is an edge
                        if minimum > graph[m][n]:
                            minimum = graph[m][n]
                            a = m
                            b = n
        count += graph[a][b]
        selected_node[b] = True
        no_edge += 1
    return count


# Kruskal's MST algorithm

# https://www.techiedelight.com/kruskals-algorithm-for-finding-minimum-spanning-tree/
class DisjointSet:
    parent = {}

    # perform MakeSet operation
    def makeSet(self, n):
        # create n disjoint sets (one for each vertex)
        for i in range(n):
            self.parent[i] = i

    # Find the root of the set in which element k belongs
    def find(self, k):
        # if k is root
        if self.parent[k] == k:
            return k

        # recur for the parent until we find the root
        return self.find(self.parent[k])

    # Perform Union of two subsets
    def union(self, a, b):
        # find the root of the sets in which elements x and y belongs
        x = self.find(a)
        y = self.find(b)

        self.parent[x] = y


def kruskal_mst(edges, n):
    # stores the edges present in MST
    MST = []

    ds = DisjointSet()
    ds.makeSet(n)

    index = 0

    # sort edges by increasing weight
    edges.sort(key=lambda x1: x1[2])

    # MST contains exactly V-1 edges
    while len(MST) != n - 1:

        # consider the next edge with minimum weight from the graph
        (src, dest, weight) = edges[index]
        index = index + 1

        x = ds.find(src)
        y = ds.find(dest)

        if x != y:
            MST.append((src, dest))
            ds.union(x, y)

    return MST

# Dijkstra's SSSP algorithm: https://www.techiedelight.com/single-source-shortest-paths-dijkstras-algorithm/


def dijkstra(graph):
    prev = []
    temp_graph = graph_to_tuple(graph)
    tuple_graph = Graph(temp_graph, len(graph))
    print("Dijkstra's APSP: ")
    for i in range(len(graph)):
        prev.append(find_shortest_paths(tuple_graph, i, len(graph)))
    return prev


def graph_to_tuple(graph):
    list_tuple = []
    for i in range(len(graph)):
        for j in range(len(graph)):
            if graph[i][j] != 0:
                list_tuple.append([i, j, graph[i][j]])
    return list_tuple


def find_shortest_paths(graph, source, n):
    temp_ans = dist = [(n*[0]) for i in range(n)]
    # create a min-heap and push source node having distance 0
    pq = []
    heappush(pq, Node(source))
    # set initial distance from the source to `v` as infinity
    dist = [sys.maxsize] * n
    # distance from the source to itself is zero
    dist[source] = 0
    # list to track vertices for which minimum cost is already found
    done = [False] * n
    done[source] = True
    # stores predecessor of a vertex (to a print path)
    prev = [-1] * n
    # run till min-heap is empty
    while pq:
        node = heappop(pq)  # Remove and return the best vertex
        u = node.vertex  # get the vertex number
        # do for each neighbor `v` of `u`
        for (v, weight) in graph.adjList[u]:
            if not done[v] and (dist[u] + weight) < dist[v]:  # Relaxation step
                dist[v] = dist[u] + weight
                prev[v] = u
                heappush(pq, Node(v, dist[v]))

        # mark vertex `u` as done so it will not get picked up again
        done[u] = True
    route = []
    for i in range(n):
        if i != source and dist[i] != sys.maxsize:
            get_route(prev, i, route)
            print(dist[i], end='\t')
            route.clear()
        else:
             print(0, end='\t')
    return prev


def findshortestpaths_dist(graph, source, n):
    temp_ans = dist = [(n*[0]) for i in range(n)]
    # create a min-heap and push source node having distance 0
    pq = []
    heapq.heappush(pq, Node(source))
    # set initial distance from the source to `v` as infinity
    dist = [sys.maxsize] * n
    # distance from the source to itself is zero
    dist[source] = 0
    # list to track vertices for which minimum cost is already found
    done = [False] * n
    done[source] = True
    # stores predecessor of a vertex (to a print path)
    prev = [-1] * n
    # run till min-heap is empty
    while pq:
        node = heapq.heappop(pq)  # Remove and return the best vertex
        u = node.vertex  # get the vertex number
        # do for each neighbor `v` of `u`
        for (v, weight) in graph.adjList[u]:
            if not done[v] and (dist[u] + weight) < dist[v]:  # Relaxation step
                dist[v] = dist[u] + weight
                prev[v] = u
                heapq.heappush(pq, Node(v, dist[v]))
        # mark vertex `u` as done so it will not get picked up again
        done[u] = True
    route = []
    for i in range(n):
        if i != source and dist[i] != sys.maxsize:
            get_route(prev, i, route)
            #print(dist[i], end='\t')
            route.clear()
        else:
            something = True
            #print(0,end='\t')
    return dist


def graph_to_list(graph): # Used to change graph again to list of tuples gor Dijkstra algo
    list_list = []
    for i in range(len(graph)):
        for j in range(len(graph)):
            if graph[i][j] != 0:
                list_list.append((i, j, graph[i][j]))
    return list_list


# Floyd's APSP algorithm https://www.geeksforgeeks.org/floyd-warshall-algorithm-dp-16/


def floyd_dist(graph):
    v = len(graph)
    dist = [(v*[0]) for i in range(v)]
    pred = [(v*[0]) for i in range(v)]
    # init loop
    for i in range(v):
        for j in range(v):
            dist[i][j] = graph[i][j]  # path of length 1, i.e. just the edge
            pred[i][j] = i  # predecessor will be vertex i
            if dist[i][j] == 0:
                dist[i][j] = sys.maxsize
        dist[i][i] = 0  # no cost
        pred[i][i] = -1  # indicates end of path
    # main loop
    for k in range(v):
        for i in range(v):
            for j in range(v):
                if dist[i][j] > dist[i][k] + dist[k][j]:  # use intermediate vertex k
                    dist[i][j] = dist[i][k] + dist[k][j]
                    pred[i][j] = pred[k][j]
    return dist


def floyd_pred(graph):
    v = len(graph)
    dist = [(v*[0]) for i in range(v)]
    pred = [(v*[0]) for i in range(v)]
    # init loop
    for i in range(v):
        for j in range(v):
            dist[i][j] = graph[i][j]  # path of length 1, i.e. just the edge
            pred[i][j] = i  # predecessor will be vertex i
            if dist[i][j] == 0:
                dist[i][j] = sys.maxsize
        dist[i][i] = 0  # no cost
        pred[i][i] = -1  # indicates end of path
    # main loop
    for k in range(v):
        for i in range(v):
            for j in range(v):
                if dist[i][j] > dist[i][k] + dist[k][j]:  # use intermediate vertex k
                    dist[i][j] = dist[i][k] + dist[k][j]
                    pred[i][j] = pred[k][j]
    # print(dist)
    # print(pred)
    return pred

# Classes from https://www.techiedelight.com/single-source-shortest-paths-dijkstras-algorithm/


class Node:
    def __init__(self, vertex, weight=0):
        self.vertex = vertex
        self.weight = weight

    # Override the __lt__() function to make `Node` class work with a min-heap
    def __lt__(self, other):
        return self.weight < other.weight


class Graph:
    def __init__(self, edges, n):
        # allocate memory for the adjacency list
        self.adjList = [[] for _ in range(n)]

        # add edges to the directed graph
        for (source, dest, weight) in edges:
            self.adjList[source].append((dest, weight))


def get_route(prev, i, route):
    if i >= 0:
        get_route(prev, prev[i], route)
        route.append(i)


def main():
    mypath = "/Users/andreeaibanescu/PycharmProjects/algorithmAndDesign/"
    files = [f for f in listdir(mypath) if isfile(join(mypath, f))]
    for file in files:
        if file[0:5] == 'graph' and file.find('_report') < 0:
            analyze_graph(file)


if __name__ == '__main__':
    main()
