# -*- coding: utf-8 -*-
# Author: Rafael Pagliuca
# Created at: 2016-07-03
# Last modified at: 2016-09-21

from __future__ import division
import networkx as nx
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import random
import os
import scipy.stats
from itertools import permutations
#import plfit
import sys
#import third_party.plpva
import re
import random
import datetime

# Decorator for calculating ellapsed time of function
def timer

# This class is used to generate the same plots available for
# each network of http://konect.uni-koblenz.de
#
# And extra methods required by Professor Francisco Rodrigues
# for Complex Networks course at ICMC-USP 2016
class GraphTools:

    def __init__(self, G):
        self.G = G
        # Initialization of caching for largest component
        self.largest_component = None
        self.configure_matplotlib()

    def configure_matplotlib(self):
        matplotlib.rcParams['axes.color_cycle'] = ['#81CB15']
        matplotlib.rcParams['lines.markeredgewidth'] = 0
        color = '#424242'
        matplotlib.rcParams['text.color'] = color
        matplotlib.rcParams['axes.labelcolor'] = color
        matplotlib.rcParams['axes.edgecolor'] = color 
        matplotlib.rcParams['xtick.color'] = color 
        matplotlib.rcParams['ytick.color'] = color 
        
        # Enables logging
    def enable_logging(self):
        self.logging_enabled = True

    # Disables logging
    def disable_loggin(self):
        self.logging_enabled = False

    # Logging
    def log(self, message):
        if (self.logging_enabled):
            print message

    # Saves or shows plot, depending if filename is passed or not
    def save_or_show(self, filename = False):
        plt.margins(0.1)
        if filename:
            print "Saving to " + filename + "..."
            plt.savefig(filename, format="pdf")
        else:
            plt.show()
        # Clear figure after showing or saving
        plt.clf()
        print "End time: " + str(datetime.datetime.now())
        print ""

    # Saves or prints a message, depending if filename is passed or not
    def save_or_print(self, text, filename = False):
        if filename:
            print "Printing to " + filename + "..."
            print ""
            with open(filename, "w") as textfile:
                textfile.write(str(text))
        else:
            print text

    # Skips function if filename already exists
    def check_if_should_skip(self, filename):
        if filename and os.path.isfile(filename):
            print "File already exists: " + filename
            print "Skipping..."
            print ""
            return True
        else:
            print "Start time: " + str(datetime.datetime.now())

    # Read text file to string
    def read(self, filename):
        with open(filename, "r") as textfile:
            return textfile.read().replace('\n', '')   

    # Plots the degree distribuion, i.e., the total count of nodes (y axis) having x (x axis) connections
    def degree_distribution(self, filename = False):
        self.log("Plotting degree distribution...")
        if self.check_if_should_skip(filename):
            return
        degrees = nx.degree_histogram(self.G)
        plt.loglog(degrees,'o')
        plt.title('Degree distribution')
        plt.xlabel('Degree (k)')
        plt.ylabel('Frequency')
        self.save_or_show(filename);

    # Plots betweenness centrality distribution, i.e., the total count of nodes (y axis) that are on x (x axis) shortest paths
    def betweenness_centrality_distribution(self, filename = False):
        self.log("Plotting betweenness centrality distribution...")
        if self.check_if_should_skip(filename):
            return
        bc = nx.betweenness_centrality(self.G).values()
        bc = np.histogram(bc, bins=30)
        plt.loglog(bc[1][0:-1], bc[0], 'o')
        plt.title('Betweenness Centrality distribution')
        plt.xlabel('Betweenness Centrality')
        plt.ylabel('Frequency')
        self.save_or_show(filename);

    # Plots Eigenvector centrality distribution
    def eigenvector_centrality_distribution(self, filename = False):
        self.log("Plotting eigenvector centrality distribution...")
        if self.check_if_should_skip(filename):
            return
        try:
            bc = nx.eigenvector_centrality(self.G).values()
            bc = np.histogram(bc, bins=30)
            plt.loglog(bc[1][0:-1], bc[0], 'o')
            plt.title('Eigenvector Centrality distribution')
            plt.xlabel('Eigenvector Centrality')
            plt.ylabel('Frequency')
        except Exception, e:
            print 'Failed calculating Eigenvector Centrality.'
            print str(e)
        self.save_or_show(filename);

    # Plots Closeness centrality distribution
    def closeness_centrality_distribution(self, filename = False):
        self.log("Plotting closeness centrality distribution...")
        if self.check_if_should_skip(filename):
            return
        bc = nx.closeness_centrality(self.G).values()
        bc = np.histogram(bc, bins=30)
        plt.loglog(bc[1][0:-1], bc[0], 'o')
        plt.title('Closeness Centrality distribution')
        plt.xlabel('Closeness Centrality')
        plt.ylabel('Frequency')
        self.save_or_show(filename);

    # Plots PageRank distribution
    def pagerank_distribution(self, filename = False):
        self.log("Plotting PageRank distribution...")
        if self.check_if_should_skip(filename):
            return
        bc = nx.pagerank(self.G).values()
        bc = np.histogram(bc, bins=30)
        plt.loglog(bc[1][0:-1], bc[0], 'o')
        plt.title('PageRank distribution')
        plt.xlabel('PageRank')
        plt.ylabel('Frequency')
        self.save_or_show(filename);

    # Plots the cumulative degree distribution, i.e., the probability of a node having more than N (x axis) connections
    def cumulative_degree_distribution(self, filename = False):
        self.log("Plotting cumulative degree distribution...")
        if self.check_if_should_skip(filename):
            return
        degrees = nx.degree_histogram(self.G)
        cumulative = list()
        sum = 0.0
        total = float(nx.number_of_nodes(self.G))
        for freq in degrees:
            sum += float(freq)
            cumulative.append(1-(sum/total))
        plt.loglog(cumulative,'o')
        plt.title('Complementary cumulative degree distribution, P(K>k)')
        plt.xlabel('Degree (k)')
        plt.ylabel('Cumulative frequency')
        self.save_or_show(filename)

    # Plots assortativity, i.e., the likeness of nodes of high degree connecting with other nodes of high degree,
    # and nodes of low degree connecting with other nodes of low degree
    # It is a scatterplot with one data point for every pair (node degree, average neighbor degree)
    def assortativity(self, filename = False):
        self.log("Plotting assortativity...")
        if self.check_if_should_skip(filename):
            return
        node_degrees = nx.degree(self.G).values()
        node_neighbors_degrees = nx.average_neighbor_degree(self.G).values()
        plt.loglog(node_degrees, node_neighbors_degrees, 'o')
        plt.title('Assortativity')
        plt.xlabel('Node degree')
        plt.ylabel('Neighbour degree')
        self.save_or_show(filename)

    # Plots Lorenz curve: the closer to ((0,0), (1,0)) the Lorenz curve is,
    # the more the edges are equally distributed among nodes
    def lorenz_curve(self, filename = False):
        self.log("Plotting Lorenz curve...")
        if self.check_if_should_skip(filename):
            return
        node_degrees = sorted(nx.degree(self.G).values())
        sum = 0
        count = 0
        total = len(node_degrees)
        step = int(total/100.0)
        percentiles = list()
        total_degrees = list()
        for deg in node_degrees:
            sum += deg
            count += 1
            if count % step == 0:
                percentile = count // step # Integer division
                percentiles.append(percentile)
                total_degrees.append(sum)
        plt.plot(percentiles, np.array(total_degrees)*100.0/sum,'-')
        plt.plot([0, 100], [0, 100], '--', color='black')
        plt.plot([0, 100], [100, 0], '--', color='black')
        plt.title('Lorenz curve')
        plt.xlabel('Percentile')
        plt.ylabel('Cumulative degree')
        plt.axes().set_aspect('equal')
        self.save_or_show(filename)

    # Plots the Clustering Coefficient Distribution
    # Percentage of nodes having clustering coefficient less than or equal the value of the x axis
    def clustering_coefficient_distribution(self, filename = False):
        self.log("Plotting clustering coefficient distribution...")
        if self.check_if_should_skip(filename):
            return
        clustering_coefficients = sorted(nx.clustering(self.G).values())
        indices = range(0, len(clustering_coefficients))
        plt.plot(clustering_coefficients, np.array(indices)/len(clustering_coefficients), 'o')
        plt.title('Clustering coefficient distribution')
        plt.xlabel('Clustering coefficient')
        plt.ylabel('Cumulative frequency')
        self.save_or_show(filename)

    # Plots the Distance Distribution up to 20 edges
    def distance_distribution(self):
        self.log("Plotting distance distribution...")
        nodes = self.G.nodes()
        random.shuffle(nodes)
        distances = [0] * 20
        plt.ion()
        plt.show()
        for node in nodes:
            print node
            lengths = nx.single_source_shortest_path_length(self.G, node)
            for id in lengths:
                try:
                    distances[lengths[id]] += 1
                except:
                    pass
            print distances
            print
            cumulative = list()
            sum = 0
            for distance in distances:
                sum += distance
                cumulative.append(sum)

            plt.clf()
            plt.plot(np.array(cumulative)/sum)
            plt.title('Distance distribution')
            plt.xlabel('Distance')
            plt.ylabel('Frequency')
            plt.pause(0.001)

    # Returns the largest component of the network
    def get_largest_component(self):
        self.log('Getting largest component...')
        if self.largest_component:
            return self.largest_component
        else:
            self.largest_component = max(nx.connected_component_subgraphs(self.G), key=len)
            return self.largest_component

    # Prints average degree
    def average_degree(self, filename = False):
        self.log('Printing average degree...')
        if self.check_if_should_skip(filename):
            return
        degree_sum = sum(self.G.degree().values())
        number_of_nodes = nx.number_of_nodes(self.G)
        self.save_or_print(degree_sum/number_of_nodes, filename)

    # Prints local clustering coefficient average
    def local_clustering_coefficient_average(self, filename = False):
        self.log('Printing local clustering coefficient average...')
        if self.check_if_should_skip(filename):
            return
        self.save_or_print(nx.average_clustering(self.G), filename)
    
    # Prints global transitivity
    def global_transitivity(self, filename = False):
        self.log('Printing global transitivity...')
        if self.check_if_should_skip(filename):
            return
        self.save_or_print(nx.transitivity(self.G), filename)

    # Prints average shortest path length
    def average_shortest_path_length(self, filename = False):
        self.log('Printing average shortest path length...')
        if self.check_if_should_skip(filename):
            return
        self.save_or_print(nx.average_shortest_path_length(self.get_largest_component()), filename)

    # Prints global efficiency
    def global_efficiency(self, filename = False):
        self.log('Printing global efficiency...')
        if self.check_if_should_skip(filename):
            return
        G = self.G
        # Copied from NetworkX source code
        # Source: https://networkx.readthedocs.io/en/latest/_modules/networkx/algorithms/efficiency.html#global_efficiency
        n = len(G)
        denom = n * (n - 1)
        efficiency = sum( self.efficiency(G, u, v) for u, v in permutations(G, 2) ) / denom
        self.save_or_print(efficiency, filename)

    # Calculates efficiency between two nodes
    # Source: https://networkx.readthedocs.io/en/latest/_modules/networkx/algorithms/efficiency.html#global_efficiency
    def efficiency(self, G, u, v):
        try:
            return 1 / nx.shortest_path_length(G, u, v)
        except:
            return 0


    # Prints diameter
    def diameter(self, filename = False):
        self.log('Printing diameter...')
        if self.check_if_should_skip(filename):
            return
        self.save_or_print(nx.diameter(self.get_largest_component()), filename)

    # Plots k(i) vs. cc(i), i.e., scatterplot of degree vs. local clustering coefficient for each node
    def cci_vs_ki(self, filename = False):
        self.log("Plotting k(i) vs. cc(i)...")
        if self.check_if_should_skip(filename):
            return
        clustering_coefficients = nx.clustering(self.G).values()
        degrees = nx.degree(self.G).values()
        plt.plot(degrees, clustering_coefficients, 'o')
        plt.title('Local clustering coefficient vs. degree (cc(i) vs. k(i))')
        plt.xlabel('Degree (k)')
        plt.ylabel('Local clustering coefficient (cc)')
        self.save_or_show(filename)

    # Prints Pearson correlation coefficient between cc(i) and k(i)
    def cci_vs_ki_pearson_correlation_coefficient(self, filename = False):
        self.log("Printing k(i) vs. cc(i) Pearson correlation coefficient...")
        if self.check_if_should_skip(filename):
            return
        clustering_coefficients = nx.clustering(self.G).values()
        degrees = nx.degree(self.G).values()
        pearson_coefficient = scipy.stats.pearsonr(degrees, clustering_coefficients)
        self.save_or_print(pearson_coefficient, filename)

    def cc_vs_k(self):
        # Histogram with all the degrees
        degrees_histogram = nx.degree_histogram(self.G)
        # Empty list with the same size of the histogram
        cc = [0.0] * len(degrees_histogram)
        # Get CC and K for every node
        nodes_clustering_coefficients = nx.clustering(self.G).values()
        nodes_degrees = nx.degree(self.G).values()
        # Cumulative CC
        for node, degree in enumerate(nodes_degrees):
            cc[degree] += nodes_clustering_coefficients[node] 
        # Get average of accumulated CC
        valid_cc = []
        valid_degrees = []
        for degree, cc_sum in enumerate(cc):
            if degrees_histogram[degree] > 0:
                valid_cc.append(cc_sum/degrees_histogram[degree])
                valid_degrees.append(degree)
        return [valid_cc, valid_degrees]

    # Plots cc(i) vs. k(i), i.e., similar to the previous plot, but with averages and inverted axis
    def cc_vs_k_plot(self, filename = False):
        self.log("Plotting cc(k) vs. k...")
        if self.check_if_should_skip(filename):
            return
        cc, degrees = self.cc_vs_k()
        plt.plot(degrees, cc, 'o')
        plt.title('Average local clustering coefficient vs. degree (cc(k) vs. k)')
        plt.xlabel('Degree (k)')
        plt.ylabel('Average local clustering coefficient (cc)')
        self.save_or_show(filename)

    # Prints Pearson correlation coefficient between cc(k) and k
    # This is similar to the cci_vs_ki pearson, but takes the average for each k
    def cc_vs_k_pearson_correlation_coefficient(self, filename = False):
        self.log("Printing cc(k) vs. k Pearson correlation coefficient...")
        if self.check_if_should_skip(filename):
            return
        cc, degrees = self.cc_vs_k()
        pearson_coefficient = scipy.stats.pearsonr(degrees, cc)
        self.save_or_print(pearson_coefficient, filename)
    
    # Get assortativity arrays for each node
    def knni_vs_ki(self):
        knn_i = nx.average_neighbor_degree(self.G).values()
        degrees = nx.degree(self.G).values()
        return [knn_i, degrees]

    # Get average assortativity for each degree
    def knn_vs_k(self):
        # Histogram with all the degrees
        degrees_histogram = nx.degree_histogram(self.G)
        # Empty list with the same size of the histogram
        knn_k = [0.0] * len(degrees_histogram)
        # Get values for each node
        knn_i, degrees = self.knni_vs_ki()
        # Cumulative CC
        for node, degree in enumerate(degrees):
            knn_k[degree] += knn_i[node] 
        # Get average of accumulated CC
        valid_knn = []
        valid_degrees = []
        for degree, knn_sum in enumerate(knn_k):
            if degrees_histogram[degree] > 0:
                valid_knn.append(knn_sum/degrees_histogram[degree])
                valid_degrees.append(degree)
        return [valid_knn, valid_degrees]

    # Plots k_nn(i) vs. k(i), i.e., assortavity
    @timer
    def knni_vs_ki_plot(self, filename = False):
        self.log("Plotting k_nn(i) vs. k(i)...")
        if self.check_if_should_skip(filename):
            return
        knn_i, degrees = self.knni_vs_ki()
        plt.plot(degrees, knn_i, 'o')
        plt.title('Assortativity - k_nn(i) vs. k(i)')
        plt.xlabel('Degree (k(i))')
        plt.ylabel('Average neighbor degree (k_nn(i))')
        self.save_or_show(filename)

    # Plots k_nn(k) vs. k, i.e., assortavity
    def knn_vs_k_plot(self, filename = False):
        self.log("Plotting k_nn(k) vs. k...")
        if self.check_if_should_skip(filename):
            return
        knn, k = self.knn_vs_k()
        plt.plot(k, knn, 'o')
        plt.title('k_nn(k) vs. k')
        plt.xlabel('Degree (k)')
        plt.ylabel('Average neighbor degree (k_nn(k))')
        self.save_or_show(filename)

    # Prints assortativity coefficient
    def knn_vs_k_assortativity_coefficient(self, filename = False):
        self.log("Printing knn(k) vs. k assortativity coefficient...")
        if self.check_if_should_skip(filename):
            return
        assortativity_coefficient = nx.degree_assortativity_coefficient(self.G)
        self.save_or_print(assortativity_coefficient, filename)

    # Prints assortativity coefficient (own implementation)
    def knn_vs_k_assortativity_coefficient_alternative(self, filename = False):
        self.log("Printing knn(k) vs. k assortativity coefficient (own implementation)...")
        if self.check_if_should_skip(filename):
            return
        A = nx.adjacency_matrix(self.G)
        nodes_degrees = nx.degree(self.G).values()
        number_of_edges = self.G.number_of_edges()
        numerator = 0
        denominator = 0
        for i, degree_i in enumerate(nodes_degrees):
            for j, degree_j in enumerate(nodes_degrees):
                numerator += (A[i,j] - degree_i*degree_j/number_of_edges)*degree_i*degree_j
                if i == j:
                    denominator += (degree_i - degree_i*degree_j/number_of_edges)*degree_i*degree_j
        assortativity_coefficient = numerator/denominator
        self.save_or_print(assortativity_coefficient, filename)

    # Prints Pearson correlation coefficient between knn(k) and k
    def knn_vs_k_pearson_correlation_coefficient(self, filename = False):
        self.log("Printing knn(k) vs. k Pearson correlation coefficient...")
        if self.check_if_should_skip(filename):
            return
        knn, k = self.knn_vs_k()
        r = scipy.stats.pearsonr(k, knn)
        self.save_or_print(r, filename)

    # Prints Spearman correlation coefficient between knn(k) and k
    def knn_vs_k_spearman_correlation_coefficient(self, filename = False):
        self.log("Printing knn(k) vs. k Spearman correlation coefficient...")
        if self.check_if_should_skip(filename):
            return
        knn, k = self.knn_vs_k()
        r = scipy.stats.spearmanr(k, knn)
        self.save_or_print(r, filename)


    # Prints the Shannon entropy for the whole network
    def shannon_entropy(self, filename = False):
        print("Printing Shannon entropy...")
        if self.check_if_should_skip(filename):
            return
        total_nodes = float(nx.number_of_nodes(self.G))
        degrees = nx.degree_histogram(self.G)
        sum = 0
        for degree,frequency in enumerate(degrees):
            prob = frequency/total_nodes
            if prob > 0:
                sum += prob * np.log2(prob)
        entropy = -sum
        self.save_or_print(entropy, filename)

    # Prints the moment of degree of an arbitrary order
    def degree_moment(self, moment_order, filename = False):
        print("Printing moment of degree...")
        if self.check_if_should_skip(filename):
            return
        total_nodes = float(nx.number_of_nodes(self.G))
        degrees = nx.degree_histogram(self.G)
        sum = 0
        for degree,frequency in enumerate(degrees):
            prob = frequency/total_nodes
            sum += degree**moment_order * prob
        moment = sum
        self.save_or_print(moment, filename)

    # Prints power law parameters (gamma and k_min)
    def power_law_parameters(self, filename = False):
        print("Printing power law paremeters...")
        if self.check_if_should_skip(filename):
            # If file exists, loads parameters from file
            try:
                params_str = self.read(filename)
                gamma = float(re.search(r'gamma = ([0-9.]+)', params_str).group(1))
                kmin = int(re.search(r'kmin = ([0-9]+)', params_str).group(1))
                self.fit_params = {'gamma': gamma, 'kmin': kmin}
                return
            except:
                print "Error loading parameters from " + filename + "."
                print "Regenerating file..."
        node_degrees = np.asarray(nx.degree(self.G).values())
        fit = plfit.plfit(node_degrees, verbose=False)
        self.fit_params = {'gamma': fit._alpha, 'kmin': fit._xmin}
        output = 'gamma = ' + str(fit._alpha) + ', kmin = ' + str(fit._xmin)
        self.save_or_print(output, filename)

    # Calculates confidence (p-value) of power law fitting
    def power_law_confidence(self, filename = False):
        total = float(nx.number_of_nodes(self.G))
        node_degrees = nx.degree(self.G).values()
        third_party.plpva.plpva(node_degrees, self.fit_params['kmin'], 'reps', 2)
    
    # Plots scatterplot of degree distribution vs betweenness centrality
    def plot_degree_vs_betweenness(self, filename = False):
        self.log("Plotting scatterplot of degree distribution vs. betweenness centrality...")
        if False and self.check_if_should_skip(filename):
            return       
        degrees = nx.degree(self.G).values()     
        bc = nx.betweenness_centrality(self.G).values()
        # Plot frequency of BC vs. frequency of degrees
        plt.loglog(bc, degrees, 'o')
        plt.title('Degrees vs betweenness centrality')
        plt.xlabel('Betweeness centrality')
        plt.ylabel('Degree')
        self.save_or_show(filename);
        
    # Plots scatterplot of degree distribution vs betweenness centrality
    def print_degree_vs_betweenness_correlation(self, filename = False):
        self.log("Printing correlation between degree distribution and betweenness centrality...")
        if False and self.check_if_should_skip(filename):
            return       
        degrees = nx.degree(self.G).values()
        bc = nx.betweenness_centrality(self.G).values()
        pearson_coefficient = scipy.stats.pearsonr(bc, degrees)
        self.save_or_print(pearson_coefficient, filename)
        
        # Plots scatterplot of degree distribution vs betweenness centrality
    def plot_pagerank_vs_closeness(self, filename = False):
        self.log("Plotting scatterplot of PageRank vs. Closeness...")
        if False and self.check_if_should_skip(filename):
            return
        p = nx.pagerank(self.G).values()
        c = nx.closeness_centrality(self.G).values()
        # Plot scatterplot of frequencies
        plt.loglog(c, p, 'o')
        plt.title('PageRank vs. closeness centrality')
        plt.xlabel('Closeness centrality')
        plt.ylabel('PageRank')
        self.save_or_show(filename);
        
    # Plots scatterplot of degree distribution vs betweenness centrality
    def print_pagerank_vs_closeness_correlation(self, filename = False):
        self.log("Printing correlation between PageRank and closeness centrality...")
        if False and self.check_if_should_skip(filename):
            return   
        p = nx.pagerank(self.G).values()
        c = nx.closeness_centrality(self.G).values()
        pearson_coefficient = scipy.stats.pearsonr(c, p)
        self.save_or_print(pearson_coefficient, filename)
        
    # Random Walk
    # @param int L Number of steps
    def plot_random_walk_visits_vs_degree(self, filename = False, L = 1000, N = 10):
        '''
        1) Sortear aleatoriamente qualquer nó
        2) Obter lista de vizinhos do nó sorteado
        3) Sortear aleatoriamente um vizinho
        4) Parar se tiver chegado no passo 1000, senão, voltar para o passo 2
        '''
        self.log("Random Walk...")
        if self.check_if_should_skip(filename):
            return
        nodes = self.G.nodes()
        
        results = []
        
        # Repeat experiment N times
        for _ in range (0, N):
            # Empty list with the size of the number of nodes
            number_of_visits = [0.0] * len(nodes)
            current_node = random.choice(nodes)
            for _ in range (0, L):
                number_of_visits[nodes.index(current_node)] += 1
                current_node = random.choice(self.G.neighbors(current_node))
            results.append(number_of_visits)
            
        # Convert list of lists into numpy object
        np_results = np.array(results)
        visits_mean = np.mean(np_results, 0)
        visits_std_deviation = np.std(np_results, 0)
            
        # Plot number of visits vs. degree
        plt.plot(nx.degree(self.G).values(), visits_mean, 'o')
        #plt.errorbar(nx.eigenvector_centrality(self.G).values(), visits_mean, marker = 'o', fmt = None, yerr = visits_std_deviation)
        plt.title('Random walk visits vs. degree')
        plt.xlabel('Degree')
        plt.ylabel('Random walk visits')
        self.save_or_show(filename)
        
    # Random Walk
    # @param int L Number of steps
    def plot_random_walk_visits_vs_eigenvector_centrality(self, filename = False, L = 1000, N = 10):
        '''
        1) Sortear aleatoriamente qualquer nó
        2) Obter lista de vizinhos do nó sorteado
        3) Sortear aleatoriamente um vizinho
        4) Parar se tiver chegado no passo 1000, senão, voltar para o passo 2
        '''
        self.log("Random Walk...")
        if self.check_if_should_skip(filename):
            return
        nodes = self.G.nodes()
        
        results = []
        
        # Repeat experiment N times
        for _ in range (0, N):
            # Empty list with the size of the number of nodes
            number_of_visits = [0.0] * len(nodes)
            current_node = random.choice(nodes)
            for _ in range (0, L):
                number_of_visits[nodes.index(current_node)] += 1
                current_node = random.choice(self.G.neighbors(current_node))
            results.append(number_of_visits)
            
        # Convert list of lists into numpy object
        np_results = np.array(results)
        visits_mean = np.mean(np_results, 0)
        visits_std_deviation = np.std(np_results, 0)
            
        # Plot number of visits vs. degree
        try:
            plt.plot(nx.eigenvector_centrality(self.G).values(), visits_mean, 'o')
            #plt.errorbar(nx.eigenvector_centrality(self.G).values(), visits_mean, marker = 'o', fmt = None, yerr = visits_std_deviation)
            plt.title('Random walk visits vs. eigenvector centrality')
            plt.xlabel('Eigenvector centrality')
            plt.ylabel('Random walk visits')
        except Exception, e:
            print "Error calculating eigenvector centrality"
            print str(e)
        self.save_or_show(filename)
        

