"""
Author: Mayur Sunil Jawalkar (mj8628)
        Kunjan Suresh Mhaske (km1556)
Big Data Analytics: Homework-07
Description: In this assignment we are trying to perform agglomerative clustering.
             We are also printing the dendograms to visualize the results of the clustering.
"""

# Import packages
import pandas as pd  # To handle the dataframes
import numpy as np  # to work with data
from matplotlib import pyplot as plt  # To plot the graphs
from scipy.cluster.hierarchy import dendrogram, linkage  # To plot dendogram
import math  # For mathematical computations.
import time  # To test how much time it takes.
import copy  # to make deep copy of objects


# Global input data to access everywhere
DATA = None


class Clusters:
    """
    This class implements a data structure to store the cluster information.
    """
    global DATA  # Access the global data

    # Variable used in the class
    __slots__ = 'record_ids', 'center', 'prototype'

    def __init__(self, id=None, cluster1=None, cluster2=None):
        """
        Initialize the cluster.
        """
        self.prototype = 0  # Initialize prototype to Zero

        # if only id is passed, create new cluster with a given data point
        if id is not None and cluster1 is None and cluster2 is None:
            self.record_ids = [id]  # initialize the record_ids in this cluster
            self.center = DATA.iloc[id].astype(float)  # initialize the center of this cluster with the record value.
            self.center = self.center.drop(['ID'])  # remove the id from the cluster center.
        # if 2 clusters are given merge to clusters and generate a new cluster.
        elif cluster1 is not None and cluster2 is not None:
            # Verify the type of two clusters.
            if type(cluster1) is Clusters and type(cluster2) is Clusters:
                # Initialize and merge the records from both clusters
                self.record_ids = cluster1.record_ids.copy()
                self.record_ids.extend(cluster2.record_ids)

                # Initialize the center for a new cluster
                self.center = cluster1.center.copy()
                # print(cluster1.record_ids, cluster2.record_ids, len(cluster1.record_ids), len(cluster2.record_ids))

                clust_1_rec_len = len(cluster1.record_ids)  # total records in cluster 1
                clust_2_rec_len = len(cluster2.record_ids)  # total records in cluster 2
                # iterate over each parameter of the cluster and compute the new centroid.
                for param in list(cluster1.center.index):
                    # calculate the center.
                    self.center[param] = (((cluster1.center[param] * clust_1_rec_len) +
                                           (cluster2.center[param] * clust_2_rec_len)) /
                                          (clust_1_rec_len + clust_2_rec_len))

    def __str__(self):
        """
        Returns the string representation of the cluster.
        """
        return 'Total Records : ' + str(len(self.record_ids)) + ' ||=> ' + 'Center = ' + str(list(self.center))


class Agglomeration:
    """
    This class implements the functionality of agglomeration. It identifies clusters from the input data
    and also plots the dendograms for the same.
    """

    # Variables to be used in the class.
    __slots__ = 'clusters', 'distances', 'merge_info', 'last_six'

    def __init__(self):
        """
        Initialize all variable for this class.
        """
        self.clusters = dict()  # Dictionary to store all clusters.
        self.merge_info = []  # List to store the information of all merges.
        self.last_six = None  # Dictionary to store the last 6 clusters.
        # Distance matrix to store all distances.
        self.distances = pd.DataFrame(99999, index=DATA.index, columns=DATA.index, dtype=float)
        # print(self.distances, self.distances.shape)
        # print(len(self.distances), len(self.distances[0]), print(self.distances))

    def generate_initial_clusters(self):
        """
        Generates the initial clusters for all records in the input data.
        """

        # Iterate over all records from the data and create the new cluster for each one of them.
        for record_ind in range(len(DATA)):
            cluster = Clusters(id=record_ind)   # Create object of cluster
            # print(cluster)
            self.clusters[record_ind] = cluster  # Add the new cluster to the cluster dictionary.

    def compute_distance_matrix(self):
        """
        Computes the distances between all pairs of the data. It stores these distances in the distance matrix.
        """
        clusters_list = self.clusters.keys()  # Extract the list of keys of all clusters.
        # Iterate over all pairs in the clusters
        for clust_it_1 in range(len(clusters_list)):
            for clust_it_2 in range(clust_it_1 + 1, len(clusters_list)):
                # Ignore the distance to itself.
                if clust_it_1 == clust_it_2:
                    continue
                else:
                    # Compute the distance between all pairs
                    distance = self.eucledian_distance(self.clusters[clust_it_1].center,
                                                       self.clusters[clust_it_2].center)
                    # Update the distances in distance matrix.
                    self.distances[clust_it_1][clust_it_2] = distance
                    self.distances[clust_it_2][clust_it_1] = distance

    def update_distance_matrix(self, clust1_ID, clust2_ID):
        """
        Update the distances after merging the given pair of clusters.
        It stores the updated values at lower index clusterId and deletes the record for heigher index clusterID.
        """
        new_clust_id = min(clust1_ID, clust2_ID)  # find the smallest index
        # drop the heigher index cluster column
        self.distances = self.distances.drop(columns=[max(clust2_ID, clust1_ID)])
        # drop the heigher index cluster row
        self.distances = self.distances.drop([max(clust2_ID, clust1_ID)])

        # iterate over all keys in the cluster dictionary
        for cluster_it in self.clusters.keys():
            # ignore the distance to itself
            if cluster_it == new_clust_id:
                continue
            else:
                # Compute new distances from given cluster to all other clusters.
                distance = self.eucledian_distance(self.clusters[cluster_it].center,
                                                   self.clusters[new_clust_id].center)
                self.distances[cluster_it][new_clust_id] = distance
                self.distances[new_clust_id][cluster_it] = distance

    def agglomerate_clusters(self):
        """"
        This function creates the clusters using the given dataframe.
        """
        print("Generating individual clusters.")
        self.generate_initial_clusters()  # Generate all clusters
        print("Generated individual clusters.\n Computing distances.")
        startt = time.time()  # note the start time
        self.compute_distance_matrix()  # calculate all distances
        endt = time.time()  # note the end time
        print("Done computing distances, took {:.3f} seconds".format(endt-startt))
        # iterate until all clusters are merged into one.
        while len(self.clusters.keys()) > 1:
            # Identify the smallest distance and the pair of clusters with smallest distance
            cluster_1_id = self.distances.min().idxmin()
            cluster_2_id = self.distances[cluster_1_id].idxmin()

            # Create a new merged cluster
            merged_cluster = Clusters(cluster1=self.clusters[cluster_1_id],
                                      cluster2=self.clusters[cluster_2_id])

            # Update the merge info
            self.merge_info.append([copy.deepcopy(merged_cluster), copy.deepcopy(self.clusters[cluster_1_id]),
                                    copy.deepcopy(self.clusters[cluster_2_id]), cluster_1_id, cluster_2_id
                                       , self.distances[cluster_2_id][cluster_1_id]])

            # Delete the record with higher clusterId from cluster dictionary
            self.clusters.pop(max(cluster_2_id, cluster_1_id), None)
            # Update the record with lower clusterId from cluster dictionary with merged cluster
            self.clusters[min(cluster_2_id, cluster_1_id)] = merged_cluster

            # update distances after merging cluster1 and cluster2
            self.update_distance_matrix(cluster_1_id, cluster_2_id)

            # Note the last 6 clusters
            if len(self.clusters) == 6:
                self.last_six = self.clusters.copy()

    def eucledian_distance(self, clust_center_1, clust_center_2):
        """
        Computes the euclidean distance between the given set of points.
        """
        SSE = 0.0
        # Iterate over each parameter to compute the squared sum of the differences.
        for param in list(clust_center_1.index):
            SSE += (clust_center_1[param] - clust_center_2[param])**2
        return math.sqrt(SSE)

    def plot_dendograms(self):
        """
        This function plots the dendogram.
        """
        DATA_without_ID = DATA.drop(columns=['ID'])  # remove the ID from data
        link = linkage(DATA_without_ID, 'ward')  # Compute the linkage
        dendrogram(link, truncate_mode='lastp', p=12, show_contracted=True)  # Plot dendograms
        plt.title("Dendogram Output")  # Assign name of plot
        plt.xlabel("Cluster Size")  # X label of plot
        plt.ylabel("Distance")  # Y label of plot
        plt.axhline(y=100)  # Line where we are marking to consider clusters.
        plt.show()

    def calculate_prototype(self, clusters):
        """
        Calculate the prototype for given set of clusters.
        """
        for clusterID in clusters.keys():
            clusters[clusterID].prototype = 0
            distance = 0
            for id in clusters[clusterID].record_ids:
                distance += self.eucledian_distance(clusters[clusterID].center, DATA.iloc[id])
            clusters[clusterID].prototype = distance/len(clusters[clusterID].record_ids)


def cross_correlation_plot(cross_correlation, data_without_id):
    """"""
    fig = plt.figure()
    ax = fig.add_subplot(111)
    cax = ax.matshow(cross_correlation, cmap='coolwarm', vmin=-1, vmax=1)
    fig.colorbar(cax)
    ticks = np.arange(0, len(data_without_id.columns), 1)
    ax.set_xticks(ticks)
    plt.xticks(rotation=90)
    ax.set_yticks(ticks)
    ax.set_xticklabels(data_without_id.columns)
    ax.set_yticklabels(data_without_id.columns)
    plt.show()


def main():
    # Consider global DATA variable to load the data
    global DATA
    DATA = pd.read_csv("HW_PCA_SHOPPING_CART_v895.csv")  # Read the data
    # DATA = pd.read_csv("Test.csv")

    # Remove the id from the data
    data_without_id = DATA.drop(columns=['ID'])
    print("Cross Correlation of the given data is : ")
    cross_correlation = data_without_id.corr()  # Compute the cross correlation
    print(cross_correlation)
    # Plot the cross correlation graph
    cross_correlation_plot(cross_correlation, data_without_id)

    # Create object of Agglomeration class.
    aglomeration = Agglomeration()

    start = time.time()  # Note the start time
    # Create and agglomerate the clusters
    aglomeration.agglomerate_clusters()
    # Note the end time
    print("Done... time = {:.2f seconds}", time.time()-start)

    # Print last 20 merges information
    print("Last 20 merges")
    for mergeinfo in aglomeration.merge_info[-20:]:
        clusterID = min(mergeinfo[3], mergeinfo[4])  # id of merged cluster
        size = len(mergeinfo[0].record_ids)  # Size of the merged cluster
        # Childs of the merged cluster
        child_1_id = mergeinfo[3]
        child_2_id = mergeinfo[4]
        # Minimum size between the child. size of the smallest child
        smallest_size = min(len(mergeinfo[1].record_ids), len(mergeinfo[2].record_ids))
        print("ClusterID = {:3d}, Size = {:3d}, Child_1_ID = {:3d}, Child_2_ID = {:3d}, Smallest Merged "
              "Cluster Size = {:3d}".format(clusterID, size, child_1_id, child_2_id, smallest_size))

    # calculate the prototype of the last 6 clusters
    aglomeration.calculate_prototype(aglomeration.last_six)
    print("\n\nSize of last 6 clusters:")
    size_of_last_6 = []
    # Iterate over last 6 clusters and extract necessary information to print
    for clusterID, clusterInfo in aglomeration.last_six.items():
        size_of_last_6.append([clusterID, len(clusterInfo.record_ids), clusterInfo.prototype, clusterInfo.center])

    # Sort the clusters according to their sizes.
    sorted_last6 = sorted(size_of_last_6, key=lambda x: x[1])
    # Create Dataframe to store centers of last 6 clusters
    dataframe = pd.DataFrame(0, columns=list(data_without_id.columns), index=[])
    print(dataframe)
    # Print all columns
    pd.set_option('display.max_columns', None)

    # Print the information about the last 6 clusters.
    for record in sorted_last6:
        print("ID = {:3d}, Size = {:3d}, Prototype = {:3.2f} ||=> Center = {}"
              "".format(record[0], record[1], record[2], record[3]))
        df = pd.concat([dataframe, pd.DataFrame([record[3]], index=[record[0]])])
    # Print center infrormation
    print(dataframe)
    # PLot dendogram
    aglomeration.plot_dendograms()


# Execute as a script
if __name__ == '__main__':
    main()

