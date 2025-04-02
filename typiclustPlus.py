import torchvision 
import torch 
import torch.nn as nn
import numpy as np 
from tqdm import tqdm
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import KMeans, MiniBatchKMeans
import pandas as pd 
import os 
from simclr_pretrained.resnet_cifar import resnet18

class TypiclustPlus:
    def __init__(self, train_dataset, feat_loader, budget_size, l0_indices, *args):
        """
        A Pytorch implementation of modified Typiclust variant TPC RP utilising a pre-trained 
        SimCLR and K-means. 

        Args: 
        train_dataset (torchvision.datasets.CIFAR10): CIFAR10 training dataset

        feat_loader (torch.utils.data.DataLoader): CIFAR10 Training dataset dataloader
        for generating feature representations. 

        budget_size (int): Sample size 

        l0_indices List[int]: Indices of the initial set of labelled data points 

        args Dict[str -> str,int]: Training parameters such as epochs, lr, etc...
        """
        # Model Parameters 
        if args:
            self.args = args[0] 
            assert isinstance(self.args, dict), "TypiclustPlus model parameters must be Dict."
        else:
            # Default argument
            self.args = {
                'data_dir' : './datasets', 
                'dataset_name' : 'cifar10',
                'device' : torch.device('cuda'),
                'alpha' : 1.0,
                'beta' : 1.0,
                'gamma' : 0.01,
            }
        
        # Max number of clusters for K-Means 
        self.MAX_NUM_CLUSTER = 500 
        # K nearest neighbours 
        self.K_NN = 20 
        # Pre-trained ResNet18 model path 
        self.PT_MODEL_PATH = "simclr_pretrained/simclr_cifar-10.pth"
        
        # Get List of indices pertaining to original CIFAR10 
        self.orig_ids = [i for i in range(len(train_dataset))]

        # Original CIFAR10 ds (non-partitioned)
        self.train_dataset = train_dataset
        self.feat_loader = feat_loader

        # Indices of images in the L-Set
        self.l_set_indices = l0_indices

        # Initial L-set and U-set 
        self.l_set = torch.utils.data.Subset(self.train_dataset,  l0_indices)
        self.u_set = torch.utils.data.Subset(self.train_dataset, list(set(self.orig_ids) - set(l0_indices)))

        # Generate ResNet18 pre-trained feature representation 
        self.feats = self.init_feat_representation(feat_loader)

        self.budget_size = budget_size 

    def init_feat_representation(self, feat_loader):
        """
        Generate feature representations from pre-trained CIFAR10 SIMCLR with
        a resnet18 backbone.

        Args: 
        feat_loader (torch.utils.data.DataLoader): CIFAR10 Training dataset dataloader
        for generating feature representations. 

        Returns: 
        feats (numpy.array): Feature representation of dataset from SimCLR 
        """
        feat_path = f'{self.args['data_dir']}/feats.npy'
        if os.path.isfile(feat_path):
            # If features already have been generated and saved locally 
            print("Loaded features in locally")
        else:
            # Load SimCLR Resnet18 model backbone 
            pt_model = resnet18()["backbone"]

            # Load Pre-train SimCLR
            state_dict = torch.load(self.PT_MODEL_PATH)
            new_state_dict = {}

            # Remove projection head from pre-trained weights
            for key, value in state_dict.items():
                # Reformat 
                if key.startswith("backbone."):
                    new_key = key.replace("backbone.", "")
                    new_state_dict[new_key] = value
                # Ignore projection head 
                if key.startswith("contrastive_head."):
                    continue
            
            # Apply pre-trained backbone weights to resnet18 
            pt_model.load_state_dict(new_state_dict,strict=True)
            pt_model.to(self.args['device'])

            # Generate feature representations
            feats, labels = [], []
            for batch_imgs, batch_labels in tqdm(feat_loader):
                batch_imgs = batch_imgs.to(self.args['device'])
                batch_feats = pt_model(batch_imgs)
                feats.append(batch_feats.detach().cpu())
                labels.append(batch_labels)

            feats = torch.cat(feats, dim=0)
            
            # Save representations locally 
            with open(feat_path, 'wb') as f: 
                np.save(f, feats)
        # Load 
        feats = np.load(feat_path)
        return feats 

    def get_uncovered_clusters(self, feats, n_clusters):
        """
        Apply K-means to the given feature representations and sort clusters into largest 
        uncovered clusters. 

        Largely completes Step 2 of TypiClust

        Args: 
        feats (np.array) : feature representations 
        n_clusters (int) : number of clusters for k-means 
        """
        # Apply k-means to feature representations for n clusters 
        self.clusters, self.cluster_centres = self.k_means(feats, n_clusters)

        # Get size of each cluster and the indices of the points associated to a cluster
        self.cluster_sizes, self.cluster_indices = self.retrieve_cluster_data(self.clusters)
        self.cluster_ids = [i for i in range(len(self.cluster_sizes))]

        # Retrieve how many labelled points each cluster contains 
        self.labelled_cluster_num = np.bincount(self.clusters[self.l_set_indices], minlength=len(self.cluster_ids))

        # Construct cluster DF 
        self.cluster_data = pd.DataFrame({
            'cluster_id': self.cluster_ids,
            'labelled_count' : self.labelled_cluster_num,
            'cluster_size' : self.cluster_sizes,
        })

        # Sort by largest cluster size and clusters with least labelled points 
        self.cluster_data = self.cluster_data.sort_values(by=['labelled_count', 'cluster_size'],
                         ascending=[True, False],
                         ignore_index=True)
        
        # self.cluster_data = self.cluster_data[self.cluster_data["labelled_count"] == 0]
        print(self.cluster_data)

    def k_means(self, feats, n_clusters):
        """
        Perform K-means based on feature representations. 
        Following the implementations of the Typiclust paper, when number of 
        cluster exceeds 50, MiniBatchKMeans is used for slightly better 
        efficiency. 
        
        Args: 
        feats (numpy.array): Feature representations of CIFAR 10 from SimCLR 
        n_clusters (int): Number of clusters 

        Returns: 
        k_means.labels_ (np.array): Labels of each point
        k_means.cluster_centre_ (np.array): Centroids 
        """
        if n_clusters <= 50: 
            km = KMeans(n_clusters=n_clusters)
            km.fit_predict(feats)
        else:
            km = MiniBatchKMeans(n_clusters=n_clusters)
            km.fit_predict(feats)

        return km.labels_, km.cluster_centers_
    
    def calculate_typicality(self, query_point, nbrs):
        """
        Calculate typicality of a point compared to it's neighbours. 

        The Nearest Neighbour algorithm is initialised outside of this method
        as typicality is called for every point in the cluster. Hence initialising
        once outside the method is more efficient. 
        
        Args:
        query_point List[int] : Feature representation of a single point 
        nbrs (NearestNeighbors): Initialised nearest neighbours (contains all points in a specified cluster)

        Returns:
        result (float): Typicality score of a point in their cluster 
        """
        query_point = np.array([query_point])

        # Return Euclidean distances 
        distances, indices = nbrs.kneighbors(query_point)
        
        # Remove the point as it will be the query point 
        distances = distances[0][1:]

        result = (1/ np.mean(distances))
        return result

    def retrieve_cluster_data(self, cluster_array):
        """
        Given the cluster labels of each point, count the number of points in each cluster
        and indices of each point in a cluster. 

        Args: 
        cluster_array (KMeans.labels_) : fitted k-means on feature representation 

        Returns: 
        cluster_indices dict[int -> List[int]] : Cluster ID mapped to a list of data point's 
        indices that belong in the cluster 

        cluster_sizes List[int] : Index of list corresponds to clusterID and the value 
        denotes the number of points in that cluster.
        """
        cluster_indices = {}
        cluster_sizes = [0] * len(np.unique(cluster_array)) 

        # Loop through cluster array 
        for count, value in enumerate(cluster_array):
            # Increment number of points in cluster by 1
            cluster_sizes[value] += 1 

            # Append point indices to corresponding cluster 
            if value not in cluster_indices:
                cluster_indices[value] = [count]
            else:
                cluster_indices[value].append(count)

        return cluster_sizes, cluster_indices
    
    def sample(self, l_set_indices):
        """
        Perform TypiClust Sampling 

        Args: 
        l_set_indices (List[int]): indices of points currently in the labelled set 

        Returns: 
        l_set (torch.utils.data.Subset) : labelled set 
        l_set_indices (List[int]) : indices of points in labelled set wrt original unpartitioned dataset
        """ 
        indices_file_name = f"typiclustplus_{self.args['dataset_name']}_budget_{len(l_set_indices)+ self.budget_size}.npy"
        file_path = os.path.join(self.args['data_dir'], indices_file_name)

        # Check if a local file of that budget size exists and has the same labelled samples
        # as the the labelled set retained in the class attributes
        if os.path.isfile(file_path) and \
            l_set_indices[0:len(self.l_set_indices)] == self.l_set_indices:
 
            # Load in labelled sample indices 
            l_set_indices = np.load(file_path).tolist()
            print(f"Loaded {indices_file_name} locally...")
            
            # Update 
            self.l_set_indices = l_set_indices
        else:
            self.n_clusters = min(len(l_set_indices) + self.budget_size, 
                                self.MAX_NUM_CLUSTER)

            # Initialise clustering of feature space 
            self.get_uncovered_clusters(self.feats, self.n_clusters)

            # Largest uncovered cluster ID 
            largest_uncovered_clust = self.cluster_data.iloc[0].cluster_id

            # Largest cluster size (regardless if it contains labelled data or not)
            largest_cluster_size = self.cluster_data['cluster_size'].max()
            
            # Cluster to sample typical point from
            # Start with largest uncovered clust 
            sample_cluster_id = largest_uncovered_clust

            # Max labelled samples in a cluster 
            largest_labelled_count = self.cluster_data['labelled_count'].max()

            # Remaining Clusters we can sample from 
            cluster_remaining = list(set(self.cluster_data['cluster_id'].tolist()) - set([largest_uncovered_clust]))

            print(f"First sampling {sample_cluster_id}")
            print(f"remaining clusters to be sampled {cluster_remaining}")

            # Query for most typical points for chosen cluster 
            queries = []
            # Sample n times  
            for i in range(0, self.budget_size):
                max_typicality = -1 
                max_index = -1
                max_e = -100
    
                nbrs = NearestNeighbors(metric="cosine",
                                        n_neighbors=(min(self.K_NN, len(self.cluster_indices[sample_cluster_id]))),
                                            algorithm='auto').fit(
                                                self.feats[self.cluster_indices[sample_cluster_id]])

                # Loop through each point in cluster
                for i in self.cluster_indices[sample_cluster_id]:
                    # Calculate typicality of a point 
                    result = (self.calculate_typicality(self.feats[i], nbrs))

                    # Access if it currently has largest typicality and not labelled 
                    if result > max_typicality and (i not in (l_set_indices + queries)):
                        max_typicality = result
                        max_index = i

                queries.append(max_index)
                print(f"sampled {max_index}")

                # Choose the next cluster to sample 
                for c in cluster_remaining:
                    
                    e = self.args['alpha']*(np.linalg.norm(self.cluster_centres[sample_cluster_id]-self.cluster_centres[c]))
                    # print(self.cluster_data.loc[self.cluster_data['cluster_id'] == c].cluster_size)
                    e += self.args['beta']*(self.cluster_data.loc[self.cluster_data['cluster_id'] == c].cluster_size / largest_cluster_size).item()
                    
                    if largest_labelled_count != 0:
                        e -= self.args['gamma']*(self.cluster_data.loc[self.cluster_data['cluster_id'] == c].labelled_count / largest_labelled_count).item()
                    
                    # Access if cluster gives a better ensemble value than others
                    if e > max_e:
                        max_e = e 
                        sample_cluster_id = c  
                    
                        print(f"cluster {c} has been selected! size: {max_e}")
                
                # Update the remaining cluster that can be queried 
                cluster_remaining = list(set(cluster_remaining) - set([sample_cluster_id]))
                print(f"Remaining clusters {cluster_remaining}")

            # Update labelled data 
            self.l_set_indices = l_set_indices + queries

            # Save indices locally 
            with open(file_path, 'wb') as f: 
                np.save(f, self.l_set_indices)

        l_set = torch.utils.data.Subset(self.train_dataset, self.l_set_indices)
    
        return l_set, self.l_set_indices