import numpy as np
import gsd.hoomd
from hoomd.data import make_snapshot, boxdim
from collections import defaultdict
import os.path
import networkx as nx
import itertools
import time
from sklearn.cluster import DBSCAN, OPTICS, MeanShift, estimate_bandwidth
from scipy.spatial import cKDTree as KDTree
import matplotlib.pyplot as plt
import freud
from scipy.spatial import distance
from textwrap import wrap
np.set_printoptions(suppress=True)

# ------------------------------------------------------------------------------
#                              Helper functions
#-------------------------------------------------------------------------------

class ClusterUtils():
    """

    """
    def __init__(self,input,frame):
        print("Reading in : " + input)
        self.positions = []
        self.types = []
        self.typeid = []
        self.velocities = []

        self.bond_types = []
        self.bond_typeid = []
        self.bond_group = []

        self.Lx = 0
        self.Ly = 0
        self.Lz = 0
        self.cluster_ids = [] ### unique cluster ids
        self.noise_id = 0
        self.cluster_N = []
        self.filename = input[:-4]
        self.frame = 0
        self.read_system(input,frame)

    def read_system(self,input,target_frame):
        """
        Read in a snapshot from a gsd file or snapshot.
        """
        try:
            with gsd.hoomd.open(name=input, mode='rb') as f:
                if (target_frame==-1):
                    frame = f.read_frame(len(f)-1)
                    self.frame = len(f)-1
                    print("Reading last frame ")
                else:
                    self.frame = target_frame
                    frame = f.read_frame(target_frame)
                    print("Reading frame ", target_frame)
                self.positions = (frame.particles.position).copy()
                self.velocities = (frame.particles.velocity).copy()

                self.types = (frame.particles.types).copy()
                self.typeid = (frame.particles.typeid).copy()

                self.bond_types = (frame.bonds.types).copy()
                self.bond_group = (frame.bonds.group).copy()
                self.bond_typeid = (frame.bonds.typeid).copy()

                self.Lx,self.Ly,self.Lz = frame.configuration.box[0:3]
                self.box = frame.configuration.box

        except:
            self.positions = (input.particles.position).copy()
            self.velocities = (input.particles.velocity).copy()
            self.types = (input.particles.types).copy()
            self.typeid = (input.particles.typeid).copy()

            self.bond_types = (input.bonds.types).copy()
            self.bond_group = (input.bonds.group).copy()
            self.bond_typeid = (input.bonds.typeid).copy()

            self.Lx = input.box.Lx
            self.Ly = input.box.Lx
            self.Lz = input.box.Lx
            self.box = input.box

    def remove_dummy(self,dummy_type):
        if(dummy_type == -1):
            print("No dummy particles specified")
        else:
            print("removing particles of type %d" %dummy_type)
            self.positions = self.positions[self.typeid!=dummy_type]
            self.velocities = self.velocities[self.typeid!=dummy_type]
            self.typeid = self.typeid[self.typeid!=dummy_type]


    def dbscan(self,cutoff,min_samples):
        self.cutoff = cutoff
        self.min_samples = min_samples
        ### writes cluster info to velocitites
        self.positions = self.positions + self.box[0]*0.5
        tree = KDTree(data=self.positions, leafsize=12, boxsize=self.box[0])
        pairs = tree.sparse_distance_matrix(tree,cutoff+1.0)
        dbscan = DBSCAN(eps=cutoff, min_samples=min_samples, metric="precomputed", n_jobs=-1)
        labels0 = dbscan.fit_predict(pairs)
        n_clusters0 = len(set(labels0)) - (1 if -1 in labels0 else 0)
        n0,cluster_count = np.unique(labels0,return_counts=True)
        #### cluster count is the size magnitude of each cluster starting wiht the
        #### noise cluster
        print("N Detected clusters : ", len(cluster_count)-1)
        print("# of particles in detected clusters : ")
        print(np.flip(np.sort(cluster_count[1:])))
        cluster_info = np.ones_like(self.positions)
        cluster_info = np.multiply(cluster_info,999)
        #labels0[labels0==0]=7
        cluster_info[:,0] = labels0
        cluster_info[:,1] = cluster_count[labels0+1]
        ### +15 below is to create a color contrast btw nonclusters(when they are -1)
        ### some clusters cnt be separated visually with ease
        n0[n0==-1] = len(cluster_count) + 5
        cluster_info[cluster_info==-1] = len(cluster_count) + 5
        self.noise_id = len(cluster_count) + 5
        self.cluster_ids = n0
        self.velocities = cluster_info
        self.cluster_N = cluster_count
        return (np.flip(np.sort(cluster_count[1:])))

    def find_cluster(self,seed_id):
        target_cluster = self.velocities[seed_id,0]
        N_target_cluster = self.velocities[seed_id,1]
        print("%d particles are in the target cluster" %N_target_cluster)
        target_ids = np.where(self.velocities[:,0]==target_cluster)
        target_types = self.typeid[target_ids]
        target_pos = self.positions[target_ids]
        ## Move molecule to the origin, I don't care about orientation
        ## since I will only do pmf vs distance
        relevant_ids = np.where(target_types==0)
        relevant_ids = relevant_ids[0]
        relevant_pos = target_pos[relevant_ids]
        cm = np.average(relevant_pos,axis=0)
        target_pos = np.subtract(target_pos,cm)
        return target_pos,target_types


    def size_analysis(self):
        ### calculating end-to-end size (ie longest distance) of a cluster
        ### if cluster is very larger than half the box size the size measurement
        ### will be wrong

        ### when calculating fractal dimensions the void inside the shapes should
        ### be accounted for but how ?

        ### Assumes sigma = 1 for every bead not - always true
        ### Assumes the largest cluster is smaller than half the box size

        ### Not clear how to calculate fractal dimension
        self.cluster_size = np.zeros_like(self.cluster_ids,dtype=float)
        self.cluster_f_dim = np.zeros_like(self.cluster_ids,dtype=float)
        for i,c_id in enumerate(self.cluster_ids):
            if (c_id!=self.noise_id):
                pos = self.positions[self.velocities[:,0]==c_id]
                tree = KDTree(data=pos, leafsize=12, boxsize=self.box[0])
                pairs = tree.sparse_distance_matrix(tree,np.sqrt(3)*self.box[0]*0.5)
                pairs = pairs.toarray()
                pairs = pairs.flatten()
                max_r = np.amax(pairs)*0.5
                N = self.cluster_N[i]
                d_fractal = np.log(N*((0.5)**3))/np.log(max_r)
                self.cluster_size[i] = max_r
                self.cluster_f_dim[i] = d_fractal

    def size_analysis_multibox(self):
        ### explicitly put image boxes for the cluster
        ### do a dbscan for that configuration
        ### the largest cluster should be the original one but without
        ### crossing any boundaries so you can calculate the distance directly
        ### if the largest cluster is bigger than the original one than there is
        ### percolation which we shouldn't have in any scenario anyway
        self.cluster_size = np.zeros_like(self.cluster_ids, dtype=float)
        self.cluster_f_dim = np.zeros_like(self.cluster_ids, dtype=float)
        for i,c_id in enumerate(self.cluster_ids):
            if (c_id!=self.noise_id):
                pos_0 = self.positions[self.velocities[:,0]==c_id]
                pos = pos_0
                for ii in range(-1,2):
                    for jj in range(-1,2):
                        for kk in range (-1,2):
                            if(ii==0 and jj==0 and kk==0):
                                pass
                            else:
                                image = np.array([ii,jj,kk])
                                pos = np.vstack((pos,np.add(pos_0,np.multiply((np.ones_like(pos_0)*self.box[0]),image[np.newaxis,:]))))
                tree = KDTree(data=pos, leafsize=12)
                pairs = tree.sparse_distance_matrix(tree,self.cutoff+1.0)
                dbscan = DBSCAN(eps=self.cutoff, min_samples=self.min_samples, metric="precomputed", n_jobs=-1)
                labels0 = dbscan.fit_predict(pairs)
                n0,cluster_count = np.unique(labels0,return_counts=True)
                max_count = np.amax(cluster_count)
                if(max_count>self.cluster_N[i]):
                    print("Check snapshot for percolation!!!!")
                    print("Multibox cluster_N : %d " %max_count)
                    print("Original cluster_N : %d " %self.cluster_N[i])
                    exit()
                elif(max_count==self.cluster_N[i]):
                    safe_pos = pos[labels0==n0[np.argmax(cluster_count)]]
                    distances = distance.pdist(safe_pos)
                    max_r = np.amax(distances)*0.5
                    N = self.cluster_N[i]
                    d_fractal = np.log(N*((0.5)**3))/np.log(max_r)
                    self.cluster_size[i] = max_r
                    self.cluster_f_dim[i] = d_fractal
                else:
                    print("error 163")
                    exit()
        self.cluster_size = self.cluster_size[self.cluster_size!=0]
        self.cluster_f_dim = self.cluster_f_dim[self.cluster_f_dim!=0]

    def plot_N_histogram(self,save,show):
        cluster_N_wo_noise = self.cluster_N[1:]
        hist, edges = np.histogram(cluster_N_wo_noise) #range=(0,220))
        center = (edges[1:] + edges[:-1])/2
        plt.figure(1)
        plt.title("Cluster N histogram of %s at frame %d - total # %d" %(self.filename,self.frame,len(self.typeid)))
        bar_width = (np.amax(self.cluster_N)/len(hist))*0.2
        plt.bar(center,hist, width=(edges[1] - edges[0])*0.8)
        plt.xlabel("N_paricle")
        plt.ylabel("cluster count")
        if(save==1):
            plt.savefig("./cluster_gsd/cluster_N_hist_%s_frame_%d_.pdf" %(self.filename,self.frame))
        if(show==1):
            plt.show()

    def plot_fraction_histogram(self,save,show):
        cluster_N_sorted = np.flip(np.sort(self.cluster_N[1:]))
        hist, edges = np.histogram(cluster_N_sorted) #range=(0,220))
        bin_sums = np.zeros(len(hist))
        for i in range(0,len(hist)):
            current_clusters = cluster_N_sorted[cluster_N_sorted>edges[i]]
            current_clusters = current_clusters[current_clusters<=edges[i+1]]
            bin_sums[i] = np.sum(current_clusters)
        bin_fractions = np.divide(bin_sums,len(self.positions))
        center = (edges[1:] + edges[:-1])/2
        plt.figure(2)
        plt.title("\n".join(wrap("fraction of the particles in the given cluster size range %s at frame %d - noise %.3f " %(self.filename,self.frame,1.0-np.sum(bin_fractions)), 60)))
        bar_width = (np.amax(self.cluster_N)/len(hist))*0.2
        plt.bar(center,bin_fractions, width=(edges[1] - edges[0])*0.7)
        # plt.plot(np.linspace(0,len(hist)+1,num=10000),np.zeros(10000) )#,marker='o',c='r',s=20.0)
        plt.xlabel("N_particle")
        plt.ylabel("fractions")
        if(save==1):
            plt.savefig("./cluster_gsd/fractions_%s_frame_%d_.pdf" %(self.filename,self.frame))
        if(show==1):
            plt.show()

    def plot_size_histogram(self,save,show):
        hist, edges = np.histogram(self.cluster_size) #range=(0,220))
        center = (edges[1:] + edges[:-1])/2
        plt.figure(3)
        plt.title("Cluster size histogram of %s at frame %d" %(self.filename,self.frame))
        plt.bar(center,hist,width=(edges[1] - edges[0])*0.7)#,marker='o',c='r',s=20.0)
        plt.xlabel("max_r/\u03C3")
        plt.ylabel("cluster count")
        if(save==1):
            plt.savefig("./cluster_gsd/cluster_size_hist_%s_frame_%d_.pdf" %(self.filename,self.frame))
        if(show==1):
            plt.show()

    # title = ax.set_title("\n".join(wrap("fraction of the particles in the given cluster size range %s at frame %d - noise %.3f " %(self.filename,self.frame,1.0-np.sum(bin_fractions)), 60)))
    def plot_fractal_dim_histogram(self,save,show):
        hist, edges = np.histogram(self.cluster_f_dim,bins=10)
        center = (edges[1:] + edges[:-1])/2
        plt.figure(4)
        plt.title("Fractal dimension histogram")
        bar_width = ((np.amax(self.cluster_f_dim) - np.amin(self.cluster_f_dim)) / len(hist))*0.2
        plt.bar(center,hist,width=(edges[1] - edges[0])*0.7)#,marker='o',c='r',s=20.0)
        plt.xlabel("df")
        plt.ylabel("cluster count")
        if(save==1):
            plt.savefig("./cluster_gsd/fractal_dim_histogram_%s_frame_%d_.pdf" %(self.filename,self.frame))
        if(show==1):
            plt.show()

    def plot_2d_scatter_size_vs_fractaldim(self,save,show):
        plt.figure(5)
        plt.title("fractal dim vs cluster size scatter of %s at frame %d" %(self.filename,self.frame))
        plt.scatter(self.cluster_size,self.cluster_f_dim)#,marker='o',c='r',s=20.0)
        plt.xlabel("size")
        plt.ylabel("fractal dim")
        plt.ylim(0.0,3.0)
        if(save==1):
            plt.savefig("./cluster_gsd/fractal_dim_vs_cluster_size_%s_frame_%d_.pdf" %(self.filename,self.frame))
        if(show==1):
            plt.show()


    def dump_snap_w_cluster_info_at_vel(self, outfile):
        snap = gsd.hoomd.Snapshot()
        snap.configuration.step = 0
        snap.configuration.box = [self.Lx, self.Ly, self.Lz, 0, 0, 0]

        # particles
        snap.particles.N = len(self.positions)
        snap.particles.position = self.positions[:]
        snap.particles.types  = self.types[:]
        snap.particles.typeid = self.typeid[:]

        snap.particles.velocity = self.velocities[:]

        # save the configuration
        with gsd.hoomd.open(name=outfile, mode='wb') as f:
            f.append(snap)

### ---------------- NO NEED BELOW ----------------------------- ####

    def color_by_position(self): ### exampe function

        velocities = np.zeros_like(self.positions)
        velocities[:,:] = np.sign(self.positions)
        self.velocities = velocities

    def cr_pair_correlation(self,bins=50,r_max=10.0):
        rdf = freud.density.RDF(bins=bins, r_max=r_max)
        rdf.compute(system=(self.box, self.positions[self.typeid==2]), reset=False)
        plt.figure(1)
        plt.plot(rdf.bin_centers,rdf.rdf)
        plt.show()

    def cr_avg_dist_density(self):
        positions = self.positions[self.typeid==2]
        origin = np.array([[0.0,0.0,0.0]])
        dists_to_origin = distance.cdist(origin,positions)
        max_dist = np.max(dists_to_origin[0])
        radius = 0.75 * max_dist ### better measure
        positions = positions[dists_to_origin[0]<radius]
        avg_dist = np.power(len(positions)/((4.0/3.0)*(3.14)*(radius**3)),-1.0/3.0  )
        return(avg_dist)


    def histogram_cubic(self,bin_size):
        positions = self.positions[self.typeid!=3]
        bin_edges = edges_for_cubic_histogram(positions,bin_size)
        H, edges = np.histogramdd(positions, bins = bin_edges)
        edges=np.array(edges)
        bin_volume = (edges[0][1]-edges[0][0])*(edges[1][1]-edges[1][0])*(edges[2][1]-edges[2][0])
        print("Bin edges : %.2f %.2f %.2f" %((edges[0][1]-edges[0][0]),(edges[1][1]-edges[1][0]),(edges[2][1]-edges[2][0])))
        print("Bin volume : ",  bin_volume)
        return H,edges

    def color_by_crosslinker_cluster(self,cutoff,min_samples):
        ## two points that are closer thn cutoff distance are considered to be
        ## in the same cluster
        ### if I understand correctly : to say that a group of points is a cluster
        ### there neds to be at least min_samples points (if min sample is 1, two
        ### points very close to each other but far from everything else is a cluster
        ### but if min_samp. is 3 they are not cluster but interpreted as noise(-1
        ### for the cluster label)    )

        ### get the crosslinker positions
        tree = KDTree(data=self.positions[self.typeid==2], leafsize=12)
        pairs = tree.sparse_distance_matrix(tree,cutoff)
        # dbscan = DBSCAN(eps=cutoff, min_samples=1, metric="precomputed", n_jobs=-1)
        dbscan = DBSCAN(eps=cutoff, min_samples=min_samples, metric="precomputed", n_jobs=-1)
        labels0 = dbscan.fit_predict(pairs)
        n_clusters0 = len(set(labels0)) - (1 if -1 in labels0 else 0)
        n0,cluster_count = np.unique(labels0,return_counts=True)
        print("Detected clusters : ", len(cluster_count)-1)
        print(np.flip(np.sort(cluster_count[1:])))
        # print(cluster_count)
        cluster_info = np.ones_like(self.positions[self.typeid==2])
        cluster_info = np.multiply(cluster_info,999)
        #labels0[labels0==0]=7

        cluster_info[:,0] = labels0

        cluster_info[:,1] = cluster_count[labels0+1]
        ### +15 below is to create a color contrast btw nonclusters(when they are -1)
        ### some clusters cnt be separated visually with ease



        cluster_info[cluster_info==-1]=len(cluster_count + 5)
        self.velocities[self.typeid==2] = cluster_info
        return (np.flip(np.sort(cluster_count[1:])))

    def color_by_monomer_cluster_MEANSHIFT(self,cutoff):
        print("Calculating tree ...")
        #tree = KDTree(data=self.positions[self.typeid!=3], leafsize=12)
        #pairs = tree.sparse_distance_matrix(tree,cutoff)
        data=self.positions[self.typeid!=3]
        # data=data[data[:,0]>0]
        # data=data[data[:,1]>0]
        # data=data[data[:,2]>0]
        meanshift = MeanShift(bandwidth=4.0, cluster_all=False, bin_seeding=True, n_jobs=-1)
        print("Calculating clusters ...")
        labels0 = meanshift.fit_predict(data)
        print(labels0)
        n_clusters0 = len(set(labels0)) - (1 if -1 in labels0 else 0)
        n0,cluster_count = np.unique(labels0,return_counts=True)
        print("Detected clusters : ", len(cluster_count)-1)
        print(np.flip(np.sort(cluster_count[1:])))
        cluster_info = np.ones_like(data)
        cluster_info = np.multiply(cluster_info,999)
        cluster_info[:,0] = labels0
        cluster_info[:,1] = cluster_count[labels0]
        self.velocities[self.typeid!=3] = cluster_info
        return (np.flip(np.sort(cluster_count[1:])))

    def color_by_crosslinker_cluster_OPTICS(self,min_samples):
        ### OPTICS is similar to DBSCAN I just try it to s if it is better or not

        tree = KDTree(data=self.positions[self.typeid==2], leafsize=12)
        pairs = tree.sparse_distance_matrix(tree,100)
        pairs = pairs.toarray()
        optics = OPTICS(min_samples=min_samples, metric="precomputed", n_jobs=-1)
        labels0 = optics.fit_predict(pairs)
        n_clusters0 = len(set(labels0)) - (1 if -1 in labels0 else 0)
        n0,cluster_count = np.unique(labels0,return_counts=True)
        print("Detected clusters (OPTICS): ", len(cluster_count)-1)
        print(np.flip(np.sort(cluster_count[1:])))
        cluster_info = np.ones_like(self.positions[self.typeid==2])
        cluster_info = np.multiply(cluster_info,999)
        cluster_info[:,0] = labels0
        cluster_info[:,1] = cluster_count[labels0]
        self.velocities[self.typeid==2] = cluster_info
        return (np.flip(np.sort(cluster_count[1:])))



    def get_snap(self,context):
        with context:


            snap = make_snapshot(N=len(self.positions),
                                particle_types=self.types,
                                bond_types=self.bond_types,
                                angle_types=self.angle_types,
                                dihedral_types=self.dihedral_types,
                                pair_types=self.pair_types,
                                box=boxdim(Lx=self.Lx,Ly=self.Ly,Lz=self.Lz))

            # set angle typeids and groups
            snap.angles.resize(len(self.angle_group))
            for k in range(len(self.angle_group)):
                snap.angles.typeid[k] = self.angle_typeid[k]
                snap.angles.group[k] = self.angle_group[k]

            # set angle typeids and groups
            snap.dihedrals.resize(len(self.dihedral_group))
            for k in range(len(self.dihedral_group)):
                snap.dihedrals.typeid[k] = self.dihedral_typeid[k]
                snap.dihedrals.group[k] = self.dihedral_group[k]

            # set specialpairs(4th neighbors)
            snap.pairs.resize(len(self.pair_group))
            for k in range(len(self.pair_group)):
                snap.pairs.typeid[k] = self.pair_typeid[k]
                snap.pairs.group[k] = self.pair_group[k]

            for k in range(len(self.positions)):
                snap.particles.position[k] = self.positions[k]
                snap.particles.typeid[k] = self.typeid[k]
            # set bond typeids and groups
            snap.bonds.resize(len(self.bond_group))
            for k in range(len(self.bond_group)):
                snap.bonds.typeid[k] = self.bond_typeid[k]
                snap.bonds.group[k] = self.bond_group[k]

        return snap



    def append_snap(self, outfile):
        ### if the gsd file exist append as the next frame, useful for example
        ### if you want to cluster with different cutoff values and pick best later
        snap = gsd.hoomd.Snapshot()
        snap.configuration.step = 0
        snap.configuration.box = [self.Lx, self.Ly, self.Lz, 0, 0, 0]\

        # particles
        snap.particles.N = len(self.positions)
        snap.particles.position = self.positions[:]
        snap.particles.types  = self.types[:]
        snap.particles.typeid = self.typeid[:]
        # bonds
        snap.bonds.N = len(self.bond_group)
        snap.bonds.types  = self.bond_types[:]
        snap.bonds.typeid = self.bond_typeid[:]
        snap.bonds.group  = self.bond_group[:]
        # angles
        snap.angles.N = len(self.angle_group)
        snap.angles.types  = self.angle_types[:]
        snap.angles.typeid = self.angle_typeid[:]
        snap.angles.group  = self.angle_group[:]

        # dihedrals
        snap.dihedrals.N = len(self.dihedral_group)
        snap.dihedrals.types  = self.dihedral_types[:]
        snap.dihedrals.typeid = self.dihedral_typeid[:]
        snap.dihedrals.group  = self.dihedral_group[:]

        # pairs
        snap.pairs.N = len(self.pair_group)
        snap.pairs.types  = self.pair_types[:]
        snap.pairs.typeid = self.pair_typeid[:]
        snap.pairs.group  = self.pair_group[:]

        snap.particles.velocity = self.velocities[:]
        # save the configuration
        with gsd.hoomd.open(name=outfile, mode='rb+') as f:
            f.append(snap)
