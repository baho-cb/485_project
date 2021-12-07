from sys import argv
import sys
import os
sys.path.insert(0,"/home/baho/Desktop/scripts")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.cm as cm
import argparse
import numpy as np
import gsd
import gsd.hoomd
from ClusterUtils import ClusterUtils,ExtractedMolecule

"""
gsd_cluster_to_molecule.py -> Takes a snapshot that has nonaggregated shapes
but the shapes are covered with NOM. Shapes + coveringNOM makes a new rigid
molecule and we spit a new text and gsd file for that molecule.

First check the simulation.gsd file, find a molecule that fits your purposes
and write down an id of a particle in that molecule and give to this script
as input. Here I will dbscan and extract the target molecule

One can think that it is easier to do this on the already dbscanned output gsd
but the fact that we omit the filler hard spheres from that complicates things.

"""

parser = argparse.ArgumentParser(description="Cluster analysis of a gsd file for a single frame")
non_opt = parser.add_argument_group("mandatory arguments")
non_opt.add_argument('-i', '--input', metavar="<dat>", type=str, dest="input_file",
required=True, help=".gsd file" )
non_opt.add_argument('--cutoff', metavar="<float>", type=float, dest="cutoff",
required=True, help=" cutoff for DBSCAN ", default = -1 )
non_opt.add_argument('--id', metavar="<int>", type=int, dest="seed_id",
required=True, help=" any id that belongs to the molecule you want to extract " )
non_opt.add_argument('--min_samples', metavar="<int>", type=int, dest="min_samples",
required=True, help=" min_samples for DBSCAN ..\
lower than this value wont count as a cluster but will be noise ")
non_opt.add_argument('-f', '--frame', metavar="<int>", type=int, dest="target_frame",
required=True, help=" - ",default = -1 )
non_opt.add_argument('--dummy', metavar="<int>", type=int, dest="dummy_type",
required=False, help="if there is a dummy particle type that you don't want to ..\
consider when calculating the clusters, specify it here - for example the particles ..\
that are filling the empty space inside the shapes should be dummy particles ",default = -1  )
non_opt.add_argument('--name', metavar="<dat>", type=str, dest="molecule_name",
required=True, help="the name for the spitted molecule ")


args = parser.parse_args()
input_file = args.input_file
cutoff = args.cutoff
min_samples = args.min_samples
target_frame = args.target_frame
seed_id = args.seed_id
mol_name = args.molecule_name

cluster_utils = ClusterUtils(input_file,target_frame)

print("--- DBSCAN with ---")
print("Cutoff : ", cutoff)
print("Min_samples : ", min_samples)
print("File : ", input_file)
print("Frame : ", target_frame )
print("Seed id : ", seed_id )

if not os.path.exists('./extracted_molecules/'):
    os.makedirs('./extracted_molecules/')

mol_name_gsd = "./extracted_molecules/" + mol_name + ".gsd"
mol_name_txt = "./extracted_molecules/" + mol_name + ".txt"
cluster_count = cluster_utils.dbscan(cutoff,min_samples)
pos,types = cluster_utils.find_cluster(seed_id)
m_ = ExtractedMolecule(pos,types)
m_.dump_gsd(mol_name_gsd)
m_.dump_txt(mol_name_txt)
