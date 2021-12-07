from sys import argv
import sys,os
sys.path.insert(0,"/home/baho/Desktop/scripts")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.cm as cm
import argparse
import numpy as np
import gsd
import gsd.hoomd
from sklearn.cluster import DBSCAN
from scipy.spatial import cKDTree as KDTree
import freud
from scipy.spatial.transform import Rotation as Rot
from scipy.spatial import distance
np.set_printoptions(linewidth=200,threshold=20000,precision=6,floatmode='fixed')

def SS(r,sigma,eps,w1,bar,w2):
    en = np.zeros_like(r)
    en[r<(sigma+w1)] = -eps
    # en[r<sigma] = 99999999.9
    en[r<sigma] = 750.0
    en[r>(sigma+w1)] = bar
    en[r>(sigma+w1+w2)] = 0.0
    return np.sum(en)

def WCA(r,eps,sigma):
    x = 4.0*eps*((sigma/r)**12 - (sigma/r)**6 + 0.25)
    cutoff = sigma * np.power(2.0,1.0/6.0)
    x[r>cutoff] = 0.0
    return np.sum(x)

def dump_snap(pos, types, outfile):
    snap = gsd.hoomd.Snapshot()
    snap.configuration.step = 0
    snap.configuration.box = [20.0, 20.0, 20.0, 0, 0, 0]

    # particles
    snap.particles.N = len(pos)
    snap.particles.position = pos
    snap.particles.types  = ['A','B','C']
    snap.particles.typeid = types

    if os.path.exists("./rotator_output/"+outfile):

        with gsd.hoomd.open(name="./rotator_output/"+outfile, mode='rb+') as f:
            f.append(snap)
    else:
        with gsd.hoomd.open(name="./rotator_output/"+outfile, mode='wb') as f:
            f.append(snap)

def rotate_quat_by_quat1(q2,q1):
    """
    First(q2) one quat to be rotated
    Second(q1) rotator quat
    evaluate  q1*q2 (quaternion multiplication = rotation)
    last term is always scalar
    onenote 6.0
    """
    b,c,d,a = q1
    f,g,h,e = q2
    q3 = np.array([
    a*f+b*e+c*h-d*g,
    a*g-b*h+c*e+d*f,
    a*h+b*g-c*f+d*e,
    a*e-b*f-c*g-d*h
    ])
    return q3

def rotate_quat_by_quat2(q2,q1):
    """
    First(q2) one quat to be rotated
    Second(q1) rotator quat
    same as the other but using dot and vector products instead
    one note 6.0
    """
    s = q1[3]
    t = q2[3]
    v = q1[:3]
    w = q2[:3]
    q3 = np.zeros(4)
    q3[3] = s*t - np.dot(v,w)
    q3[:3] = s*w + t*v + np.cross(v,w)
    return q3

def quat_to_matrix(quat):
    """
    Convert a quaternion (assuming last term is scalar)
    to a rotation matrix that rotates columns when multiplies
    from left.
    https://en.wikipedia.org/wiki/Rotation_formalisms_in_three_dimensions#Euler_rotations
    """
    i = quat[0]
    j = quat[1]
    k = quat[2]
    r = quat[3]
    matrix = np.zeros((3,3))
    matrix[0,0] = -1.0 + 2.0*i*i + 2.0*r*r
    matrix[0,1] = 2.0*(i*j-k*r)
    matrix[0,2] = 2.0*(i*k+j*r)
    matrix[1,0] = 2.0*(i*j+k*r)
    matrix[1,1] = -1.0 + 2.0*j*j + 2.0*r*r
    matrix[1,2] = 2.0*(j*k-i*r)
    matrix[2,0] = 2.0*(i*k-j*r)
    matrix[2,1] = 2.0*(j*k+i*r)
    matrix[2,2] = -1.0 + 2.0*k*k + 2.0*r*r
    return matrix

def matrix_to_quat(m):
    """
    take a rotation matrix and return corresponding quat
    onenote 6.0
    """
    q = np.zeros(4)
    q[3] = 0.5*np.sqrt(1.0+m[0,0]+m[1,1]+m[2,2])
    q[0] = (1.0/(4.0*q[3]))*(m[2,1]-m[1,2])
    q[1] = (1.0/(4.0*q[3]))*(m[0,2]-m[2,0])
    q[2] = (1.0/(4.0*q[3]))*(m[1,0]-m[0,1])
    return q



def random_quat():
    """
    Not sure the first or last term is the scalar
    probably last
    """
    rands = np.random.uniform(size=3)
    quat = np.array([np.sqrt(1.0-rands[0])*np.sin(2*np.pi*rands[1]),
            np.sqrt(1.0-rands[0])*np.cos(2*np.pi*rands[1]),
            np.sqrt(rands[0])*np.sin(2*np.pi*rands[2]),
            np.sqrt(rands[0])*np.cos(2*np.pi*rands[2])])
    return quat

def rotate_rows(matrix,pos):
    """
    takes a position matrix (N,3)-> rows are positions
    and a rotation matrix to rotate the positions
    - simple r_mat *matmul* pos_mat rotates the columns only
    which is not what we want. It is possible to take
    transpose of row-wise position vector do the matmul
    and than retranspose it again to row wise to get the
    same result.
    """
    pos = np.matmul(matrix,np.transpose(pos))
    return np.transpose(pos)

def from_quat_to_euler(q):
    """
    transform given quaternion to euler angles (fi,teta,psi)
    https://en.wikipedia.org/wiki/Conversion_between_quaternions_and_Euler_angles
    becareful in the link first term is the scalar
    """
    q0 = q[3]
    q1 = q[0]
    q2 = q[1]
    q3 = q[2]
    fi = np.arctan2(2.0*(q0*q1+q2*q3),1.0-2.0*(q1*q1 + q2*q2))
    teta = np.arcsin(2.0*(q0*q2 - q3*q1))
    psi = np.arctan2(2.0*(q0*q3 + q1*q2),1.0 - 2.0*(q2*q2+q3*q3))
    return np.array([fi,teta,psi])


"""
Adjust WCA parameters, distance range
"""

parser = argparse.ArgumentParser(description="do stuff with a molecule gsd ")
non_opt = parser.add_argument_group("mandatory arguments")
non_opt.add_argument('-i', '--input', metavar="<dat>", type=str, dest="input_file", required=True, help=".gsd file of a single molecule" )
non_opt.add_argument('--sigma', metavar="<float>", type=float, dest="sigma", required=False, default=1.0)
non_opt.add_argument('--eps', metavar="<float>", type=float, dest="eps", required=False, default=1.0)
non_opt.add_argument('--w1', metavar="<float>", type=float, dest="w1", required=False, default=1.0)
non_opt.add_argument('--bar', metavar="<float>", type=float, dest="bar", required=False, default=1.0)
non_opt.add_argument('--w2', metavar="<float>", type=float, dest="w2", required=False, default=1.0)
non_opt.add_argument('--run_name', metavar="<dat>", type=str, dest="run_name", required=True)
non_opt.add_argument('--z_offset', metavar="<float>", type=float, dest="z_val", required=False, default=0.0)

args = parser.parse_args()
input_file = args.input_file
sigma = args.sigma
eps = args.eps
w1 = args.w1
bar = args.bar
w2 = args.w2
run_name = args.run_name
z_val = args.z_val

"""
Very similar to molecule_rotator that calculates angular pmf at a given distance
Here I only rotate around z axis (I may even hold shape constant and play with
the particle) Instead of pmf now I will only get a 2d potential field

Hold the shape in place rotate only the ob the circle around CM of shape
while incrementing the angle and record dist angle energy than plot as 2d
polar plot
# dist angle energy
"""
#potential params
eps_WCA = 1.0
sigma_WCA = 1.35
# N_dists = 100 ## default ones
# N_angles = 500
N_dists = 700
N_angles = 2000

in_file=gsd.fl.GSDFile(name=input_file, mode='rb',application="hoomd",schema="hoomd", schema_version=[1,0])
N_frames=in_file.nframes
trajectory  = gsd.hoomd.HOOMDTrajectory(in_file)
frame = trajectory[N_frames-1]
pos = frame.particles.position[:]
types = frame.particles.typeid[:]
types_dump = np.hstack((types,np.array([2])))
dists = np.linspace(1.9,7.5,num=N_dists)
angles = np.linspace(0.0,np.pi*2,num=N_angles)

cm_actual = np.average(pos,axis=0)
pos = np.subtract(pos,cm_actual)
z_offset = np.array([0.0,0.0,z_val])
pos = np.add(pos,z_offset)
data = []
for d in dists:
    for a in angles:
        p_pos = np.array([[d*np.cos(a),d*np.sin(a),0.0]])
        dists = distance.cdist(p_pos,pos)
        dists = dists[0]
        d_ss = dists[types==0]
        d_wca = dists[types==1]
        en_ss = SS(d_ss,sigma,eps,w1,bar,w2)
        en_wca = WCA(d_wca,eps_WCA,sigma_WCA)
        en = en_wca + en_ss
        data.append(np.array([d,a,en]))
        # pos_dump = np.vstack((pos,p_pos))
        # dump_snap(pos_dump,types_dump,"301_rotZ_norepulsion.gsd")

data = np.array(data)
if not os.path.exists('./angleZ_out/'):
    os.makedirs('./angleZ_out/')

np.savetxt("./angleZ_out/%s.txt"%run_name,data,fmt="%1.3f %1.3f %1.3f" )
