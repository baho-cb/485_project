import numpy as np
import matplotlib.pyplot as plt
import argparse
import sys, os
from matplotlib import cm
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
from matplotlib.backends.backend_pdf import PdfPages

"""
-> take the output of molecule_rotator_z.py
# dist angle energy
take the output of rotation around only one axis
so this script actually plots a radial 2d potential field not pmf
there is no averaging going on

https://matplotlib.org/stable/gallery/pie_and_polar_charts/polar_scatter.html
"""

parser = argparse.ArgumentParser(description="merge mpi-pmf outputs from mc_sim ")
non_opt = parser.add_argument_group("mandatory arguments")
non_opt.add_argument('-i', '--input', metavar="<dat>", type=str, dest="input_file", required=True, help="first of series of .pmf.dat files" )
non_opt.add_argument('--axis', metavar="<dat>", type=int, dest="axis", required=False, help="x-1,y-2,z-3" )
non_opt.add_argument('--extra', metavar="<dat>", type=str, dest="extra_info", required=False, help="plot with a different name", default="_" )
args = parser.parse_args()
input_file = args.input_file
axis = args.axis
extra_info = args.extra_info

data = np.loadtxt(input_file)
name = input_file[:-4] + '_angleZ_fancy'
pp = PdfPages(name + '.pdf')
theta = data[:,1]

top = cm.get_cmap('Greens_r', 128)
bottom = cm.get_cmap('Reds', 128)
newcolors = np.vstack((top(np.linspace(0, 1, 128)),
                       bottom(np.linspace(0, 1, 128))))

newcmp = ListedColormap(newcolors, name='GreenRed')
colormap = newcmp
newcmp.set_over(color='black')
vmin,vmax = -15.0,15.0
r = data[:,0]
colors = data[:,2]
fig = plt.figure(1)
ax = fig.add_subplot(projection='polar')
ax.set_ylim(0,6.5)
imdata = np.zeros((len(data),2))
imdata[:,0] = data[:,2]
imdata[:,1]= data[:,2]
im = ax.imshow(imdata,cmap=colormap,vmin=vmin,vmax=vmax)
c = ax.scatter(theta, r, c=colors, s=0.2*r*r, cmap=colormap, vmin=vmin, vmax=vmax)#, alpha=0.75 )
plt.title(input_file)
plt.colorbar(im)

"""
add some linear plots that approach through different angles
"""
N_approach = 5
approach = np.linspace(0,np.pi,N_approach)
r_app = np.sort(np.unique(r))
r_app_trace = np.linspace(r_app[0],r_app[-1],num=10000)

for angle in approach:
    ax.plot(np.ones_like(r_app_trace)*angle,r_app_trace,c="c")
plt.savefig(pp,format='pdf')


n_increment = np.int((len(np.unique(theta))*0.5)/(N_approach - 1))
unique_angles = np.sort(np.unique(theta))
for i_a,angle in enumerate(approach):
    u_a = unique_angles[i_a*n_increment]
    u_data = data[data[:,1]==u_a]
    plt.figure(i_a+2)
    plt.plot(u_data[:,0],u_data[:,2],"k")
    plt.plot(u_data[:,0],u_data[:,0]*0.0,"r")
    plt.ylim(-15.0,15.0)
    plt.title("%s_%.2f_degrees"%(name,u_a*(180.0/np.pi)))
    # plt.show()
    plt.savefig(pp,format='pdf')
pp.close()
exit()
