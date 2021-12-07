import numpy as np
import matplotlib.pyplot as plt
import argparse

parser = argparse.ArgumentParser(description="plots merged pmf data ")
non_opt = parser.add_argument_group("mandatory arguments")
non_opt.add_argument('-i', '--input', metavar="<dat>", type=str, dest="input_file", required=True, help="merged pmf data file" )
args = parser.parse_args()
input_file = args.input_file
with open(input_file) as f:
    firstline = f.readline().rstrip()

data = np.loadtxt(input_file)
N_sample = data[3,-1]

plt.figure(1)
plt.plot(data[:,1],data[:,2])
plt.plot(np.linspace(0,20,10000),np.linspace(0,20,10000)*0.0,"--r")
plt.xlabel("r",fontsize=16)
plt.ylabel("PMF / kbT",fontsize=16)
plt.title(" PMF vs CM-to-CM distance ",fontsize=16)
# plt.title("%s %s at N_sample = %d "%(input_file,firstline[-7:],N_sample))
# plt.ylim(np.min(data[:,2])-3.0,15.0)
# plt.xlim(1.7,6.7)
plt.tick_params(axis='both', which='major', labelsize=17)

plt.show()
exit()
