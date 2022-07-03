import numpy as np
from numpy import pi
from numpy import inf
import matplotlib.pyplot as plt
import dionysus as sus
import pandas as pd


def Delay_Embeding(x, d=2):
  N = x.shape[0]
  Y = np.zeros((N-d, d))
  for i in range(N - d):
    Y[i, :] = x[i:i+d]
    
  return Y

def DGMS(point, dim, rad):
  simp = sus.fill_rips(point, dim, rad)
  hp = sus.homology_persistence(simp)
  dgms = sus.init_diagrams(hp, simp)
  return dgms


def PH(dgms, Betti = 0):
  I=[]
  persist=[]
  rc=[]
  dgm0 = []
  dgm1 = []
  dgm2 = []
  for i, dgm in enumerate(dgms):
    if i == 0 : dgm0.append(dgm)
    if i == 1 : dgm1.append(dgm)
    if i == 2 : dgm2.append(dgm)
    for pt in dgm:
          I.append(i)
          persist.append(pt.death-pt.birth)
          rc.append(pt.birth)
  np.savetxt('./tda_results/dgm0.txt', dgm0)
  np.savetxt('./tda_results/dgm1.txt', dgm1)
  np.savetxt('./tda_results/dgm2.txt', dgm2)
  pa=[]
  ra=[]
  ia = []
  for i in range (0, len(persist)):
    if persist[i]!= inf and I[i] == Betti:
          pa.append(persist[i])
          ra.append(rc[i])
          ia.append(I[i])
  return pa, ra, ia  

def PDF(ra, pa):
    plt.figure(figsize=(12, 5))
    df=pd.DataFrame( {'f(r)':ra,'persistence':pa})
    ax=df.plot.hexbin(x='f(r)',y='persistence',gridsize=25)
    plt.show();

def Barcode(ra, pa, ia):
    plt.figure(figsize=(12, 5))
    for i in range(len(ia)):
        if(ia[i] == 0): clr='red'
        if(ia[i] == 1): clr='darkorange'
        if(ia[i] == 2): clr='blue'
        # print([pa[i],pa[i]+ra[i]])
        # print(ra[i])
        plt.yticks([])
        plt.title(r'$Barcode$ $\beta_1$', fontsize = 18, fontweight='bold')
        plt.xlabel(r'$\rho$', fontsize=14, fontweight='bold')
        plt.plot([pa[i],pa[i]+ra[i]],[ia[i]+i,ia[i]+i], color = clr)


mag1 = np.loadtxt("1.12.txt")
mag2 = np.loadtxt("2.24.txt")
mag3 = np.loadtxt("3.56.txt")

dim = 2
n = 10000
point1 = Delay_Embeding(mag1, dim)
point2 = Delay_Embeding(mag2, dim)
point3 = Delay_Embeding(mag3, dim)

dgms1 = DGMS(point1[:n], dim, 0.1)
dgms2 = DGMS(point2[:n], dim, 0.1)
dgms3 = DGMS(point3[:n], dim, 0.1)

print(dgms1)
print(dgms2)
print(dgms3)

beti = 1
pa1, ra1, ia1 = PH(dgms1, beti)
pa2, ra2, ia2 = PH(dgms2, beti)
pa3, ra3, ia3 = PH(dgms3, beti)

np.savetxt('./tda_results/pa1.txt', pa1)
np.savetxt('./tda_results/pa2.txt', pa2)
np.savetxt('./tda_results/pa3.txt', pa3)

np.savetxt('./tda_results/ra1.txt', ra1)
np.savetxt('./tda_results/ra2.txt', ra2)
np.savetxt('./tda_results/ra3.txt', ra3)

np.savetxt('./tda_results/ia1.txt', ia1)
np.savetxt('./tda_results/ia2.txt', ia2)
np.savetxt('./tda_results/ia3.txt', ia3)
