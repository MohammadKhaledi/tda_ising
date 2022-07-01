import numpy as np
from numpy import exp
from numpy import pi
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
from tqdm import tqdm


def calculate_energy(config):
    energy = 0
    N = len(config)
    for i in range(0, len(config)):
        for j in range(0, len(config)):
            energy += (-1)*config[i][j]*(config[(i+1)%N][j] +
                                         config[(i-1)%N][j] +
                                         config[i][(j+1)%N] +
                                         config[i][(j-1)%N])
    return energy/4

def monte_carlo(config, temperature, en, mag):
    current_energy = en
    N = len(config)
    for i in range(N):
      for j in range(N):
        spin_x = np.random.randint(0, N)
        spin_y = np.random.randint(0, N)
        del_E = 2 * config[spin_x][spin_y] *  ( config[(spin_x-1)%N][spin_y] + config[(spin_x+1)%N][spin_y] +
                                               config[spin_x][(spin_y-1)%N] + config[spin_x][(spin_y+1)%N] )
        if( (del_E <= 0) or (exp((-del_E)/temperature) >= np.random.uniform(0, 1)) ):
            current_energy += del_E
            config[spin_x][spin_y] *= (-1)
            mag += 2 * config[spin_x][spin_y]

    return current_energy, mag, config


def magnetization(config):
    return np.sum(config)

N = 64
mc_step = 10000
Temp = np.linspace(0.1, 5.1, 50)
Temp = np.array(Temp)

Energy_tt = np.zeros(len(Temp))
Mag_tt = np.zeros(len(Temp))
C_tt = np.zeros(len(Temp))
X_tt = np.zeros(len(Temp))


for i in tqdm(range(len(Temp))):
    E = 0
    M = 0
    E2 = 0
    M2 = 0
    temp = Temp[i]
    config = 2*np.random.randint(2, size=(N, N)) - 1
    current_energy = calculate_energy(config)
    current_magentization = magnetization(config)
    Mag_mcStep = np.zeros(mc_step)
    for step in range(mc_step):
        en, mag, config = monte_carlo(config, temp, current_energy, current_magentization)
        current_energy = en
        current_magentization = mag
        Mag_mcStep[step] += (mag/(N**2))
        E += en
        E2 += en*en
        M += mag
        M2 += mag**2
    np.savetxt(f'./mag_mcstep/{temp}.txt', Mag_mcStep)
    E /= mc_step
    M /= mc_step
    E2 /= mc_step
    M2 /= mc_step
    beta = 1/temp
    Energy_tt[i] += E/(N**2)
    Mag_tt[i] += M/(N**2)
    C_tt[i] += ( (beta**2) * (E2 - (E**2)) )/(N**2)
    X_tt[i] += (  beta * (M2 - (M**2)) )/(N**2)

np.savetxt("Mag.txt", Mag_tt)
np.savetxt("C.txt", C_tt)
np.savetxt("X.txt", X_tt)


fig1 = figure(figsize=(18, 10))
fig1.suptitle(r'$2D$ $Ising$ $Model$ $for$ $Lattice$ $Size$ $20x20$, $B_{ext}$ $=$ 0 ', fontsize=20)

ax1 = fig1.add_subplot(221)
# plt.xlim(1, 3)
plt.ylabel("Energy", fontsize=10)
plt.xlabel("Temperature", fontsize=10)
plt.grid()
plt.plot(Temp, Energy_tt, 'o', color='blue')

ax2 = fig1.add_subplot(222)
# plt.xlim(1, 3)
plt.ylabel("Magnetization", fontsize=10)
plt.xlabel("Temperature", fontsize=10)
plt.grid()
plt.plot(Temp,np.abs(Mag_tt), 'o', color='red')

ax3 = fig1.add_subplot(223)
# plt.xlim(2, 3)
plt.ylabel("Specific Heat", fontsize=10)
plt.xlabel("Temperature", fontsize=10)
plt.plot(Temp, C_tt, 'o', color='green')
plt.grid()

ax4 = fig1.add_subplot(224)
# plt.xlim(2, 3)
plt.ylabel("Susceptibility", fontsize=10)
plt.xlabel("Temperature", fontsize=10)
plt.plot(Temp, X_tt, 'o', color='purple')
plt.grid()

plt.savefig("results.png")
plt.close()
