# %%
import numpy as np
import matplotlib.pyplot as plt

# configure plots
plt.rcParams['xtick.minor.visible'], plt.rcParams['xtick.top'] = True,True
plt.rcParams['ytick.minor.visible'], plt.rcParams['ytick.right'] = True,True
plt.rcParams['xtick.direction'], plt.rcParams['ytick.direction'] = 'in','in'


# load data for star
star, name = np.loadtxt('http://www.astro.lu.se/Education/utb/ASTM21/P4data/hd000142.txt'), '142'
# star, name = np.loadtxt('http://www.astro.lu.se/Education/utb/ASTM21/P4data/hd027442.txt'), '27442'
# star, name = np.loadtxt('http://www.astro.lu.se/Education/utb/ASTM21/P4data/hd102117.txt'), '102117'
star, name = np.loadtxt('./hd000142.txt'), '142'
# star, name = np.loadtxt('./hd027442.txt'), '27442'
# star, name = np.loadtxt('./hd102117.txt'), '102117'

t = star[:,0]   # times of measurements
rv = star[:,1]  # radial velocity
var = star[:,2] # uncertainty


# plot given data
plt.figure(figsize=(10, 4))
plt.errorbar(t, rv, yerr=var, fmt='o')
plt.xlabel('time (days)')
plt.ylabel('radial velocity (m/s)')
plt.title(f'HD {name}')
    
plt.savefig(f'{name}data', bbox_inches='tight')

# %%


def LombScargle(t, rv, var, omega):    
    ''' This calculates the Lomb Scargle periodogram for data over a range of frequencies '''
    
    
    h = (rv - np.mean(rv)) / var    # subract mean and divide by uncertainty 
    sigma = np.sum(( h )  ** 2)     # calculate total variance
    
    
    # calculate tau
    tau = [np.arctan(np.sum(np.sin(2 * omega[i] * t) / var ** 2 ) /
                     np.sum(np.cos(2 * omega[i] * t) / var ** 2  ) ) / (2 * omega[i]) for i in range(len(omega))]
    
    # calculate and return the normalized periodogram
    return [(((np.sum(h * np.cos(omega[i] * (t - tau[i])))) ** 2 /
              (np.sum(np.cos(freq[i] * (t - tau[i])) ** 2))) +
             ((np.sum(h * np.sin(omega[i] * (t - tau[i])))) ** 2 /
              (np.sum(np.sin(freq[i] * (t - tau[i])) ** 2)))) / (sigma) for i in range(len(omega))]



periods = np.linspace(1, 1000, 10000)   # create range of periods to be investigated
freq = 1 / periods                      # create frequencies
omega = 2 * np.pi * freq                # create angular frequencies

periodogram = LombScargle(t, rv, var, omega)    # call function for periodogram

# plot results for periods
plt.figure(figsize=(10, 4))
plt.plot(periods, periodogram)
plt.ylim(0, 1)
plt.xlim(np.amin(periods), np.amax(periods))
plt.xlabel('period (days)')
plt.ylabel('P(T)')
plt.title(f'HD {name}')
    
plt.savefig(f'{name}periodogram', bbox_inches='tight')

# plot results for frequencies
plt.figure(figsize=(10, 4))
plt.plot(freq, periodogram)
plt.ylim(0, 1)
plt.xlim(np.amin(freq), 0.1)
plt.xlabel('freq (1/day)')
plt.ylabel('P(f)')
plt.title(f'HD {name}')
    
plt.savefig(f'{name}periodogramfreq', bbox_inches='tight')


# %%

period = float(periods[np.where(np.amax(periodogram) == periodogram)]) # most likely period
f = 1 / (period)                                                       # most likely frequency

t_fold = t - np.floor(t / period) * period   # fold data using period found from periodogram

# plot folded data
plt.figure(figsize=(10, 4))   
plt.errorbar(t_fold, rv, yerr=var, fmt='o')
plt.xlabel('time (days)')
plt.ylabel('velocity (m/s)')
plt.title(f'HD {name}')

plt.savefig(f'{name}folded', bbox_inches='tight')


# %%

max_prob = []   # initialize array for random data max probabilities

# change this to speed up test
num_montecarlo = 1000

for i in range(num_montecarlo):
    # create random data
    rand_data = rv + np.random.normal(0, var, len(rv))
        
    # call Lomb Scargle function on random data
    pg = LombScargle(t, rand_data - np.mean(rand_data), var, omega)
    
    # find most likely period for random data
    max_prob.append(np.amax(pg))

# calculate number of independant frequencies
M = np.log(1 - np.mean(max_prob)) / np.log(1 - np.exp(-0.5))

# use M to calculate significance of most likely period
significance = - np.log(1 - np.exp(np.log(1 - np.amax(periodogram)) / M))

# array of values to be calculated
z = np.array([0.5, 0.1, 0.01, significance])

# calculate significance of values in z
prob = 1 - ( 1 -  np.exp(- z) ) ** M

# plot results
plt.figure(figsize=(10, 4))
plt.plot(periods, periodogram)
plt.xlabel('period (days)')
plt.ylabel('P(T)')
plt.ylim(0, 1)
plt.xlim(np.amin(periods), np.amax(periods))
plt.title(f'HD {name}')

# add lines showing significance
plt.axhline(prob[3], linewidth=.8, color='black', label=f'{int((1 - significance) * 100)}%')
plt.axhline(prob[2], linewidth=.8, color='b', linestyle="--", label='99%')
plt.axhline(prob[1], linewidth=.8, color='g', linestyle="--", label='90%')
plt.axhline(prob[0], linewidth=.8, color='r', linestyle="--", label='50%')
plt.legend()

plt.savefig(f'{name}sig', bbox_inches='tight')