"""Rate-based network simulation: finite-size scaling of network sims.

Author: Asem Wardak
"""
import numpy as np
from tqdm import tqdm

import os

BATCH = 'PBS_JOBID' in os.environ

def run_network(M, h, nsteps, rec_nsteps, dt, phi):
    """Simulates a rate-based network, returning the local field."""
    hs = np.zeros([int(nsteps/rec_nsteps), len(h)])
    for i in tqdm(range(len(hs)), disable=BATCH):
        for _ in range(rec_nsteps):
            h = h + (-h + np.dot(M,phi(h)))*dt
        hs[i] = h
    return hs

def autocorrelation(xs, p):
    """Modified autocorrelation function: gets the full width at p-maximum.
    
    This returns the distribution of relaxation timescales over neurons in the
    network.
    """
    from scipy.signal import correlate
    result = np.zeros(xs.shape[1])
    for neuron in range(len(result)):
        autocorr = correlate(xs[:,neuron], xs[:,neuron])
        midpoint = xs.shape[0]-1
        p_maximum = p*autocorr[midpoint]
        for lag in range(xs.shape[0]):
            if autocorr[midpoint+lag] < p_maximum: break
        result[neuron] = lag
    return result

def run_sims(N, dt, nsteps, rec_nsteps, p, alpha, g):
    # Generate initial input.
    h = np.random.randn(N)
    # Generate coupling matrix.
    from scipy.stats import levy_stable
    M = levy_stable.rvs(alpha, 0, size=[len(h),len(h)],
                                  scale=g*(0.5/len(h))**(1./alpha))
    np.fill_diagonal(M, 0)
    # Run the network, obtaining the local field time series.
    hs = run_network(M, h, nsteps, rec_nsteps, dt, np.tanh)
    # Generate output data.
    output = autocorrelation(hs, p)  # np.tanh(hs) for neural activity
    return output

def save_scaling(dt, nsteps, rec_nsteps, p,
                 N, alpha100, g100, rep,
                 path):
    folder = f'N_{N}_alpha_{alpha100}_g_{g100}/'
    filename = f'rep_{rep}.dat'
    if os.path.exists(path+folder+filename):
        print(f'{path+folder+filename} exists')
        return
    # Extract numeric arguments.
    N = int(N)
    dt = float(dt)
    nsteps = int(nsteps)
    rec_nsteps = int(rec_nsteps)
    p = float(p)
    alpha = int(alpha100)/100.
    g = int(g100)/100.
    # Run sims.
    output = run_sims(N, dt, nsteps, rec_nsteps, p, alpha, g)
    # Save output to file.
    try: os.makedirs(path+folder)
    except FileExistsError: pass
    output.astype('float32').tofile(path+folder+filename)

# check which files are missing:
#   ls -v | xargs -n1 sh -c 'echo $0; ls $0 | wc'

def submit_scaling(*args):
    from qsub import qsub
    import sys
    pbs_array_data = [(N, alpha100, g100, rep)
                      for N in range(10, 101, 10)
                      for alpha100 in range(120, 201, 80)
                      for g100 in range(75, 301, 75)
                      for rep in range(100)
                      ]
    qsub(f'python {sys.argv[0]} save_scaling {" ".join(args)}',
         pbs_array_data,
         pass_path=True)

if __name__ == '__main__':
    import sys
    if len(sys.argv) < 2:
        print('Usage: python %s FUNCTION_NAME ARG1 ... ARGN' % sys.argv[0])
        quit()
    globals()[sys.argv[1]](*sys.argv[2:])

