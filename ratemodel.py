"""Rate-based network simulation.

Author: Asem Wardak

Use `env time python ...` to check memory and CPU time statistics on a subjob
before submitting array jobs.
All paths entered should end with a /.
Functions that accept string arguments can be run directly from the command line.

Example uses:
- submit reps to PBS server:
- submit filled reps to PBS server:
- quickfill:
    python ratemodel.py submit_fillreps 100 200 5 5 300 5 100 none none 1000000 10000 quickfill:/import/orr1/wardak/artemis/ratedatacorrelation_1000000_10000/
- run subjob:


Notes:
- .dat (tofile/fromfile) is 10x more efficient than .txt (savetxt/genfromtxt)
  but requires reshaping when importing
- Helias: N = 250, L = 10, dt = 0.05, 27 timeconstants, L_reps = 1, sigma = 0.3

"""
import numpy as np
from tqdm import tqdm

import os

# for the PBS job submission routines:
# keeps track of which script version generated the data
import sys
OUTPUT_FOLDERPREFIX = f'{sys.argv[0].split(".")[0]}_'

def run_network(M, h, nsteps, rec_nsteps, dt, phi, usetqdm=True):
    """Simulates a rate-based network, returning the local field."""
    hs = np.zeros([int(nsteps/rec_nsteps), len(h)])
    for i in tqdm(range(len(hs)),
                  leave=(usetqdm!='noleave'),
                  disable=(not usetqdm)):
        for _ in range(rec_nsteps):
            h = h + (-h + np.dot(M,phi(h)))*dt
        hs[i] = h
    return hs

def autocorrelation(xs, p):
    """Modified autocorrelation function: gets the full width at p-maximum.
    
    This returns the distribution of relaxation timescales over neurons in the
    network.
    
    # if the autocorrelation is not self-averaging over space, then taking a
    # spatial mean may not smoothen the autocorr curve over time. Hence compute
    # the FWHM of neurons directly.
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
    
# Local field autocorrelation section (relaxation time + phase diagram)

def run_subjob(N, dt, nsteps, rec_nsteps, p,
               alpha100, g100,
               path, usetqdm=False):
    """Saves network simulations for a parameter set.
    
    Assumes path has been created and ends with a /.
    This function accepts string arguments.
    """
    from time import time
    start = time()
    # Extract numeric arguments.
    N = int(N)
    dt = float(dt)
    nsteps = int(nsteps)
    rec_nsteps = int(rec_nsteps)
    p = float(p)
    alpha = int(alpha100)/100.
    g = int(g100)/100.
    # Generate initial input.
    h = np.random.randn(N)
    # Generate coupling matrix.
    from scipy.stats import levy_stable
    M = levy_stable.rvs(alpha, 0, size=[len(h),len(h)],
                                  scale=g*(0.5/len(h))**(1./alpha))
    np.fill_diagonal(M, 0)
    # Run the network, obtaining the local field time series.
    hs = run_network(M, h, nsteps, rec_nsteps, dt, np.tanh, usetqdm)
    # Generate output data.
    output = autocorrelation(hs, p)  # np.tanh(hs) for neural activity
    # Save output to file.
    rep = 0
    while 1:
        filename = f'alpha_{alpha100}_g_{g100}_rep_{rep}.dat'
        if not os.path.isfile(path+filename): break
        rep += 1
    print(f'Saving {filename}')
    # np.savetxt(path+filename, output)
    # https://mathematica.stackexchange.com/a/144772
    output.astype('float32').tofile(path+filename)
    print(f"{time()-start} seconds elapsed.")


def submit_reps(N, dt, nsteps, rec_nsteps, p,
                alpha100_min, alpha100_max, alpha100_step,
                g100_min, g100_max, g100_step,
                nreps,
                queue='defaultQ', mem='1GB',
                mode=''):
    const_args = ['run_subjob', N, dt, nsteps, rec_nsteps, p]
    #const_args[0] = 'print_args' ## debug
    # Create the output path.
    path = f"{os.path.dirname(os.path.abspath(__file__))}/{OUTPUT_FOLDERPREFIX}{'_'.join(map(str, const_args))}/"
    if ':' in mode: path = mode.split(':')[1]
    print(f'{path}')
    try: os.makedirs(path+'job')
    except FileExistsError: pass
    # Determine the PBS array data.
    alphas100 = range(int(alpha100_min), int(alpha100_max)+1, int(alpha100_step))
    gs100 = range(int(g100_min), int(g100_max)+1, int(g100_step))
    pbs_array_data = [(alpha100, g100) for alpha100 in alphas100
                                       for g100 in gs100
                                       for _ in range(int(nreps))]
    if mode.startswith('fill'):  # find which subjobs need to be completed
        total_pbs_array_data = pbs_array_data
        pbs_array_data = []
        for alpha100, g100 in tqdm(set(total_pbs_array_data)):
            for rep in range(int(nreps)):
                filename = f'alpha_{alpha100}_g_{g100}_rep_{rep}.dat'
                if not os.path.isfile(path+filename):
                    pbs_array_data.append((alpha100, g100))
    # Run the PBS array.
    if mode.startswith('fillquick'):  # run the subjobs in the current process
        for pbs_array_args in tqdm(pbs_array_data):
            globals()[const_args[0]](*const_args[1:], *pbs_array_args, path, usetqdm='noleave')
    elif mode.startswith('fillpbs') or not mode:  # submit the PBS array job
        submit_arrayjob(const_args, pbs_array_data, queue, mem, path)
    else:
        print(f'{len(pbs_array_data)} elements in the proposed PBS array')

    
# Noisy binary pattern classification task described by Helias

def run_subjob_classifier(N, dt, nsteps, rec_nsteps, L, L_reps, sigma, pattern_strength, nreps,
                          alpha100, g100, seed, pattern_seed,
                          path, usetqdm=False):
    filename = f'alpha_{alpha100}_g_{g100}_seed_{seed}_patternseed_{pattern_seed}_nreps_{nreps}.dat'
    np.array([run_subjob_classifier_worker(N, dt, nsteps, rec_nsteps, L, L_reps, sigma, pattern_strength,
                                           alpha100, g100, seed, pattern_seed,
                                           path)
              for rep in tqdm(list(range(int(nreps))),disable=not usetqdm)
    ]).astype('float32').tofile(path+filename)

def run_subjob_classifier_worker(N, dt, nsteps, rec_nsteps, L, L_reps, sigma, pattern_strength,
                          alpha100, g100, seed, pattern_seed,
                          path, usetqdm=False):
    """Saves network simulations for a parameter set.
    
    Assumes path has been created and ends with a /.
    This function accepts string arguments.
    """
    from time import time
    start = time()
    # Extract numeric arguments.
    N = int(N)
    dt = float(dt)
    nsteps = int(nsteps)
    rec_nsteps = int(rec_nsteps)
    L = int(L)
    L_reps = int(L_reps)
    sigma = float(sigma)
    pattern_strength = float(pattern_strength)  # multiplies sigma too, defaults to 1 in Helias
    alpha = int(alpha100)/100.
    g = int(g100)/100.
    seed = int(seed)
    pattern_seed = int(pattern_seed)
    # Generate pattern.
    original_state = np.random.get_state()
    np.random.seed(pattern_seed)
    pattern = np.random.choice([-pattern_strength, pattern_strength], L)
    # Generate coupling matrix and background of initial state.
    from scipy.stats import levy_stable
    np.random.seed(seed)
    h = np.random.randn(N)
    M = levy_stable.rvs(alpha, 0, size=[len(h),len(h)],
                                  scale=g*(0.5/len(h))**(1./alpha))
    np.fill_diagonal(M, 0)
    # Generate random perturbation to pattern.
    np.random.set_state(original_state)
    for L_rep in range(L_reps):
        h[len(pattern)*L_rep:len(pattern)*(L_rep+1)] = pattern + sigma*pattern_strength*np.random.randn(len(pattern)) 
    # Run the network, obtaining the local field time series.
    hs = run_network(M, h, nsteps, rec_nsteps, dt, np.tanh, usetqdm)
    # Generate output data.
    output = np.tanh(hs)
#    print(output.shape)
    print(f"{time()-start} seconds elapsed.")
    return output



def submit_reps_classifier(N, dt, nsteps, rec_nsteps, L, L_reps, sigma, pattern_strength,
                           alpha100_min, alpha100_max, alpha100_step,
                           g100_min, g100_max, g100_step,
                           seed_min, seed_max,
                           pattern_seed_min, pattern_seed_max,
                           nreps,
                           queue='defaultQ', mem='1GB',
                           mode=''):
    """
    Since this is usually run with a large number (~ 1 million) of very fast
    (~ 1s) reps, to prevent excessive cluster overhead each PBS subjob will
    run all the reps for the corresponding parameter set. `nreps` is thus a
    constant arg here.
    """
    const_args = ['run_subjob_classifier', N, dt, nsteps, rec_nsteps, L, L_reps, sigma, pattern_strength]
    #const_args[0] = 'print_args' ## debug
    # Create the output path.
    path = f"{os.path.dirname(os.path.abspath(__file__))}/{OUTPUT_FOLDERPREFIX}{'_'.join(map(str, const_args))}/"
    if ':' in mode: path = mode.split(':')[1]
    print(f'{path}')
    try: os.makedirs(path+'job')
    except FileExistsError: pass
    const_args.append(nreps)
    # Determine the PBS array data.
    alphas100 = range(int(alpha100_min), int(alpha100_max)+1, int(alpha100_step))
    gs100 = range(int(g100_min), int(g100_max)+1, int(g100_step))
    seeds = range(int(seed_min), int(seed_max)+1)
    pattern_seeds = range(int(pattern_seed_min), int(pattern_seed_max)+1)
    pbs_array_data = [(alpha100, g100, seed, pattern_seed)
                      for alpha100 in alphas100
                      for g100 in gs100
                      for seed in seeds
                      for pattern_seed in pattern_seeds]
#                      for _ in range(int(nreps))]
    if mode.startswith('fill'):  # find which subjobs need to be completed
        total_pbs_array_data = pbs_array_data
        pbs_array_data = []
        for alpha100, g100, seed, pattern_seed in tqdm(set(total_pbs_array_data)):
#            for rep in range(int(nreps)):
#                filename = f'alpha_{alpha100}_g_{g100}_seed_{seed}_patternseed_{pattern_seed}_rep_{rep}.dat'
#                if not os.path.isfile(path+filename):
#                    pbs_array_data.append((alpha100, g100, seed, pattern_seed))
#                    break  # subjobs grouped with reps
            filename = f'alpha_{alpha100}_g_{g100}_seed_{seed}_patternseed_{pattern_seed}_nreps_{nreps}.dat'
            if not os.path.isfile(path+filename):
                pbs_array_data.append((alpha100, g100, seed, pattern_seed))
    # Run the PBS array.
    if mode.startswith('fillquick'):  # run the subjobs in the current process
        for pbs_array_args in tqdm(pbs_array_data):
            globals()[const_args[0]](*const_args[1:], *pbs_array_args, path, usetqdm='noleave')
    elif mode.startswith('fillpbs') or not mode:  # submit the PBS array job
        submit_arrayjob(const_args, pbs_array_data, queue, mem, path)
    else:
        print(f'{len(pbs_array_data)} elements in the proposed PBS array')
    

def run_subjob_readout(N, dt, nsteps, rec_nsteps, L, L_reps, sigma, pattern_strength, sigma_readout, sigma_readout_z, p_train,
                       pattern_seed_min, pattern_seed_max, nreps,  # custom constant args
                       alpha100, g100, seed,
                       path, usetqdm=False):
    from time import time
    start = time()
    # Extract numeric arguments.
    N = int(N)
    dt = float(dt)
    nsteps = int(nsteps)
    rec_nsteps = int(rec_nsteps)
#    L = int(L)
#    L_reps = int(L_reps)
#    sigma = float(sigma)
    sigma_readout = float(sigma_readout)
    sigma_readout_z = float(sigma_readout_z)  # set to 0.2 in helias's fig10b.py
    p_train = float(p_train)  # the fraction of the data used as training data (0.8 in Helias)
    pattern_seeds = range(int(pattern_seed_min), int(pattern_seed_max))
    nreps = int(nreps)
    nreps_train = int(p_train*nreps)
    seed = int(seed)
    T = int(nsteps/rec_nsteps)
    # Load neural activity files.
    xs = np.zeros([len(pattern_seeds), nreps, T, N])
    for pattern_seed in tqdm(pattern_seeds,
                             disable=(not usetqdm), leave=(usetqdm!='noleave')):
#        for rep in range(nreps):
#            filename = f'alpha_{alpha100}_g_{g100}_seed_{seed}_patternseed_{pattern_seed}_rep_{rep}.dat'
#            xs[pattern_seed, rep] = np.fromfile(path+filename, 'float32').reshape([T, N])
        filename = f'alpha_{alpha100}_g_{g100}_seed_{seed}_patternseed_{pattern_seed}_nreps_{nreps}.dat'
        xs[pattern_seed] = np.fromfile(path+filename, 'float32').reshape([nreps, T, N])
    # Shuffle class labels for baseline.
    xs_shuffled = xs.copy()[:,:nreps_train]
    xs_shuffled = xs_shuffled.reshape(-1, *xs.shape[2:]) # collapse the first two dimensions
    np.random.shuffle(xs_shuffled)
    xs_shuffled = xs_shuffled.reshape(len(pattern_seeds), nreps_train, T, N)
    # Prepare readout arrays.
    C = np.zeros([T, xs.shape[-1], xs.shape[-1]])  # correlation matrix
    w = np.zeros([T, len(pattern_seeds), xs.shape[-1]])  # readout weights
    norms = np.zeros([T, len(pattern_seeds)])  # norms of readout weights
    w_shuffled = np.zeros([T, len(pattern_seeds), xs.shape[-1]])
    accuracies = np.zeros(T)
    accuracies_shuffled = np.zeros(T)
    distances = np.zeros([2, T])  # Hamming signal and noise distances
#    distances_counts = np.zeros(2)
    for t in tqdm(range(T), disable=(not usetqdm), leave=(usetqdm!='noleave')):
        # Hamming distances
        '''
        for p1 in range(len(pattern_seeds)):
            for r1 in range(nreps):
                for p2 in range(len(pattern_seeds)):
                    for r2 in range(nreps):
                        H_12 = np.linalg.norm(xs[p1,r1,t] - xs[p2,r2,t])
                        distances[p1==p2, t] += H_12  # signal if p1 != p2
                        distances_counts[p1==p2] += 1
        distances = distances / np.transpose([distances_counts])
        '''
        # for brevity use the first rep (signal)
        # Measure signal dimensionality.
        for p1 in range(len(pattern_seeds)):
            for p2 in range(p1+1, len(pattern_seeds)):
                distances[0, t] += np.linalg.norm(xs[p1,0,t] - xs[p2,0,t])**2
        # Measure noise dimensionality.
        for p in range(len(pattern_seeds)):
            for r in range(1,nreps_train):
                distances[1, t] += np.linalg.norm(xs[p,0,t] - xs[p,r,t])**2
        # correlation matrix
        for pattern_seed in range(len(pattern_seeds)):
            for rep in range(nreps_train):
                C[t] += np.outer(xs[pattern_seed, rep, t], xs[pattern_seed, rep, t])
        C[t] /= nreps_train
        # training the readout
        for pattern_seed in range(len(pattern_seeds)):
            w[t, pattern_seed] = np.dot(np.linalg.pinv(C[t]),
                                        np.mean(xs[pattern_seed, :nreps_train, t],
                                                axis=0))
            norms[t, pattern_seed] = np.linalg.norm(w[t, pattern_seed])
            w_shuffled[t, pattern_seed] = np.dot(np.linalg.pinv(C[t]),
                                        np.mean(xs_shuffled[pattern_seed, :nreps_train, t],
                                                axis=0))
        # classification
        for pattern_seed in range(len(pattern_seeds)):
            for rep in range(nreps_train, nreps):
                signals = np.dot(w[t], xs[pattern_seed, rep, t] + sigma_readout*np.random.randn(xs.shape[-1])) + sigma_readout_z*np.random.randn(len(pattern_seeds))  # a vector of readout signal strengths over pattern classifier
                if signals.argmax() == pattern_seed: accuracies[t] += 1
                signals_shuffled = np.dot(w_shuffled[t], xs[pattern_seed, rep, t] + sigma_readout*np.random.randn(xs.shape[-1])) + sigma_readout_z*np.random.randn(len(pattern_seeds))
                if signals_shuffled.argmax() == pattern_seed: accuracies_shuffled[t] += 1
    accuracies /= len(pattern_seeds)*(nreps-nreps_train)
    accuracies_shuffled /= len(pattern_seeds)*(nreps-nreps_train)
    # Save accuracies to file.
    filename = f'alpha_{alpha100}_g_{g100}_seed_{seed}_accuracies_{sigma_readout}_{sigma_readout_z}.dat'  # move to beginning where variables are still strings?
    print(f'Saving {filename}')
    accuracies.astype('float32').tofile(path+filename)
    filename = f'alpha_{alpha100}_g_{g100}_seed_{seed}_accuracies_shuffled_{sigma_readout}_{sigma_readout_z}.dat'  # move to beginning where variables are still strings?
    print(f'Saving {filename}')
    accuracies_shuffled.astype('float32').tofile(path+filename)
    # Save norms to file.
    filename = f'alpha_{alpha100}_g_{g100}_seed_{seed}_norms.dat'
    print(f'Saving {filename}')
    norms.astype('float32').tofile(path+filename)
    # Save Hamming distances to file.
    filename = f'alpha_{alpha100}_g_{g100}_seed_{seed}_distances.dat'
    print(f'Saving {filename}')
    distances[0] /= (len(pattern_seeds)-1)*len(pattern_seeds)/2
    distances[1] /= (nreps_train-1)*len(pattern_seeds)
    distances.astype('float32').tofile(path+filename)
    print(f"{time()-start} seconds elapsed.")


def submit_reps_readout(N, dt, nsteps, rec_nsteps, L, L_reps, sigma, pattern_strength, sigma_readout, sigma_readout_z, p_train,
                        alpha100_min, alpha100_max, alpha100_step,
                        g100_min, g100_max, g100_step,
                        seed_min, seed_max,
                        pattern_seed_min, pattern_seed_max,  # this is passed in as a constant arg
                        nreps,  # this is passed in as a constant arg
                        queue='defaultQ', mem='1GB',
                        mode=''):
    const_args = ['run_subjob_classifier', N, dt, nsteps, rec_nsteps, L, L_reps, sigma, pattern_strength]
    # Create the output path.
    path = f"{os.path.dirname(os.path.abspath(__file__))}/{OUTPUT_FOLDERPREFIX}{'_'.join(map(str, const_args))}/"
    if ':' in mode: path = mode.split(':')[1]
    print(f'{path}')
    try: os.makedirs(path+'job')
    except FileExistsError: pass
    const_args[0] = 'run_subjob_readout'  # run the readout with the classifier's path
    const_args.extend([sigma_readout, sigma_readout_z, p_train,
                       pattern_seed_min, pattern_seed_max, nreps])
    # Determine the PBS array data.
    alphas100 = range(int(alpha100_min), int(alpha100_max)+1, int(alpha100_step))
    gs100 = range(int(g100_min), int(g100_max)+1, int(g100_step))
    seeds = range(int(seed_min), int(seed_max)+1)
    # pattern_seeds = range(int(pattern_seed_min), int(pattern_seed_max)+1)
    pbs_array_data = [(alpha100, g100, seed) # , pattern_seed)
                      for alpha100 in alphas100
                      for g100 in gs100
                      for seed in seeds
#                      for pattern_seed in pattern_seeds
#                      for _ in range(int(nreps))
                     ]
    if mode.startswith('fill'):  # find which subjobs need to be completed
        total_pbs_array_data = pbs_array_data
        pbs_array_data = []
        for alpha100, g100, seed in tqdm(set(total_pbs_array_data)):
            filename = f'alpha_{alpha100}_g_{g100}_seed_{seed}_accuracies_{sigma_readout}_{sigma_readout_z}.dat' #_patternseed_{pattern_seed}_rep_{rep}.dat'
            if not os.path.isfile(path+filename):
                pbs_array_data.append((alpha100, g100, seed)) #, pattern_seed))
    # Run the PBS array.
    if mode.startswith('fillquick') or mode.startswith('runquick'):  # run the subjobs in the current process
        for pbs_array_args in tqdm(pbs_array_data):
            globals()[const_args[0]](*const_args[1:], *pbs_array_args, path, usetqdm='noleave')
    elif mode.startswith('fillpbs') or not mode or mode.startswith('runpbs'):  # submit the PBS array job
        submit_arrayjob(const_args, pbs_array_data, queue, mem, path)
    else:
        print(f'{len(pbs_array_data)} elements in the proposed PBS array')

    

def submit_arrayjob(const_args, pbs_array_data, queue, mem, path=None):
    """A general PBS array job submitting function.
    
    Submits a PBS array job, each subjob calling this script with arguments
    `const_args` followed by the arguments of an element of `pbs_array_data`,
    ending with the path of the output folder:
    
        python scriptpath *const_args *(pbs_array_data[i]) path
    
    This is equivalent to calling the function named by const_args[0] (such as
    "run_subjob") with arguments ending in `path`.
    
    The subjob argument ordering follows the input-processing-output paradigm.
    
    Args:
        `const_args` has the form ('function_name', 'arg1', 'arg2', ...)
        `pbs_array_data` is an array of argument tuples
        `queue`, `mem` are PBS queue allocations
        `path` is also necessary to determine the PBS job output location
    """
    if not path:
        # Create output folder.
        path = f"{os.path.dirname(os.path.abspath(__file__))}/{'_'.join(map(str, const_args))}/"
        print(f'Creating {path}')
        try: os.makedirs(path+'job')
        except FileExistsError: print('path exists')
    # Submit array job.
    from random import shuffle
    shuffle(pbs_array_data)  # for the scavenger queue
    print(f"Submitting {len(pbs_array_data)} subjobs")
    # PBS array jobs are limited to 1000 subjobs by default
    pbs_array_data_chunks = [pbs_array_data[x:x+1000]
                             for x in range(0, len(pbs_array_data), 1000)]
    if len(pbs_array_data_chunks[-1]) == 1:  # array jobs must have length >1
        pbs_array_data_chunks[-1].insert(0, pbs_array_data_chunks[-2].pop())
    for i, pbs_array_data_chunk in enumerate(pbs_array_data_chunks):
        PBS_SCRIPT = f"""<<'END'
            #!/bin/bash
            #PBS -N ratemodel -P THAP
            #PBS -q {queue}
            #PBS -V
            #PBS -m n
            #PBS -o {path+'job'} -e {path+'job'}

            #PBS -l select=1:ncpus=1:mem={mem}
            #PBS -l walltime=23:59:00
            #PBS -J {1000*i}-{1000*i + len(pbs_array_data_chunk)-1}

            cd $PBS_O_WORKDIR

            args=($(python -c "import sys;print(' '.join(map(str, {pbs_array_data_chunk}[int(sys.argv[1])-{1000*i}])))" $PBS_ARRAY_INDEX))

            script_command="{os.path.abspath(__file__)} {' '.join(map(str, const_args))} ${{args[*]}} {path}"
            
            echo   $script_command
            python $script_command
END"""
        os.system(f'qsub {PBS_SCRIPT}')
        #print(PBS_SCRIPT)


def print_args(*args):
    """Debug"""
    print(args)

def plot_vector(fname):
    """Debug"""
    data = np.fromfile(fname, 'float32')
    data = data.reshape(1000, -1)  # (nsteps/rec_nsteps, len(h))
    print(data.shape)
    np.savetxt('temp/nice.txt', data)
    import matplotlib.pyplot as plt
    plt.plot(data)
    plt.show()


if __name__ == '__main__':
    import sys
    if len(sys.argv) < 2:
        print('Usage: python %s FUNCTION_NAME ARG1 ... ARGN' % sys.argv[0])
        quit()
    globals()[sys.argv[1]](*sys.argv[2:])

