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
    import torch
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print(device)
    M = torch.tensor(M, device=device)
    h = torch.tensor(h, device=device)
    phi = torch.tanh
    #hs = np.zeros([int(nsteps/rec_nsteps), len(h)])
    hs = torch.zeros([int(nsteps/rec_nsteps), len(h)], device=device)
    for i in tqdm(range(len(hs)),
                  leave=(usetqdm!='noleave'),
                  disable=(not usetqdm)):
        for _ in range(rec_nsteps):
            #h = h + (-h + np.dot(M,phi(h)))*dt
            h = h + (-h + torch.matmul(M,phi(h)))*dt
        hs[i] = h
    #return hs
    return hs.cpu().numpy()

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

def run_subjob_reps(N, dt, nsteps, rec_nsteps, p,
                    alpha100_min, alpha100_max, alpha100_step,
                    nreps,
                    *args):
    alphas100 = range(int(alpha100_min), int(alpha100_max)+1, int(alpha100_step))
    for alpha100 in alphas100:
        for _ in range(int(nreps)):
            run_subjob(N, dt, nsteps, rec_nsteps, p,
                       alpha100, *args)

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
    # gpu
    const_args[0] = 'run_subjob_reps'
    const_args.extend([alpha100_min, alpha100_max, alpha100_step, nreps])
    nreps = 1
    # Determine the PBS array data.
    alphas100 = range(int(alpha100_min), int(alpha100_max)+1, int(alpha100_step))
    gs100 = range(int(g100_min), int(g100_max)+1, int(g100_step))
#    pbs_array_data = [(alpha100, g100) for alpha100 in alphas100
#                                       for g100 in gs100
#                                       for _ in range(int(nreps))]
    pbs_array_data = [(g100,) for g100 in gs100]
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

            #PBS -l select=1:ncpus=1:mem={mem}:ngpus=1
            #PBS -l walltime=23:59:00
            #PBS -J {1000*i}-{1000*i + len(pbs_array_data_chunk)-1}
            
            module load python/3.8.2 magma/2.5.3

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

