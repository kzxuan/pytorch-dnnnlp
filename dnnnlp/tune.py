import time
import torch
import torch.multiprocessing as mp
from dnnnlp.utils.predict_eval import tabular, average


def _pool_pack_func(gpus, func, **run_params):
    while not any(gpus):
        time.sleep(0.1)
    else:
        n = gpus.index(True)
        gpus[n] = False
    get = func(device=n, **run_params)
    gpus[n] = True
    return get


def _device_count():
    return torch.cuda.device_count()


def average_several_run(run, args, times=5, n_paral=2, verbose=1, **run_params):
    """Get average result after several running.

    Args:
        run [function]: model run function which returns a result dict by 'prfacc_1d/2d'
        args [argparse.Namespace]: set arguments
        times [int]: run several times for average
        n_paral [int]: number of processes
        verbose [int]: verbose level
        run_params [parameter]: some parameters for run function
    """
    pool = mp.Pool(processes=1)
    check = pool.apply_async(_device_count)
    device_count = check.get()

    if args.n_gpu > 0:
        assert args.n_gpu <= device_count, ValueError("Available GPUs is less than 'n_gpu'.")
        assert 0 < n_paral <= args.n_gpu, ValueError("Number of processes should be no more than 'n_gpu'.")

    scores, processes = [], []
    pool = mp.Pool(processes=n_paral)

    if n_paral > 0 and args.n_gpu > 0:
        manager = mp.Manager()
        gpus = manager.list([True] * n_paral)
        for _ in range(times):
            processes.append(pool.apply_async(_pool_pack_func, args=(gpus, run), kwds=run_params))
    else:
        for _ in range(times):
            run_params.update({'device': torch.device('cpu')})
            processes.append(pool.apply_async(run, kwds=run_params))

    for p in processes:
        result = p.get()
        scores.append(result)

    print("- Average score after {} rounds:".format(times))
    aver = average(*scores)
    print(tabular(aver))
