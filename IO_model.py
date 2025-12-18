import torch
import joblib
from utils import misc as m
import numpy as np
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UTILS_DIR = os.path.join(BASE_DIR, 'utils')
MODELS_DIR = os.path.join(UTILS_DIR, 'models')
xscaler_path_mpi = os.path.join(MODELS_DIR, 'xscaler_mpi.pkl')
yscaler_path_mpi = os.path.join(MODELS_DIR, 'yscaler_mpi.pkl')


xscaler_path_runtime = os.path.join(MODELS_DIR, 'xscaler_runtime.pkl')
yscaler_path_runtime = os.path.join(MODELS_DIR, 'xscaler_runtime.pkl')

nprocs, maxblocks = m.get_running_settings()

chassis, cpu_stat = m.load_cpu_iostat()

time, day = m.get_current_time_and_date()

procs_to_check = []

if nprocs*maxblocks != 2000 and nprocs*maxblocks != 32000: # controlla se siamo nei casi limiti (4,500) e (32,1000)
    if maxblocks == 500:
        procs_to_check.append((nprocs, 500))
        procs_to_check.append((nprocs/2, 1000))
    if maxblocks == 1000:
        procs_to_check.append((nprocs*2, 500))
        procs_to_check.append((nprocs, 1000))
else:
    procs_to_check.append((nprocs, maxblocks))

xscaler_mpi = joblib.load(xscaler_path_mpi)
xscaler_runtime = joblib.load(xscaler_path_runtime)

yscaler_mpi = joblib.load(yscaler_path_mpi)
yscaler_runtime = joblib.load(yscaler_path_runtime)

ib = m.InputBuilder(xscaler_mpi)
ib.fill_runtime_values(chassis, cpu_stat, time, day)

with torch.no_grad():

    Model = m.load_model('mpi')

    high_performance = [0,0,0,0,0]

    for couple in procs_to_check:
        ib.set(nprocs=couple[0], maxblocks=couple[1])
        for t in ib.grid_variants({"cb_nodes":[1,2,8,16], "status":[0,1]}):
            y = Model(t)
            tensor_runtimes = t.detach().numpy()
            if np.isclose(tensor_runtimes[0][15], 0.99602291, atol=1e-6):
                status = "independent"
            elif np.isclose(tensor_runtimes[0][15], -1.00399297, atol=1e-6):
                status = "collective"
            else:
                print('Something went wrong. Unable to catch writing mode')
                print(tensor_runtimes[0][15])
                exit()
            
            tensor_runtimes = xscaler_mpi.inverse_transform(tensor_runtimes[0].reshape(1,-1))[0]

            orig_y_pred = yscaler_mpi.inverse_transform(y.detach().numpy()[0][0].reshape(-1,1))

            if orig_y_pred[0][0] > high_performance[0]:
                high_performance[0] = orig_y_pred[0][0]
                high_performance[1] = tensor_runtimes[0]
                high_performance[2] = tensor_runtimes[1]
                high_performance[3] = status
                high_performance[4] = tensor_runtimes[2]

            #print("unscaled MPI output: ", yscaler_mpi.inverse_transform(orig_y_pred[0][0].reshape(-1,1))[0][0])
            print('runtime MPI:', orig_y_pred[0][0], "MiB/s")

            if status == 'independent':
                print(f"settings -- nprocs {np.round(tensor_runtimes[0], 3)} -- status {status} -- maxblocks {tensor_runtimes[2]}")
            else:
                print(f"settings -- nprocs {np.round(tensor_runtimes[0], 3)} -- cbnodes {tensor_runtimes[1]} -- status {status} -- maxblocks {tensor_runtimes[2]}")


### TEMPORARY


ib = m.InputBuilder(xscaler_runtime)
ib.fill_runtime_values(chassis, cpu_stat, time, day)

print("\n*******\n")

with torch.no_grad():

    Model = m.load_model('runtime')

    low_runtime = [1e6,0,0,0,0]
    
    for couple in procs_to_check:
        ib.set(nprocs=couple[0], maxblocks=couple[1])
        for t in ib.grid_variants({"cb_nodes":[1,2,8,16], "status":[0,1]}):
            y = Model(t)
            tensor_runtimes = t.detach().numpy()
            if np.isclose(tensor_runtimes[0][15], 0.99602291, atol=1e-6):
                status = "independent"
            elif np.isclose(tensor_runtimes[0][15], -1.00399297, atol=1e-6):
                status = "collective"
            else:
                print('Something went wrong. Unable to catch writing mode')
                print(tensor_runtimes[0][15])
                exit()
            tensor_runtimes = xscaler_runtime.inverse_transform(tensor_runtimes[0].reshape(1,-1))[0]

            orig_y_pred = yscaler_mpi.inverse_transform(y.detach().numpy()[0][0].reshape(-1,1))

            if orig_y_pred[0][0] < low_runtime[0]:
                low_runtime[0] = orig_y_pred[0][0]
                low_runtime[1] = tensor_runtimes[0]
                low_runtime[2] = tensor_runtimes[1]
                low_runtime[3] = status
                low_runtime[4] = tensor_runtimes[2]

            #print("unscaled runtime output: ", yscaler_runtime.inverse_transform(orig_y_pred[0][0])[0])
            print('runtime output:', orig_y_pred[0][0], "s")

            if status == 'independent':
                print(f"settings -- nprocs {np.round(tensor_runtimes[0], 2)} -- status {status} -- maxblocks {tensor_runtimes[2]}")
            else:
                print(f"settings -- nprocs {np.round(tensor_runtimes[0], 2)} -- cbnodes {np.round(tensor_runtimes[1], 2)} -- status {status} -- maxblocks {tensor_runtimes[2]}")

print("\n ******** \n Final Result:")

if high_performance[3] == 'independent':
    print("\n ******* \n Best settings that estimate is possible to achieve highest I/O performance (", high_performance[0] ,"Mbi/s ) are: \n nprocs:", np.round(high_performance[1], 2), '\n status:', high_performance[3], "\n maxblocks", high_performance[4] )
else:
    print("\n ******* \n Best settings that estimate is possible to achieve highest I/O performance (", high_performance[0] ,"Mbi/s ) are: \n nprocs:", np.round(high_performance[1], 2), "\n cb_nodes:", np.round(high_performance[2], 2), '\n status:', high_performance[3], "\n maxblocks", high_performance[4] )


if low_runtime[3] == 'independent':
    print("\n ******* \n Best settings that estimate is possible to achieve lowest runtime (", np.round(low_runtime[0], 6) ,"s ) are: \n nprocs:", np.round(low_runtime[1], 2), '\n status:', low_runtime[3], "\n maxblocks", low_runtime[4] )
else:
    print("\n ******* \n Best settings that estimate is possible to achieve lowest runtime (", np.round(low_runtime[0], 6) ,"s ) are: \n nprocs:", np.round(low_runtime[1], 2), "\n cb_nodes:", np.round(low_runtime[2], 2), '\n status:', low_runtime[3], "\n maxblocks", low_runtime[4] )