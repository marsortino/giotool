import torch
import joblib
from utils import misc as m
import numpy as np
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UTILS_DIR = os.path.join(BASE_DIR, 'utils')
MODELS_DIR = os.path.join(UTILS_DIR, 'models')
xscaler_path = os.path.join(MODELS_DIR, 'scaler.pkl')
#yscaler_path = os.path.join(MODELS_DIR, 'scaler_Y.pkl')

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

xscaler = joblib.load(xscaler_path)

ib = m.InputBuilder(xscaler)
ib.fill_runtime_values(chassis, cpu_stat, time, day)

Model = m.load_model()

high_performance = [0,0,0,0,0]

with torch.no_grad():

    for couple in procs_to_check:
        ib.set(nprocs=couple[0], maxblocks=couple[1])
        for t in ib.grid_variants({"cb_nodes":[1,2,8,16], "status":[0,1]}):
            y = Model(t)
            tensor_runtimes = t.detach().numpy()
            if np.isclose(tensor_runtimes[0][15], -0.9312661, atol=1e-6):
                status = "independent"
            elif np.isclose(tensor_runtimes[0][15], 1.0738069, atol=1e-6):
                status = "collective"
            else:
                print('Something went wrong. Unable to catch writing mode')
                print(tensor_runtimes[0][15])
                exit()
            tensor_runtimes = xscaler.inverse_transform(tensor_runtimes[0].reshape(1,-1))[0]

            orig_y_pred = y.detach().numpy()

            if orig_y_pred[0][0] > high_performance[0]:
                high_performance[0] = orig_y_pred[0][0]
                high_performance[1] = tensor_runtimes[0]
                high_performance[2] = tensor_runtimes[1]
                high_performance[3] = status
                high_performance[4] = tensor_runtimes[2]

            print("unscaled output: ", orig_y_pred[0][0])

            if status == 'independent':
                print(f"settings -- nprocs {tensor_runtimes[0]} -- status {status} -- maxblocks {tensor_runtimes[2]}")
            else:
                print(f"settings -- nprocs {tensor_runtimes[0]} -- cbnodes {tensor_runtimes[1]} -- status {status} -- maxblocks {tensor_runtimes[2]}")

if high_performance[3] == 'independent':
    print("\n ******* \n Best settings that estimate is possible to achieve", high_performance[0] ," Mbi/s are: \n nprocs:", high_performance[1], '\n status:', high_performance[3], "\n maxblocks", high_performance[4] )
else:
    print("\n ******* \n Best settings that estimate is possible to achieve", high_performance[0] ," Mbi/s are: \n nprocs:", high_performance[1], "\n cb_nodes:", high_performance[2], '\n status:', high_performance[3], "\n maxblocks", high_performance[4] )