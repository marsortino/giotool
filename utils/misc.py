import subprocess
import sys
import pandas as pd
import numpy as np
import torch.nn as nn
from torch import load, tensor, float32
from itertools import product
from datetime import datetime 

import os

UTILS_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(UTILS_DIR, 'models')



def get_running_settings():

    settings_table = {
        'nprocs': [4,8,16,32],
        'maxblocks': [500,1000]
    }

    check = True
    while check:
        try:
            nprocs = int(input("Provide the number of cores that you intend to use:"))
            maxblocks = int(input("Provide the number of blocks per core you intend to use:"))
            check = False

            if nprocs not in settings_table['nprocs'] or maxblocks not in settings_table['maxblocks']:
                print("\nError. Value not yet available.")
                print("nprocs: 4, 8, 16, 32")
                print("maxblocks: 500, 1000\n")
                check = True
                #sys.stderr.write("Error. Value not yet available.\n")
                #sys.stderr.write("nprocs: 4, 8, 16, 32\n")
                #sys.stderr.write("maxblocks: 500, 1000\n")
                #sys.exit(1)            
        
        except ValueError:
            print("You have to provide a number.\n")
            continue
            #sys.stderr.write("You have to provide a number.")
            #sys.exit(1)

        except KeyboardInterrupt:
            sys.exit(1)


    
    return nprocs, maxblocks

def get_current_time_and_date():

    now = datetime.now()

    # Giorno della settimana (lunedì=1, domenica=7)
    day_num = now.isoweekday()  

    # Ora attuale
    hour = now.hour

    # Fascia oraria
    if 6 <= hour < 12:
        time_block = 1  # mattina
    elif 12 <= hour < 18:
        time_block = 2  # pomeriggio
    elif 18 <= hour < 24:
        time_block = 3  # sera
    else:
        time_block = 4  # notte
    return day_num, time_block


def get_node(text_string):
    """
    Trova e restituisce la prima sequenza di caratteri (non spazi) racchiusa tra parentesi
    tonde in una stringa, SENZA usare espressioni regolari.

    Args:
        text_string (str): La stringa di testo da analizzare.

    Returns:
        str or None: La prima parola trovata tra parentesi, oppure None se non trovata.
    """

    start_paren_index = text_string.find('(')
    if start_paren_index == -1:
        return None

    end_paren_index = text_string.find(')', start_paren_index + 1)

    if end_paren_index == -1:
        return None

    content = text_string[start_paren_index + 1 : end_paren_index]

    content = content.strip()

    # Se ci sono parole nel contenuto, restituisci la prima
    if content:
        return content
    else:
        return None # Nessuna parola trovata tra le parentesi (es. "()")

def load_cpu_iostat():
    """
    Lancia iostat e prende i valori della cpu
    """
    result = subprocess.run(['iostat'], capture_output=True, text=True)
    
    lines = [line.strip() for line in result.stdout.splitlines() if line.strip()]
    i = 0
    node = get_node(lines[0])[4]
    node = 4
    chassis = [0,0,0,0]
    
    try:
        if node < 6 and node>1:
            chassis[node-2] = 1
        elif node != 6:
            sys.stderr.write("Error node number is wrong.")
    except (TypeError):
        sys.stderr.write("Got a string, expected an int. I got:" + node +'\n')
        exit(1)

    while i < len(lines):
        if lines[i].startswith('avg-cpu'):
            i += 1
            cpu_vals = list(map(float, lines[i].replace(',','.').split()))
            break
        i+=1
    return chassis, cpu_vals

def scale_x(values, scaler):
    """
    Scale values and returns the tensor containing it.
    """

    feature_cols = [
        'nprocs',
        'cb_nodes',
        'maxblocks',
        'cpu_user_start_0',   
        'cpu_nice_start_0' ,
        'cpu_system_start_0' ,
        'cpu_iowait_start_0',
        'cpu_steal_start_0'  ,
        'cpu_idle_start_0',
        "time"    ,
        "day"         
    ]  

    values_to_be_scaled = np.array(values[:11]).reshape(1,-1)
    values_to_be_scaled = pd.DataFrame(values_to_be_scaled, columns=feature_cols)
    values_scaled = scaler.transform(values_to_be_scaled)

    for value in values[11:]:
         values_scaled = np.append(values_scaled, value)

    return tensor(values_scaled, dtype=float32).unsqueeze(0)



def load_model(Input_layer=16, Hidden_layer_1=256, Hidden_layer_2=128):

    Output_layer = 1

    Model = nn.Sequential(nn.Dropout(p=0.00), nn.ReLU(),
                        nn.Linear(Input_layer, Hidden_layer_1),
                        nn.Dropout(p=0.05), nn.ReLU(),
                        nn.Linear(Hidden_layer_1, Hidden_layer_2),
                        nn.Dropout(p=0.05), nn.ReLU(),
                        nn.Linear(Hidden_layer_2, Output_layer))
    
    model_path = os.path.join(MODELS_DIR, 'best_model.pth')

    Model.load_state_dict(load(model_path,  weights_only=True))

    Model.eval()

    return Model

class InputBuilder:
    def __init__(self, scaler):
        self.scaler = scaler
        self.feature_cols = [
            'nprocs',
            'cb_nodes',
            'maxblocks',
            'cpu_user_start_0',   
            'cpu_nice_start_0',
            'cpu_system_start_0',
            'cpu_iowait_start_0',
            'cpu_steal_start_0',
            'cpu_idle_start_0',
            "time",
            "day"
        ]
        # valori di default
        self.values = {col: 0 for col in self.feature_cols}
        self.values.update({
            "status": 0,
            "chassis": []
        })

    def set(self, **kwargs):
        """Aggiorna parametri manuali."""
        self.values.update(kwargs)

    def fill_runtime_values(self, chassis, cpu_stat, time, day):
        """Inserisce valori che devono venire da funzioni esterne."""
        for k, v in zip([
            'cpu_user_start_0',
            'cpu_nice_start_0',
            'cpu_system_start_0',
            'cpu_iowait_start_0',
            'cpu_steal_start_0',
            'cpu_idle_start_0'
        ], cpu_stat):
            self.values[k] = v
        self.values["chassis"] = chassis
        self.values["time"] = time
        self.values["day"] = day

    def build_tensor(self):
        """Costruisce tensore finale."""
        vals = [self.values[c] for c in self.feature_cols]
        df_vals = pd.DataFrame([vals], columns=self.feature_cols)
        scaled = self.scaler.transform(df_vals)

        extra = [self.values["status"]] + list(self.values["chassis"])
        final_vals = np.concatenate([scaled.flatten(), extra])

        return tensor(final_vals, dtype=float32).unsqueeze(0)

    def grid_variants(self, param_grid):
        """
        Genera combinazioni valide rispettando vincoli.
        param_grid: dict es. {"nprocs": [4,8], "cb_nodes": [1,2,4]}
        """
        keys, values = zip(*param_grid.items())
        for combo in product(*values):
            candidate = dict(zip(keys, combo))

            # vincolo: cb_nodes <= nprocs/2
            if "cb_nodes" in candidate:
                if candidate["cb_nodes"] > self.values['nprocs']/2:
                    continue

            self.set(**candidate)
            yield self.build_tensor()
