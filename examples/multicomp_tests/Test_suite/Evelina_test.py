# requirements (extra):
#   matplotlib, pandas

import spynnaker8 as p
from pyNN.utility.plotting import Figure, Panel
import matplotlib.pyplot as plt
import numpy as np
from spinnman.processes import ReadRouterDiagnosticsProcess
from spinnman.transceiver import create_transceiver_from_hostname
from spinnman.constants import ROUTER_REGISTER_REGISTERS
from pacman.model.constraints.placer_constraints import ChipAndCoreConstraint, SameChipAsConstraint
import itertools
import time
import pandas as pd
import sys, os, glob
from xml.etree import ElementTree
from collections import OrderedDict

# NOTE FOR EEV: the location of the cfg file is:
# multicomp/sPyNNaker/spynnaker/PyNN/spynnaker.cfg

# This test replicates the Urbanczik-Senn experiment in the 2014 paper.
# It runs the simulation on SpiNNaker, then computes the expected values in floating point on Host side and compares
# them by plotting both.

def stringToTuple(string):
    temp = []
    for token in string.split(","):
        num = int(token.replace("(", "").replace(")", ""))
        temp.append(num)
        if ")" in token:
            res = tuple(temp)
    return res

def place_radial(pivot, n_sources, transceiver):
    if transceiver is not None:
        machine = transceiver.get_machine_details()
    #if isinstance(pivot, str):
    #    pivot = stringToTuple(pivot)
    # directions_to_links = {'E':0, 'NE':1, 'N':2, 'O':3, 'SO':4, 'S':5}
    source_chips_list = [pivot]

    if pivot == (6,7):
        allowed_directions = [4,5]
    elif pivot == (3,0):
        allowed_directions = [1,2]
    elif pivot == (7,5):
        allowed_directions = [3,4]
    else:
        allowed_directions = list(range(0,6))

    base_chip = machine.get_chip_at(pivot[0], pivot[1])
    print(base_chip.router.links)

    j = 0
    placed_sources = 0
    while placed_sources < n_sources:
        # iterate until we have placed all the sources
 #       for direction in directions_to_links:
 #           current_link_id = directions_to_links[direction]
            # try all directions
 #           if(base_chip.router.is_link(current_link_id)):
 #               link = base_chip.router.get_link(current_link_id)
        for link in base_chip.router.links:
            if (link.destination_x, link.destination_y) not in source_chips_list and\
                link.source_link_id in allowed_directions:
                source_chips_list.append((link.destination_x, link.destination_y))
                placed_sources += 1
        # iterate on neighbors
        base_chip = machine.get_chip_at(source_chips_list[j][0], source_chips_list[j][1])
        j += 1
        
    return source_chips_list[1:n_sources+1]


def place_directions(pivot, n_sources, transceiver, directions='all'):

    directions_dict = { 'all'       : ['E', 'NE', 'N', 'O', 'SO', 'S'],
                        'cardinal'  : ['E', 'N', 'O', 'S'],
                        'angle_n'   : ['E', 'NE', 'N'],
                        'angle_s'   : ['O', 'SO', 'S'],
                        'delta'     : ['NE', 'O', 'S'],
                        'poles'     : ['NE', 'SO'],
                        'south'     : ['SO', 'S']   }

    directions = directions_dict[directions]

    if transceiver is not None:
        machine = transceiver.get_machine_details()
    directions_to_links = {'E':0, 'NE':1, 'N':2, 'O':3, 'SO':4, 'S':5}
    source_chips_list = []

    base_chip = machine.get_chip_at(pivot[0], pivot[1])
    print(base_chip.router.links)

    j = 0
    for i in range(n_sources):
        current_direction_id = i % len(directions)
        current_direction = directions[current_direction_id]
        current_link_id = directions_to_links[current_direction]

        if i >= len(directions):
            base_chip_coords = source_chips_list[j]
            base_chip = machine.get_chip_at(base_chip_coords[0], base_chip_coords[1])
            j += 1

        if(base_chip.router.is_link(current_link_id)):
            link = base_chip.router.get_link(current_link_id)
            source_chips_list.append((link.destination_x, link.destination_y))
        else:
            raise SystemExit('--- Error: selected graph cannot fit on board')


    return source_chips_list


def test(target_chip, multicomp_pops_per_source, rate_source_pops, directions, n_neurons_per_multicomp_pop, sim_time, timestamp, n_pop_target, g_D=2, g_L=0.1, exc_times=[1, 2, 5, 6], inh_times=[3, 4, 5, 6], exc_r_diff=[1, 1.3, 3.3, 1.5] ,
         Ee=4.667, Ei=-0.333):

    # # Create transceiver & boot if necessary
    # transceiver = create_transceiver_from_hostname('spinnaker.cs.man.ac.uk', 5)
    # transceiver.ensure_board_is_ready()

    # if (directions == 'rad'):
    #     source_constraint_list = place_radial(target_chip, rate_source_pops, transceiver)
    # elif (directions != 'any'):
    #     source_constraint_list = place_directions(target_chip, rate_source_pops, transceiver, directions=directions)

    # if transceiver is not None:
    #     machine = transceiver.get_machine_details()

    # Simulator setup, important
    p.setup(timestep=1, min_delay=1.0, max_delay=144.0)

    synapse_amts = OrderedDict([("soma_exc", 4), ("dendrite_exc", 3), ("soma_inh", 7), ("dendrite_inh", 0)])

    # Place the sources
    pop_filter = []

    for j in range(sum(multicomp_pops_per_source)): #16
        # selected_chip = source_constraint_list[j]
        for s in synapse_amts:
            if synapse_amts[s] > 0:
                pop_filter.append(
                    p.Population(
                        n_neurons_per_multicomp_pop*synapse_amts[s],
                        p.extra_models.IFExpRateTwoComp(g_D=g_D, g_L=g_L, rate_update_threshold=0), 
                            in_partitions=[0,0,0,0], out_partitions=synapse_amts[s],
                            label='pop_filter_'+str(j)+"_"+str(s),
                            # constraints=[ChipAndCoreConstraint(x=selected_chip[0],y=selected_chip[1],p=None)]
                    )
                )

    # Place the target chip
    pop_target = []
    # Just one population (14 synapses + 1 neuron)
    pop_target.append (
        p.Population (
            n_neurons_per_multicomp_pop, 
            p.extra_models.IFExpRateTwoComp(g_D=g_D, g_L=g_L, rate_update_threshold=0),
            label='population_target',
            in_partitions=[4,3,7,0], out_partitions=1,
            # constraints=[ChipAndCoreConstraint(x=target_chip[0], y=target_chip[1], p=None)]
        ))

    # Alright so now we have to actually place some projections...
    
    # Dendritic initial weight
    weight_to_spike = 0.2
    learning_rate = 0.07

    plasticity = p.STDPMechanism(
        timing_dependence=p.extra_models.TimingDependenceMulticompBern(),
        weight_dependence=p.extra_models.WeightDependenceMultiplicativeMulticompBern(w_min=-10, w_max=10,
                                                                                     learning_rate=learning_rate),
        weight=weight_to_spike)

    synapse_types = ["soma_exc", "dendrite_exc", "soma_inh", "dendrite_inh"]

    proj = []

    pop_filter_index = 0
    syn_amts_tmp = synapse_amts.copy()
    for j in range(sum(multicomp_pops_per_source)): #16
        n=0
        for s in synapse_amts:
            if synapse_amts[s] > 0:
                pop_target_index = 0
                pop_filter_index = j*3+n
                proj.append(p.Projection(pop_filter[pop_filter_index], pop_target[pop_target_index], 
                    p.OneToOneConnector(), #synapse_type=plasticity,
                    receptor_type=s))
                n += 1
    
    # Setup diagnostic stuff
    # transceiver.clear_router_diagnostic_counters(target_chip[0],target_chip[1])
    # for i in range(rate_source_pops):
    #     c = machine.BOARD_48_CHIPS[i]
    #     transceiver.clear_router_diagnostic_counters(c[0],c[1])

    for projection in proj:
        print(projection.label)

    # Run simulation
    p.run(sim_time)

    # End simulation
    p.end()

    diagnostics = {}

    # Get whole board diagnostics
    for i in range(len(machine.BOARD_48_CHIPS)):
        c = machine.BOARD_48_CHIPS[i]
        try:
            diagnostics[c] = transceiver.get_router_diagnostics(c[0], c[1])
        except:
            diagnostics[c] = "ERROR"
            pass

    return diagnostics


def success_desc():
    return "Urbanczik-Senn test PASSED"


def failure_desc():
    return "Urbanczik-Senn test FAILED"

def getReportPath():
    # Provenance data path
    report_path = '/home/eev/Documents/spinnaker/multicomp/reports/'
    # get the latest created directory (alphabetically last)
    prov_data_path = report_path + sorted(os.listdir(report_path))[-1] + "/run_1/provenance_data/" 
    report_path = sorted(os.listdir(report_path))[-1]
    return report_path, prov_data_path

def getProvenanceData(target_chip, prov_data_path):
    prov_data_files = glob.glob(prov_data_path+str(target_chip[0])+'_'+str(target_chip[1])+"_*.xml")
    data = dict()
    for filename in prov_data_files:
        core = filename.split('/')[-1].split('_')[2]
        data[core] = dict()
        root = ElementTree.parse(filename).getroot()
        data[core]['pop'] = root.get('name')
        for item in root.findall('provenance_data_item'):
            data[core][item.get('name')] = item.text

    return data

def validate_params(p):
    ret = False

    if isinstance(p['target_chip'], str):
        p['target_chip'] = stringToTuple(p['target_chip'])

    if isinstance(p['multicomp_pops_per_source'], str):
        p['multicomp_pops_per_source'] = list(map(int, p['multicomp_pops_per_source'][1:-1].split(',')))

    if (p['directions'] == 'all' and p['rate_source_pops'] in [6, 12]) or \
        (p['directions'] in ['angle_n', 'angle_s', 'delta'] and p['rate_source_pops'] in [3, 6, 9]) or \
        (p['directions'] == 'poles' and p['rate_source_pops'] in [2, 4, 6]) or \
        (p['directions'] == 'south' and p['rate_source_pops'] in [2, 4, 6, 8]) or \
        (p['directions'] == 'any' and p['rate_source_pops'] <= 48) or \
        (p['directions'] == 'rad' and p['rate_source_pops'] <= 48):
        ret = True
    return ret

if __name__ == "__main__":
    # definition of all parameter values
    """    
    params = {
#        'target_chip'                   : [(7,7), (7,4), (1,0), (0,3), (4,4)],
        'target_chip'                   : [(3,0)],
        'multicomp_pops_per_source'     : [[8,8]],    # N_pop_syn
        'rate_source_pops'              : [2*14],       # total number of source chips
        'directions'                    : ['rad'],      
        'n_neurons_per_multicomp_pop'   : [8],          # N_neur_pop
        'sim_time'                      : [20],         
        'n_pop_target'                  : [14]          # N_pop_target
    }
    """    
    # get actual params from file
    params_list = pd.read_csv("./temp.csv").to_dict(orient="records")

    # database creation
    ts = int(time.time())

    filename = "./exp_params_" + str(ts) + ".csv"
    filename2 = "./exp_results_" + str(ts) + ".csv"
#    df_params = pd.DataFrame(columns=[*params.keys()])
    df_params = pd.DataFrame(columns=[*params_list[0].keys()])
    df_results = pd.DataFrame(columns=['exp_ID', 'chip'])
    for register in ROUTER_REGISTER_REGISTERS:
        df_results[register.name] = ''
    df_results['core'] = ''
    df_results['pop'] = ''
    
#    for vals in itertools.product(*params.values()):
#        itp = dict(zip(params, vals))
    for itp in params_list:
        if validate_params(itp):
            #print(itp)
            ts = int(time.time())
            print("---\n", ts, itp, "\n---", file=sys.stderr)
            itp.update(timestamp=ts)
            rtrdiag = test(**itp)
            # --- write results
            for key in itp:
                df_params.at[ts, key] = str(itp[key])

            (df_params['report_folder'], prov_data_path) = getReportPath()

            print(df_params)

            # --- now let's create the detailed experiment results
            # Create the row 


            for chip in rtrdiag:
                row = dict()
                row['exp_ID'] = ts
                row['chip'] = chip
                for register in ROUTER_REGISTER_REGISTERS:
                    row[register.name] = rtrdiag[chip].registers[register.value]
                # get the provenance data
                data = getProvenanceData(chip, prov_data_path)
                # iterate over the cores we just retrieved
                if data:
                    for core in data:
                        row['core'] = core
                        # create columns if they don't exist
                        for key in data[core]:
                            row[key] = data[core][key]
                        #print(row)
                        df_results = df_results.append(row, ignore_index=True, sort=False)
                else:
                    df_results = df_results.append(row, ignore_index=True, sort=False)
            df_params.to_csv(filename)
            df_results.to_csv(filename2)
    

    
    


