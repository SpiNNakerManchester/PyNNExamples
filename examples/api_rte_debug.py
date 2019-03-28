from spinnman.transceiver import create_transceiver_from_hostname
from spinn_front_end_common.interface.interface_functions.host_execute_data_specification import HostExecuteDataSpecification
from spinn_front_end_common.utilities.utility_objs.executable_finder import ExecutableFinder
import spynnaker.pyNN.model_binaries as binary_path
import glob
from spinn_utilities.make_tools.replacer import Replacer
from spinn_machine import CoreSubsets, CoreSubset
from time import sleep
from spinnman.model.enums.cpu_state import CPUState
from spinnman.messages.scp.enums import Signal

test_directory = './rte_gen_binary_data/27_03_19/'

# from spinn_utilities.log import
lif_binary_file = test_directory+'IF_cond_exp.aplx'
izk_binary_file = test_directory+'IZK_cond_exp.aplx'

chip = (89,6)

replacer_lif = Replacer(lif_binary_file)
replacer_izk = Replacer(izk_binary_file)

tx_rx = create_transceiver_from_hostname('192.168.240.253',3)
tx_rx.ensure_board_is_ready()

machine = tx_rx.get_machine_details()
# executable_finder = ExecutableFinder(binary_search_paths=binary_path.__file__)
# data_files = sorted(glob.glob('./*.dat'))
data_files = glob.glob(test_directory+'*_{}_{}_*.dat'.format(chip[0],chip[1]))
# cores = []
# HostExecuteDataSpecification._execute(tx_rx, machine, 17, 0, 0, 7, '10.11.221.1_dataSpec_3_86_7.dat')
izk_cores = range(1,11)
lif_cores = range(11,12)

# cores = izk_cores + lif_cores
cores = [11,2,7,5,10,4,3,6,1,8,9]
lif_core_subsets = CoreSubsets([CoreSubset(0, 0, lif_cores)])
izk_core_subsets = CoreSubsets([CoreSubset(0, 0, izk_cores)])
core_subsets = CoreSubsets([CoreSubset(0, 0, cores)])

while 1:
    # for data_spec in data_files:
    #     core_index = int(str.split(data_spec,'_')[-1][:-4])
    for core_index in cores:
        data_spec = [d_spec for d_spec in data_files if int(str.split(d_spec,'_')[-1][:-4]) == core_index][0]
        results = HostExecuteDataSpecification._execute(tx_rx,machine,17,0,0,core_index,data_spec)#10.11.221.1_dataSpec_3_86_1.dat')
        print "start address:{}".format(hex(results['start_address']))

    # tx_rx.execute(0,0,lif_cores,lif_binary_file,17,is_filename=True,wait=True)
    # tx_rx.execute(0,0,izk_cores,izk_binary_file,17,is_filename=True,wait=True)
    tx_rx.execute_flood(lif_core_subsets,lif_binary_file,17,is_filename=True,wait=True)
    tx_rx.execute_flood(izk_core_subsets,izk_binary_file,17,is_filename=True,wait=True)
    # sleep(1)
    # tx_rx.send_signal(app_id=17,signal=Signal.CONTINUE)
    tx_rx.wait_for_cores_to_be_in_state(
        core_subsets, 17, [CPUState.READY])
    tx_rx.send_signal(17, Signal.START)

    sleep(0.1)

    rte_count = tx_rx.get_core_state_count(app_id=17,state=CPUState.RUN_TIME_EXCEPTION)
    if rte_count>0:
        io_buffers=list(tx_rx.get_iobuf(core_subsets))
        for i,io_buf in enumerate(io_buffers):
            print "=================core {}========================".format(cores[i])
            for line in io_buf.iobuf.split("\n"):
                if i in lif_cores:
                    print replacer_lif.replace(line)
                else:
                    print replacer_izk.replace(line)
        break

    tx_rx.stop_application(app_id=17)


