from spinnman.transceiver import create_transceiver_from_hostname
from spinn_front_end_common.interface.interface_functions.host_execute_data_specification import HostExecuteDataSpecification
from spinn_front_end_common.utilities.utility_objs.executable_finder import ExecutableFinder
import spynnaker.pyNN.model_binaries as binary_path
import glob
from spinn_utilities.make_tools.replacer import Replacer
from spinn_machine import CoreSubsets, CoreSubset
from time import sleep
from spinnman.model.enums.cpu_state import CPUState

# from spinn_utilities.log import
lif_binary_file = './IF_cond_exp.aplx'
izk_binary_file = './IZK_cond_exp.aplx'

replacer_lif = Replacer(lif_binary_file)
replacer_izk = Replacer(izk_binary_file)

tx_rx = create_transceiver_from_hostname('192.168.240.253',3)
tx_rx.ensure_board_is_ready()

machine = tx_rx.get_machine_details()
# executable_finder = ExecutableFinder(binary_search_paths=binary_path.__file__)

data_files = sorted(glob.glob('./*.dat'))
cores = []
# HostExecuteDataSpecification._execute(tx_rx, machine, 17, 0, 0, 7, '10.11.221.1_dataSpec_3_86_7.dat')
lif_cores = range(1,6)
izk_cores = range(6,10)
while 1:
    for i,data_spec in enumerate(data_files):
        cores.append(i+1)
        results = HostExecuteDataSpecification._execute(tx_rx,machine,17,0,0,i+1,data_spec)#10.11.221.1_dataSpec_3_86_1.dat')
        print "start address:{}".format(hex(results['start_address']))
        # HostExecuteDataSpecification._execute(tx_rx,machine,17,0,0,7,'10.11.221.1_dataSpec_3_86_7.dat')

    tx_rx.execute(0,0,lif_cores,lif_binary_file,17,is_filename=True)
    tx_rx.execute(0,0,izk_cores,izk_binary_file,17,is_filename=True)
    # for core in cores:
    #     if core >5:
    #         binary_file = izk_binary_file
    #     else:
    #         binary_file = lif_binary_file
    #         tx_rx.execute(0,0,[core],binary_file,17,is_filename=True)

    sleep(1)

    rte_count = tx_rx.get_core_state_count(app_id=17,state=CPUState.RUN_TIME_EXCEPTION)
    if rte_count>0:
        core_subsets = CoreSubsets([CoreSubset(0, 0, cores)])
        io_buffers=list(tx_rx.get_iobuf(core_subsets))
        for i,io_buf in enumerate(io_buffers):
            print "=================core {}========================".format(cores[i])
            for line in io_buf.iobuf.split("\n"):
                if i<5:
                    print replacer_lif.replace(line)
                else:
                    print replacer_izk.replace(line)
        break

    tx_rx.stop_application(app_id=17)


