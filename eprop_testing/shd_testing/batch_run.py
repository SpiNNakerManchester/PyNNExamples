import os
import subprocess
import time

h_eta = [1.]
r_eta = [0.001, 0.0003]
var_1 = [0., 0.1, 1, 10]
var_2 = [0.1, 1, 10]
w_fb = [3]
recs = [0]

processes = []
logs = []
for h in h_eta:
    for r in r_eta:
        for v1 in var_1:
            for v2 in var_2:
                for fb in w_fb:
                    for rec in recs:
                        screen_name = "h{}_r{}_vm{}_vf{}_fb{}_rec{}".format(h, r, v1, v2, fb, rec)
                        open_screen = "screen -dmS " + screen_name + " bash -c "
                        move_and_source = "eprop_python3_source && "
                        # command = "\"" + move_and_source + " python3 incremental_shd.py {} {} {} {} {}\"".format(h, r, v1, v2, fb)
                        command = "\"python3 incremental_shd.py {} {} {} {} {} {}; exec bash\"".format(h, r, v1, v2, fb, rec)

                        logs.append(open("log_output_{}.txt".format(screen_name), 'a'))
                        # process = subprocess.Popen(open_screen+command, stdout=subprocess.PIPE)
                        processes.append(subprocess.Popen(
                            'screen -d -m -S {} bash -c {}'.format(screen_name, command),
                            shell=True,
                            stdout=logs[-1],
                            stdin=logs[-1],
                            stderr=logs[-1]))
                        print("Set up config", screen_name)

                        time.sleep(0.2)

days = 4
print("Done - beginning wait of", days, "days")
time.sleep(60*60*24*days)
print("Finished waiting")