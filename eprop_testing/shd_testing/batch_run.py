import os
import subprocess
import time

h_eta = [0.03]
r_eta = [0.0004, 0.0002, 0.0001]
var_v = [0.]
var_f = [0.01]
w_fb = [3]
fb_m = [100., 300., 500., 700., 900.]
recs = [0]
batchs = [5, 20]

processes = []
logs = []
for h in h_eta:
    for r in r_eta:
        for v in var_v:
            for f in var_f:
                for fb in w_fb:
                    for rec in recs:
                        for m in fb_m:
                            for b in batchs:
                                screen_name = "h{}_r{}_b{}_vm{}_vf{}_fb{}x{}_rec{}".format(h, r, b, v, f, fb, m, rec)
                                open_screen = "screen -dmS " + screen_name + " bash -c "
                                move_and_source = "eprop_python3_source && "
                                # command = "\"" + move_and_source + " python3 incremental_shd.py {} {} {} {} {}\"".format(h, r, v1, v2, fb)
                                command = "\"python3 incremental_shd.py {} {} {} {} {} {} {} {}; exec bash\"".format(h, r, v, f, fb, rec, m, b)

                                # logs.append(open("log_output_{}.txt".format(screen_name), 'a'))
                                # process = subprocess.Popen(open_screen+command, stdout=subprocess.PIPE)
                                processes.append(subprocess.Popen(
                                    'screen -d -m -S {} bash -c {}'.format(screen_name, command),
                                    shell=True,
                                    stdout=subprocess.PIPE,
                                    stdin=subprocess.PIPE,
                                    stderr=subprocess.PIPE))
                                print("Set up config", screen_name)

                                time.sleep(30)

days = 4
print("Done - beginning wait of", days, "days")
time.sleep(60*60*24*days)
print("Finished waiting")