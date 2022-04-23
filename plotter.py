from matplotlib import pyplot as plt
import subprocess
import time
from PIL import Image
import tabulate
#measure execution time of a command
path_to_binary = "/home/toorajtaraz/Documents/university/MP/projects/phase1/mpche/build/bin/mpche"
now = time.time()
i_path = "/home/toorajtaraz/Documents/university/MP/projects/phase1/mpche/images/he4.jpg"
o_path = "/home/toorajtaraz/Documents/university/MP/projects/phase1/mpche/images/he4_output.jpg"
t_num = 12
r_num = 0.2
mode = 2
s_num = 0
window = 51
configuration = []
for mode in range(1, 3):
    for r in range(1, 11, 2):
        r_num = r / 10
        for t_num in range(1, 13):
            for window in range(51, 52, 50):
                command = [
                    path_to_binary, 
                    "-i", 
                    i_path, 
                    "-o", 
                    o_path,
                    "-c",
                    "-t",
                    str(t_num),
                    "-r",
                    str(r_num),
                    "-m",
                    str(mode),
                    "-s",
                    str(s_num),
                    "-w",
                    str(window)
                ]
                now = time.time()
                ret = subprocess.call(command)
                print("ret = ", ret)
                print(command)
                time_lapsed = time.time() - now
                print("Time taken: ", time_lapsed)
                if ret == 0:
                    out_put_width = 0
                    out_put_height = 0
                    with Image.open(o_path) as img:
                        out_put_width, out_put_height = img.size
                    configuration.append([t_num, mode, s_num, window, out_put_width, out_put_height, time_lapsed])
                now = time.time()



#Plot based on configuration
mode_1_configuration = []
mode_2_configuration = []
mode_3_configuration = []
mode_4_configuration = []
for i in configuration:
    if i[1] == 1:
        mode_1_configuration.append(i)
    elif i[1] == 2:
        mode_2_configuration.append(i)
    elif i[1] == 3:
        mode_3_configuration.append(i)
    elif i[1] == 4:
        mode_4_configuration.append(i)

t = tabulate.tabulate(mode_1_configuration, headers=["t_num", "mode", "s_num", "window", "out_put_width", "out_put_height", "time_lapsed"])
print(t)
t = tabulate.tabulate(mode_2_configuration, headers=["t_num", "mode", "s_num", "window", "out_put_width", "out_put_height", "time_lapsed"])
print(t)
t = tabulate.tabulate(mode_3_configuration, headers=["t_num", "mode", "s_num", "window", "out_put_width", "out_put_height", "time_lapsed"])
print(t)
t = tabulate.tabulate(mode_4_configuration, headers=["t_num", "mode", "s_num", "window", "out_put_width", "out_put_height", "time_lapsed"])
print(t)

fig, ax = plt.subplots()
ax.set_xlabel("T_num")
ax.set_ylabel("Time taken")
ax.set_title("Time taken vs T_num for mode 1")
ax.plot([x[0] for x in mode_1_configuration], [x[6] for x in mode_1_configuration])
plt.savefig("/home/toorajtaraz/Documents/university/MP/projects/phase1/mpche/images/output1.png")

fig, ax = plt.subplots()
ax.set_xlabel("T_num")
ax.set_ylabel("Time taken")
ax.set_title("Time taken vs T_num for mode 2")
ax.plot([x[0] for x in mode_2_configuration], [x[6] for x in mode_2_configuration])
plt.savefig("/home/toorajtaraz/Documents/university/MP/projects/phase1/mpche/images/output2.png")
fig, ax = plt.subplots()
