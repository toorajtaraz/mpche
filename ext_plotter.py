from matplotlib import pyplot as plt

lines = open("/home/toorajtaraz/Documents/university/MP/projects/phase1/mpche/images/res.txt", "r").readlines()


configuration = []
for line in lines:
    conf = []
    splited = line.split(" ")
    for i in range(len(splited)):
        if splited[i] == '' or splited[i] == ' ':
            continue
        conf.append(splited[i])
    configuration.append(conf)

# print(configuration)
configuration_280_187 = []
configuration_840_560 = []
configuration_1400_934 = []
configuration_1960_1307 = []
configuration_2520_1680 = []

for conff in configuration:
    conf = []
    for i in range(6):
        conf.append(int(conff[i]))
    conf.append(float(conff[6]))
    if conf[4] == 280 and conf[5] == 187:
        configuration_280_187.append(conf)
    elif conf[4] == 840 and conf[5] == 560:
        configuration_840_560.append(conf)
    elif conf[4] == 1400 and conf[5] == 934:
        configuration_1400_934.append(conf)
    elif conf[4] == 1960 and conf[5] == 1307:
        configuration_1960_1307.append(conf)
    elif conf[4] == 2520 and conf[5] == 1680:
        configuration_2520_1680.append(conf)

for i in range(1, 3):
    fig, ax = plt.subplots()
    ax.set_ylabel("Time elapsed")
    ax.set_xlabel("Working threads")
    ax.plot([x[0] for x in configuration_280_187 if x[1] == i], [x[6] for x in configuration_280_187 if x[1] == i], marker='o', color="blue", label="280x187")
    ax.legend()
    ax.plot([x[0] for x in configuration_840_560 if x[1] == i], [x[6] for x in configuration_840_560 if x[1] == i], marker='x', color="red", label="840x560")
    ax.legend()
    ax.plot([x[0] for x in configuration_1400_934 if x[1] == i], [x[6] for x in configuration_1400_934 if x[1] == i], marker='s', color="orange", label="1400x934")
    ax.legend()
    ax.plot([x[0] for x in configuration_1960_1307 if x[1] == i], [x[6] for x in configuration_1960_1307 if x[1] == i], marker='8', color="violet", label="1960x1307")
    ax.legend()
    ax.plot([x[0] for x in configuration_2520_1680 if x[1] == i], [x[6] for x in configuration_2520_1680 if x[1] == i], marker='+', color="yellow", label="2520x1680")
    ax.legend()


    plt.savefig("/home/toorajtaraz/Documents/university/MP/projects/phase1/mpche/images/output{}.png".format(i))