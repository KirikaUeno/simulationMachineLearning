import io
import math
import sys

import numpy as np
import os
import os.path
from numpy import savetxt


def get_inputs_main(main_input):
    values_to_find_input = []
    f = io.open("inputParameters.txt", 'r', encoding='utf-8', newline='\n', errors='ignore')
    for line in f:
        values_to_find_input.append(line[:-1])
    f.close()
    answer = {}
    finI = io.open(main_input, 'r', encoding='utf-8', newline='\n', errors='ignore')
    for i, line in enumerate(finI):
        tokens = line.strip().split('=')
        for value in values_to_find_input:
            if tokens[0] == value:
                if tokens[0] == "MAXB":
                    answer[tokens[0]] = (-float(tokens[1]) - 0.0000372) / 0.000588
                else:
                    answer[tokens[0]] = float(tokens[1])
    finI.close()
    return answer


def get_outputs(output_file):
    arr_init = np.loadtxt(output_file)
    lines_to_delete = []
    for i, line in enumerate(arr_init):
        if line[9] < 0:
            lines_to_delete.append(i)
    arr = np.delete(arr_init, lines_to_delete, 0)
    ref_is_exist = arr_init[0, 2] > 5
    answer = {}
    if ref_is_exist:
        arr_ref = arr_init[0, :]
        answer["X_pos"] = float(arr_ref[0]) + np.mean(arr[:, 0])
        answer["Y_pos"] = float(arr_ref[1]) + np.mean(arr[:, 1])
        answer["Z_pos"] = float(arr_ref[2]) + np.mean(arr[:, 2])
        answer["average_kin_energy"] = float(arr_ref[5]) - 511000 + np.mean(arr[:, 5])
        answer["alpha_X"] = np.sqrt(np.mean(np.square(arr[:, 3]))) / (float(arr_ref[5]) - 511000)
        answer["alpha_Y"] = np.sqrt(np.mean(np.square(arr[:, 4]))) / (float(arr_ref[5]) - 511000)
    answer["Charge"] = np.sum(arr[:, 7])
    answer["sigZ"] = np.sqrt(np.mean(np.square(arr[:, 2])))
    answer["sigX"] = np.sqrt(np.mean(np.square(arr[:, 0])))
    answer["sigY"] = np.sqrt(np.mean(np.square(arr[:, 1])))
    answer["energy_spread"] = np.sqrt(np.mean(np.square(arr[:, 5])))
    answer["emittance_Y"] = np.sqrt(
        np.mean(np.square(arr[:, 1])) * np.mean(np.square(arr[:, 4])) - math.pow(np.mean((arr[:, 1] * arr[:, 4])), 2))
    answer["emittance_X"] = np.sqrt(
        np.mean(np.square(arr[:, 0])) * np.mean(np.square(arr[:, 3])) - math.pow(np.mean((arr[:, 0] * arr[:, 3])), 2))
    return answer


def load_dataset_csv(mid_file_name):
    x_dict = get_inputs_main("run" + str(0) + ".in")
    y_dict = get_outputs("run" + str(0) + "." + mid_file_name + ".001")

    arrX = np.array([list(x_dict.values())])
    arrY = np.array([list(y_dict.values())])
    i = 0
    import os.path
    while os.path.exists("run" + str(i + 1) + ".in"):
        x_dict = get_inputs_main("run" + str(i + 1) + ".in")
        y_dict = get_outputs("run" + str(i + 1) + "." + mid_file_name + ".001")

        arrX = np.concatenate((arrX, np.array([list(x_dict.values())])))
        arrY = np.concatenate((arrY, np.array([list(y_dict.values())])))
        i += 1
    savetxt("informationX.csv", arrX, delimiter=",")
    savetxt("informationY.csv", arrY, delimiter=",")


def load_dataset_txt(mid_file_name):
    info = ""
    f = open("information.txt", "w")
    i = -1
    print("start python")
    while os.path.exists("run" + str(i + 1) + "." + mid_file_name + ".001"):
        print(i)
        i += 1
        x_dict = get_inputs_main("run" + str(i) + ".in")
        y_dict = get_outputs("run" + str(i) + "." + mid_file_name + ".001")

        info += "input:" + "\n"
        for j in range(len(x_dict)):
            info += (str(list(x_dict.keys())[j]) + " " + str(list(x_dict.values())[j]) + "\n")
        info += "output:" + "\n"
        for j in range(len(y_dict)):
            info += (str(list(y_dict.keys())[j]) + " " + str(list(y_dict.values())[j]) + "\n")
        try:
            os.system("rm " + "run" + str(i) + ".LandF.001")
            os.system("rm " + "run" + str(i) + ".Xemit.001")
            os.system("rm " + "run" + str(i) + ".Yemit.001")
            os.system("rm " + "run" + str(i) + ".Zemit.001")
            os.system("rm " + "run" + str(i) + ".log")
            os.system("rm " + "run" + str(i) + ".Log.001")
            os.system("rm " + "run" + str(i) + ".ref.001")
            os.system("rm " + "o_out" + str(i) + ".txt")
            os.system("rm " + "e_out" + str(i) + ".txt")
        except Exception:
            pass
    f.write(info)
    f.close()
    print("end python")


def delete_files(ini, mid_file_name):
    print("start python")
    for j in range(10):
        print(j + ini)
        i = j + ini
        try:
            os.system("rm " + "run" + str(i) + ".in")
            os.system("rm " + "run" + str(i) + ".LandF.001")
            os.system("rm " + "run" + str(i) + ".Xemit.001")
            os.system("rm " + "run" + str(i) + ".Yemit.001")
            os.system("rm " + "run" + str(i) + ".Zemit.001")
            os.system("rm " + "run" + str(i) + ".log")
            os.system("rm " + "run" + str(i) + ".Log.001")
            os.system("rm " + "run" + str(i) + ".ref.001")
            os.system("rm " + "o_out" + str(i) + ".txt")
            os.system("rm " + "e_out" + str(i) + ".txt")
            os.system("rm " + "run" + str(i) + "." + mid_file_name + ".001")
        except Exception:
            pass
    print("end python")


# delete_files(int(sys.argv[1]),int(sys.argv[2]))
load_dataset_txt(int(sys.argv[1]))
