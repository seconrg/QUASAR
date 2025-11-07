import csv
import numpy as np
import matplotlib.pyplot as plt

FOLDER = "/media/csl-wanhanglu/SSD/QUASAR/ProxyTimeResult/"
# SUBFOLDERS = ["quadstream_robot_full/", "quadstream_robot_multiview/"]
SUBFOLDERS = ["quasar_robot_full/", "quasar_robot_hiddenlayer/", "quasar_robot_onlyfront/"]



def drawProxyNumberAndTime(folder, subfolders):
    
    proxy_times = []
    proxy_sizes = []

    for subfolder in subfolders:
        with open(FOLDER + subfolder + "proxytime.csv", 'r') as time_file, \
             open(FOLDER + subfolder + "proxysize.csv", 'r') as size_file:
            time_reader = csv.reader(time_file)
            size_reader = csv.reader(size_file)
            for time_row, size_row in zip(time_reader, size_reader):
                proxy_times.extend([float(x) for x in time_row])
                proxy_sizes.extend([int(x) for x in size_row])
    # Make the dots smaller for better visualization
    plt.scatter(proxy_sizes, proxy_times, marker='o', linestyle='None', s=5)
    plt.xlabel("Proxy Size (bytes)")
    plt.ylabel("Proxy Time (ms)")
    plt.title("Proxy Size vs Time")
    plt.show()

if __name__ == "__main__":
    drawProxyNumberAndTime(FOLDER, SUBFOLDERS)