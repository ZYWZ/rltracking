import os
import csv

basepath = "C:\\Users\\liuxi\\PycharmProjects\\gym-rltracking\\gym_rltracking\\envs\\rltrack\\seq_result"
filenames = []
names = []
for root, dirs, files in os.walk(basepath):
    for name in files:
        name = os.path.join(basepath, name)
        filenames.append(name)
        name = name.split("\\")
        names.append(name[-1])

for datafile, name in zip(filenames, names):
    print("Processing ", datafile)
    with open (datafile) as f:
        lines = f.readlines()
        line_results = [line.rstrip() for line in lines]

    processed_lines = []
    for line in line_results:
        line_list = line.split(",")
        processed_lines.append([line_list[0], line_list[1], line_list[2], line_list[3], line_list[4], line_list[5], -1, -1, -1 ,-1])

    output_file = os.path.join(basepath, "processed", name)
    with open(output_file, 'w') as f:
        for _list in processed_lines:
            for i, _string in enumerate(_list):
                if i != len(_list)-1:
                    f.write(str(_string) + ',')
                else:
                    f.write(str(_string))
            f.write('\n')
