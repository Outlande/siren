import random

file_pre = "/home/yan/Desktop/datasets/ICL2_clean_consistent_normal/10.xyz"
file_now = "/home/yan/Desktop/datasets/ICL2_clean_consistent_normal/120.xyz"

file_res = "/home/yan/Desktop/datasets/10_120.xyz"

pre_xyz = []
with open(file_pre, "r") as f:
    lines = f.readlines()
    for line in lines:
        pre_xyz.append(line.split())

now_xyz = []
with open(file_now, "r") as f:
    lines = f.readlines()
    for line in lines:
        now_xyz.append(line.split())

now_xyz = now_xyz+random.sample(pre_xyz, len(pre_xyz)//20)

with open(file_res, "w") as f:
    for xyz in now_xyz:
        for value in xyz:
            f.write(value)
            f.write(" ")
        f.write("\n")