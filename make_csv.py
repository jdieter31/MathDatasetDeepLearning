import os
from os import listdir
from os.path import isfile, join
from tqdm import tqdm

def txt_to_tsv(file_name):
    i = 0
    qs = []
    ans = []
    with open(file_name, "r") as f:
        line = f.readline()
        while line:
            if i % 2 == 0:
                qs.append(line)
            else:
                ans.append(line)
            line = f.readline()
            i += 1

    new_file = os.path.splitext(file_name)[0]+'.tsv'
    with open(new_file, "w+") as f:
        f.write("question\tanswer\n")
        for i in range(len(qs)):
            f.write(f"{qs[i][:-1]}\t{ans[i][:-1]}\n")

directories = [
    'mathematics_dataset-v1.0/extrapolate/',
    'mathematics_dataset-v1.0/interpolate/',
    'mathematics_dataset-v1.0/train-easy/',
    'mathematics_dataset-v1.0/train-hard/',
    'mathematics_dataset-v1.0/train-medium/'
]


for directory in directories:
    files = [f for f in listdir(directory) if isfile(join(directory, f))]
    for file_name in tqdm(files):
        file_loc = directory + file_name
        if os.path.splitext(file_loc)[1] == ".txt":
            txt_to_tsv(file_loc)
            

    
