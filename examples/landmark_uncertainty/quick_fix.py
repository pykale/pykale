import os


from os import listdir
from os.path import isfile, join

mypath = "/mnt/tale_shared/schobs/data/tabular/cardiac_landmark_uncertainty/Uncertainty_tuples/U-NET/ISBI/12std"
onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]


for name in onlyfiles:
    if "play" not in name and "calibration" not in name and "calib" not in name: 
        split_name= name.split("_")
        #          checkpointuncertainty_pairs_
        first =  split_name[0]
        second = "_" + split_name[1]
        if  "val" in split_name[5]:
            third = "_valid"
        else:
            third = "_test"
        fourth = "_" + split_name[3].split("E")[0]

        new_name = first + second + third + fourth + '.csv'
        
        print("old name %s, new name %s" % (name, new_name))

        os.rename(os.path.join(mypath, name), os.path.join(mypath, new_name))