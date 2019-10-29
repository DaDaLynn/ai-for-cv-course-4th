import os
import pandas as pd
from PIL import Image

CLASSES = {'chickens':0, 'rabbits':1, 'rats':1}
SPECIES = {'chickens':0, 'rabbits':1, 'rats':2}
STAGES = {'train', 'val'}

root_path = r'D:\Lynn\AI-for-CV\Class_material\project\projectI\Data'
for s in STAGES:
    data_info = {"image":[], "class":[], "species":[]}
    for c in CLASSES.keys():
        files_list = os.listdir(os.path.join(root_path, s, c))
        for f in files_list:
            file_path = os.path.join(root_path, s, c, f)
            try:
                Img = Image.open(file_path)
            except:
                continue
            data_info["image"].append(file_path)
            data_info["class"].append(CLASSES[c])
            data_info["species"].append(SPECIES[c])
    data_pd = pd.DataFrame(data_info)
    data_pd.to_csv("Classes_Species_%s_annotation.csv"%(s))

