import os
from PIL import Image
import pandas as pd
Data_root_path = r'D:\Lynn\AI-for-CV\Class_material\project\projectI\Data'
stage = ["train", "val"]
species = ["chickens", "rabbits", "rats"]

for st in stage:
    data_info = {'data':[], 'label':[]}
    for sp in species:
        data_path = os.path.join(Data_root_path, st, sp)
        files = os.listdir(data_path)
        for f in files:
            file_path = os.path.join(data_path, f) 
            try:
                img = Image.open(file_path)
            except :
                continue
            data_info['data'].append(file_path)
            data_info['label'].append(species.index(sp))
    df = pd.DataFrame(data_info)
    df.to_csv("Species_%s_annotation.csv" % st)

