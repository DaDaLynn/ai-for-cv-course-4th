import os
import pandas as pd
from PIL import Image

root_path = r"D:\Lynn\AI-for-CV\Class_material\project\projectI\Data"
Phase = ["train", "val"]
Species = ["rabbits", "chickens"]
Classes = ["Mammals", "Birds"] #0 1


for p in Phase:
    Datas = {"path":[], "label":[]}
    for s in Species:
        data_path = root_path + "\\" + p + "\\" + s
        files = os.listdir(data_path)
        
        for item in files:
            item_path = os.path.join(data_path, item)
            try:                
                img = Image.open(item_path)
            except:
                break

            Datas["path"].append(item_path)
            Datas["label"].append(Species.index(s))

    pd_data = pd.DataFrame(Datas)
    pd_data.to_csv("Classes_%s_annotation.csv" % (p))

