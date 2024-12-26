import pandas as pd
import numpy as np
import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))

dataset_list = ["Linear", "Linear_large", "Linear_real", "RadialIn"]
for dataset in dataset_list:
    for k in range(1, 11):
        for l in range(1, 21):
            folder_name = f"{dataset}_{k}_{l}"
            iloc_df = pd.read_csv(f"./{dataset}/{folder_name}/input_iloc.csv")
            jloc_df = pd.read_csv(f"./{dataset}/{folder_name}/input_jloc.csv")

            i_list = iloc_df["i"].unique().tolist()
            j_list = jloc_df["j"].unique().tolist()
            iloc_param = np.zeros((len(i_list), 2))
            jloc_param = np.zeros((len(j_list), 2))
            j_nearest_param = np.ones(len(j_list)) * 10000
            for i in i_list:    
                iloc_param[i - 1, 0] = iloc_df.query("i == @i & coord == 1").reset_index()["value"][0]
                iloc_param[i - 1, 1] = iloc_df.query("i == @i & coord == 2").reset_index()["value"][0]
            for j in j_list:    
                jloc_param[j - 1, 0] = jloc_df.query("j == @j & coord == 1").reset_index()["value"][0]
                jloc_param[j - 1, 1] = jloc_df.query("j == @j & coord == 2").reset_index()["value"][0]
                for i in i_list:
                    j_nearest_param[j - 1] = min(j_nearest_param[j - 1], np.linalg.norm(iloc_param[i - 1, :] - jloc_param[j - 1, :]))

            nearest_df = pd.DataFrame({"j": [j + 1 for j in range(len(j_list))], "value": [j_nearest_param[j] for j in range(len(j_list))]})
            nearest_df.to_csv(f"./{dataset}/{folder_name}/input_jnearest.csv", index = None)