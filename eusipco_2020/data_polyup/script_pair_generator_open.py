#TODO: reorganize similar to SDUMLA file structure
import os
import glob
import pandas as pd
import time
import random


blue_dir = "D:/data_polyup/ROI Database/Blue/"
green_dir = "D:/data_polyup/ROI Database/Green/"
red_dir = "D:/data_polyup/ROI Database/Red/"
nir_dir = "D:/data_polyup/ROI Database/NIR/"


file_blue = glob.glob(blue_dir+"/*")
file_green = glob.glob(green_dir+"/*")
file_red = glob.glob(red_dir+"/*")
file_nir = glob.glob(nir_dir+"/*")

time0 = time.time()
df_train = pd.DataFrame()
df_val = pd.DataFrame()
df_pair=pd.DataFrame()

pos_counter=0
for X1,Y1,Z1,T1 in (zip(file_blue[0:250],file_green[0:250],file_red[0:250],file_nir[0:250])):
    filx = glob.glob(X1 + "/1*")
    fily = glob.glob(Y1 + "/1*")
    filz = glob.glob(Z1 + "/1*")
    filt = glob.glob(T1 + "/1*")
    filx2 = glob.glob(X1 + "/2*")
    fily2 = glob.glob(Y1 + "/2*")
    filz2 = glob.glob(Z1 + "/2*")
    filt2 = glob.glob(T1 + "/2*")
    idx=1
    for fx1,fy1,fz1,ft1 in zip(filx,fily,filz,filt):
        idy=1
        for fx2, fy2, fz2, ft2 in zip(filx2, fily2, filz2, filt2):
            if True:
                df_pair = df_pair.append({'idx_0': fx1, 'idx_1': fy1,'idx_2': fz1,'idx_3': ft1,
                                          'idy_0': fx2, 'idy_1': fy2,'idy_2': fz2,'idy_3': ft2,
                                          'class': 1},
                       ignore_index=True)
                pos_counter +=1
            idy += 1
        idx +=1

filex = glob.glob(blue_dir+"/*/*")[0:3000]
filey = glob.glob(green_dir+"/*/*")[0:3000]
filez = glob.glob(red_dir+"/*/*")[0:3000]
filet = glob.glob(nir_dir+"/*/*")[0:3000]

files=list(zip(filex,filey,filez,filet))
random.shuffle(files)
neg_counter=0
while neg_counter!=pos_counter:
    in_x=random.randint(0, 2999)
    in_y = random.randint(0, 2999)
    print(in_x)
    print(in_y)
    face_labelx = os.path.basename(os.path.dirname(files[in_x][0]))
    ses_x=os.path.basename(files[in_x][0]).split("_")[0]
    face_labely = os.path.basename(os.path.dirname(files[in_y][0]))
    ses_y = os.path.basename(files[in_y][0]).split("_")[0]
    if face_labely != face_labelx and ses_x !=ses_y:
        df_pair = df_pair.append({'idx_0': files[in_x][0], 'idx_1': files[in_x][1], 'idx_2': files[in_x][2], 'idx_3': files[in_x][3],
                                  'idy_0': files[in_y][0], 'idy_1': files[in_y][1], 'idy_2': files[in_y][2], 'idy_3': files[in_y][3],
                                  'class': 0},
                                 ignore_index=True)
        neg_counter += 1
    else:
        random.shuffle(files)

df_pair = df_pair.sort_values(by=['class']).reset_index(drop=True)
df_pair.to_csv("poly_pairs.csv", index=False)



counter=0
for X,Y,Z,T in (zip(file_blue[250:500],file_green[250:500],file_red[250:500],file_nir[250:500])):
    filx = glob.glob(X + "/*")
    fily = glob.glob(Y + "/*")
    filz = glob.glob(Z + "/*")
    filt = glob.glob(T + "/*")
    for fx,fy,fz,ft in zip(filx,fily,filz,filt):
        idx = os.path.basename(fx).split('_')[1]
        if idx=='05' or idx=='06':
            df_val = df_val.append({'idx_0': fx, 'idx_1': fy,'idx_2': fz,'idx_3': ft, 'class': int(counter)},
                       ignore_index=True)
        else:
            df_train = df_train.append({'idx_0': fx, 'idx_1': fy, 'idx_2': fz, 'idx_3': ft, 'class': int(counter)},
                                       ignore_index=True)

    counter += 1

df_train = df_train.sort_values(by=['class']).reset_index(drop=True)
df_train.to_csv("poly_train.csv", index=False)
df_val = df_val.sort_values(by=['class']).reset_index(drop=True)
df_val.to_csv("poly_val.csv", index=False)













