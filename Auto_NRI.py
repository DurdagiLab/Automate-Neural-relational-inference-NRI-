# -*- coding: utf-8 -*-
"""
Created on Wed Jan 10 14:10:17 2024

@author: Ehsan.Sayyah
"""
import os
import glob
import shutil
import pandas as pd
import subprocess
import sys
import runpy
import re
import numpy as np

nri_path= 'C:/Users/ehsan/Desktop/NRI_HITMER/NRI_pdb/wt/'
# import convert_dataset as cd

path= 'C:/Users/ehsan/Desktop/NRI_HITMER/NRI_pdb/'
os.chdir(path)
files= glob.glob('*-1.pdb')

frame_no = int(input('Write down the number of frames you have : \n'))
step= int(input('How many step size you need to run: \n'))
exp= int(frame_no/step)
hidden_encode= int(input('How many encoder you need: \n'))
hidden_decode= int(input('How many decoder you need: \n'))
def directory(file):
    os.chdir(path)
    os.makedirs(file)
    shutil.copy(name+'.pdb', file+'/'+name+'.pdb')
    shutil.copytree(nri_path, file+'/NRI')
    shutil.copy(file+'/'+name+'.pdb', file+'/NRI/data/pdb/{}.pdb'.format(name))
    shutil.copy(file+'/'+name+'.pdb', file+'/NRI/{}.pdb'.format(name))
    return print('Directory is Ready for Start The NRI')

def res_num(file):
    with open(n) as pdb:
        num=pdb.readlines()
    model2= num.index('MODEL     2\n')
    residue_number = model2 - 2
    for index, string in enumerate(num):
        if 'UNK' in string:
            protein_num = index-1
            break
    return residue_number, protein_num

def change_directory(file):
    os.chdir(path+'/'+file+'/NRI/')
    return print('Entering To {}'.format(file))

def convert_dataset_filename(file):
    residue_number, protein_num = res_num(file)
    with open('convert_dataset.py', 'r') as filename:
        line=filename.readlines()
        # line[341] = 'pdb_file= "{}"\n'.format(str(file))
        for a in range(len(line)):
            if "pdb_file= ''" in line[a]:
                line[a] = line[a].replace("pdb_file= ''", 'pdb_file= "{}"\n'.format(str(file)))
            if 'my_pdb_res_num' in line[a]:
                line[a] = line[a].replace('my_pdb_res_num', str(residue_number))
            else:
                pass
    os.remove('convert_dataset.py')
    lines= ''.join(line)
    with open('convert_dataset.py', 'w') as convertfile:
        convertfile.write(lines)
    with open('main.py', 'r') as filename:
        line=filename.readlines()
        for a in range(len(line)):
            if 'my_pdb_res_num' in line[a]:
                line[a] = line[a].replace('my_pdb_res_num', str(residue_number))
            if 'time_step' in line[a]:
                line[a] = line[a].replace('time_step', str(step))
            if 'no_exp' in line[a]:
                line[a] = line[a].replace('no_exp', str(exp))
            if 'hidencod' in line[a]:
                line[a] = line[a].replace('hidencod', str(hidden_encode))
            if 'hiddecod' in line[a]:
                line[a] = line[a].replace('hiddecod', str(hidden_decode))
            else:
                pass
    os.remove('main.py')
    lines= ''.join(line)
    with open('main.py', 'w') as convertfile:
        convertfile.write(lines)
    
    with open('postanalysis_visual.py', 'r') as filename:
        line=filename.readlines()
        for a in range(len(line)):
            if 'my_pdb_res_num' in line[a]:
                line[a] = line[a].replace('my_pdb_res_num', str(residue_number))
            if 'my_prot_res_num' in line[a]:
                line[a] = line[a].replace('my_prot_res_num', str(protein_num))
            # if 'probs_train' in line[a]:
            #     line[a] = line[a].replace('probs_train', 'probs_train_{}'.format(n))
            else:
                pass
    os.remove('postanalysis_visual.py')
    lines= ''.join(line)
    with open('postanalysis_visual.py', 'w') as convertfile:
        convertfile.write(lines)
    with open('Test_Trajectory.py', 'r') as filename:
        line=filename.readlines()
        for a in range(len(line)):
            if 'my_pdb_res_num' in line[a]:
                line[a] = line[a].replace('my_pdb_res_num', str(residue_number))
            if 'my_prot_res_num' in line[a]:
                line[a] = line[a].replace('my_prot_res_num', str(protein_num))
            if 'time_step' in line[a]:
                line[a] = line[a].replace('time_step', str(step))
            if 'no_exp' in line[a]:
                line[a] = line[a].replace('no_exp', str(exp))
            if 'hidencod' in line[a]:
                line[a] = line[a].replace('hidencod', str(hidden_encode))
            if 'hiddecod' in line[a]:
                line[a] = line[a].replace('hiddecod', str(hidden_decode))
            if 'Frame_Number' in line[a]:
                line[a] = line[a].replace('Frame_Number', str(frame_no))
            else:
                pass
    os.remove('Test_Trajectory.py')
    lines= ''.join(line)
    with open('Test_Trajectory.py', 'w') as convertfile:
        convertfile.write(lines)
    with open('Test_Trajectory_valid.py', 'r') as filename:
        line=filename.readlines()
        for a in range(len(line)):
            if 'my_pdb_res_num' in line[a]:
                line[a] = line[a].replace('my_pdb_res_num', str(residue_number))
            if 'my_prot_res_num' in line[a]:
                line[a] = line[a].replace('my_prot_res_num', str(protein_num))
            if 'time_step' in line[a]:
                line[a] = line[a].replace('time_step', str(step))
            if 'no_exp' in line[a]:
                line[a] = line[a].replace('no_exp', str(exp))
            if 'hidencod' in line[a]:
                line[a] = line[a].replace('hidencod', str(hidden_encode))
            if 'hiddecod' in line[a]:
                line[a] = line[a].replace('hiddecod', str(hidden_decode))
            if 'Frame_Number' in line[a]:
                line[a] = line[a].replace('Frame_Number', str(frame_no))
            else:
                pass
    os.remove('Test_Trajectory_valid.py')
    lines= ''.join(line)
    with open('Test_Trajectory_valid.py', 'w') as convertfile:
        convertfile.write(lines)
    return

def run(file):
    # subprocess.run(["python", "-c","python convert_dataset.py"])
    runpy.run_path('convert_dataset.py')
    print("convert_dataset.py completed")
    runpy.run_path('main.py')
    result=subprocess.run(['python', 'postanalysis_visual.py'],capture_output=True, text=True)
    #energy_score= postanalysis_visual.Sum
    energy_score= result.stdout.split('\n')
    runpy.run_path('Test_Trajectory.py')
    runpy.run_path('Test_Trajectory_valid.py')
    return (energy_score)
score=[]
def energy_score_save(energy_score,file):
    score.append(file)
    for i in range(len(energy_score)):
        score.append(f'sum{i}')
        score.append(energy_score[i])
    return score

def nll_plot(file):
    os.environ['KMP_DUPLICATE_LIB_OK']='True'
    import matplotlib.pyplot as plt  
    os.chdir(path+'/'+file+'/NRI/logs/')
    log = pd.read_csv(file+'_log_train.txt',header=None, sep=' ')
    epochs = log.iloc[::10, 1][:]
    nll_train = log.iloc[::10, 3][:]
    nll_val = log.iloc[::10, 11][:]
    fig = plt.figure(figsize=(16, 10))
    ax = fig.add_subplot(111)
    plt.plot(epochs, nll_train, linestyle='-', color='blue', label='nll_train')
    plt.plot(epochs, nll_val, linestyle='-', color='red', label='nll_val')
    ax.set_xlabel('Epoch', fontsize=14, fontweight='bold', labelpad=12)  # Added labelpad for spacing
    ax.set_ylabel('nll', fontsize=14, fontweight='bold', labelpad=12)  # Added labelpad for spacing
    ax.set_title('Training and Validation Loss Over Epochs', fontsize=16, fontweight='bold')  # Increased font size for the title

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    ax.legend(fontsize=12, frameon=False)

    ax.tick_params(axis='both', which='major', labelsize=12, width=1, length=5, direction='out', 
                   pad=8, colors='black', labelcolor='black', bottom=True, left=True)
    plt.savefig('nll_train_val.png', dpi=500)
    plt.show()


for n in files:
    file=n.split('-')[0]
    name= n.split('.')[0]
    directory(file)
    res_num(file)
    change_directory(file)
    convert_dataset_filename(file)
    energy_score= run(file)
    data= energy_score_save(energy_score,file)
    energy= pd.DataFrame({'compound': score[::11], 'sum':score[1::11], 'Energy':score[2::11], 'sum2':score[3::11], 'Energy2':score[4::11],'sum3':score[5::11], 'Energy3':score[6::11],'sum4':score[7::11], 'Energy4':score[8::11]})
    energy.to_csv(path+'Energy_score_for_compounds.csv', index=False)
    nll_plot(file)
