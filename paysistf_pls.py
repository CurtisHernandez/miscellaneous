# -*- coding: utf-8 -*-
"""
Created on Thu May  9 16:51:38 2019

@author: ekh9
"""

# what are the conditions?
foundations = ['auth','carm','carp','fair','lbrt','lylt','pure','socn']

# where am I gonna find all this stuff
path = '/mnt/BIAC/munin3.dhe.duke.edu/DeBrigard/MCThink.01/Notes/MFT.01/'

# what are the scanIDs 
import os
scanIDs = os.listdir(path + 'Data/Func/')

# prompt for window size
# note that for some reason this line doesn't work in Spyder
# don't worry, it works on the cluster in a script
win_size = input('How many TRs does the event of interest take place over?...')

# fix the behavioral output file
from shutil import copyfile
import pandas as pd

def fixBehavioralFile(sourceExcelFile):
    # it has to be a textfile for some reason
    newName = sourceExcelFile.split(".")[0] + ".txt"
    copyfile(sourceExcelFile,newName)
    newBehavioralFile = pd.DataFrame.from_csv(newName,sep="\t")
    try:
        os.remove(newName)
    except:
        pass
    
    return newBehavioralFile



# make the timing file
def prepareTimingFile(subid,appendages,conditions,behavioralOutputFile,scanID):
    # subid = "MFT_01"
    # appendages = "_denoised_first_attempt" for example (will go on the session mat files)
    # conditions = a list of the foundations
    # behavioralOutputFile = their 
    # get TR onsets
    

    
    timing_file = open(path + 'Analysis/PLS_2019/' + subid + 'timing_file.txt',"a+")
    timing_file.write('%%%General section start%%%\n\n')    
    timing_file.write('prefix\t' + subid + '_' + appendages + '\t% prefix for session file and datamat file\n')
    timing_file.write('brain_region\t0.15 % threshold or file name for brain region \n') # this will only ever be threshold for this study
    timing_file.write('win_size\t' + win_size + '\t % number of scans in one hemodynamic period\n')
    timing_file.write('across_run\t1\t% 1 for merge data across all runs, 0 for within each run\n')
    timing_file.write('single_subj\t0\t% 1 for single subject analysis, 0 for normal analysis\n\n')
    timing_file.write('%%%General section end%%%\n')
    timing_file.write('%-----------------------\n')
    timing_file.write('%%%Condition section start%%%\n\n')
    for i, condition in enumerate(conditions):
        timing_file.write('cond_name\t' + condition + '\t% condition' + str(i+1) + ' name\n')
        timing_file.write('ref_scan_onset\t0\t% reference scan onset for condition' + str(i+1) + '\n')
        timing_file.write('num_ref_scan\t1\t% number of reference scan for condition' + str(i+1) + '\n\n')
    timing_file.write('%%%Condition section end%%%\n\n')
    timing_file.write('%-----------------------\n')
    timing_file.write('%%%Run section end%%%\n\n')
    for j in range(1,4):
        timing_file.write('data_files\t' + path + 'Data/Func/' + str(scanID) + '/f' + str(j) + '_reor_preprocessed.feat/filtered_func_reg*.nii % run' + str(j) + ' data pattern (must use wildcard)\n')
        for condition in conditions:
            timing_file.write('event_onsets\t')
        timing_file.write('')
        timing_file.write('')
        timing_file.write('')
        timing_file.write('')
        timing_file.write('')
        timing_file.write('')
