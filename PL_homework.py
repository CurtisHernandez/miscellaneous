# GOOD MORNING, ELEANOR.  RUN EACH CHUNK AGAIN AS YOU GO.  SPLIT UP THE THING WHERE IT GETS RID OF LOW-VARIANCE.  SOMEHOW THERE ARE STILL NANS IN YOUR DATA.

import os

# two datasets with the same parameters, one from the third quarter of 2018 and one from the fourth quarter of 2018
path_t1 = "/Users/eleanorhanna/Desktop/precisionLender_homework/FFIEC CDR Call Bulk All Schedules 09302018"
path_t2 = "/Users/eleanorhanna/Desktop/precisionLender_homework/FFIEC CDR Call Bulk All Schedules 12312018"

# get a list of all the files
filenames_t1 = os.listdir(path_t1)
filenames_t2 = os.listdir(path_t2)

# get rid of files that aren't formatted the same
filenames_t1 = [f for f in filenames_t1 if f.split(".")[-1]=="txt"]
filenames_t2 = [f for f in filenames_t2 if f.split(".")[-1]=="txt"]

filenames_t1.remove("Readme.txt")
filenames_t2.remove("Readme.txt")

filenames_t1.remove("FFIEC CDR Call Bulk POR 09302018.txt")
filenames_t2.remove("FFIEC CDR Call Bulk POR 12312018.txt")

filenames_t1.remove("FFIEC CDR Call Schedule NARR 09302018.txt")
filenames_t2.remove("FFIEC CDR Call Schedule NARR 12312018.txt")

# check that the lists are the same length 
print(len(filenames_t1))
print(len(filenames_t2))

import pandas as pd
import numpy as np

def joinFrames(filenames):
    for i, filename in enumerate(filenames):
        designator = filename.split(" ")[10] # so I know which file it was from originally - there are some redundant column names
        if i==0:
            alldata = pd.DataFrame.from_csv(filename, sep="\t", header=1)
            alldata.columns = [designator + "_" + i for i in alldata.columns]
        else:
            data = pd.DataFrame.from_csv(filename, sep="\t", header=1)
            data.columns = [designator + "_" + i for i in data.columns]
            if isinstance(data.index[0],str)==True: # the index is all strings
                data["idx"] = data.index
                for di in list(data.index):
                    try:
                        data.loc[di,"idx"] = np.int64(di)
                    except ValueError:
                        print(di)
                        data.drop(di,axis=0,inplace=True)
                data.set_index('idx',inplace=True)
            alldata = pd.concat([alldata,data],axis=1,join="inner")
    return alldata

t1_data = joinFrames([path_t1 + "/" + filename for filename in filenames_t1])
t2_data = joinFrames([path_t2 + "/" + filename for filename in filenames_t2])
print("")
print("Number of observations at T1:")
print(len(t1_data))
print("Number of observations at T2:")
print(len(t2_data))
print("Number of features at T1:")
print(len(t1_data.columns))
print("Number of features at T2:")
print(len(t2_data.columns))

t1_data.fillna(np.nan,inplace=True)
t2_data.fillna(np.nan,inplace=True)

# are any nans strings or anything?
print("T1: " + str(sum(["nan" in t1_data[tc].dropna() for tc in t1_data.columns])))
print("T2: " + str(sum(["nan" in t2_data[tc].dropna() for tc in t2_data.columns])))

t1_data.replace([np.inf, -np.inf],np.nan,inplace=True)
t2_data.replace([np.inf, -np.inf],np.nan,inplace=True)

t1_data = t1_data._get_numeric_data()
t2_data = t2_data._get_numeric_data()

print(len(t1_data.columns))
print(len(t2_data.columns))

# total assets in the t1 data
rck_t1 = pd.DataFrame.from_csv(path_t1 + "/" + [f for f in filenames_t1 if "rck" in f.lower()][0],sep="\t",header=1)
rck_t2 = pd.DataFrame.from_csv(path_t2 + "/" + [f for f in filenames_t2 if "rck" in f.lower()][0],sep="\t",header=1)
#print(rck_t1.columns)
#print(rck_t2.columns)
len(rck_t1["QTLY AVG OF TOTAL ASSETS"].dropna()) + len(rck_t1["QTLY AVG OF TOTAL ASSETS.1"].dropna()) == len(rck_t1.index)
len(rck_t2["QTLY AVG OF TOTAL ASSETS"].dropna()) + len(rck_t2["QTLY AVG OF TOTAL ASSETS.1"].dropna()) == len(rck_t2.index)
t1_data["assets_for_peer_group_assignment"] = t1_data["RCK_QTLY AVG OF TOTAL ASSETS"].apply(lambda x: x if np.isnan(x)==False else 0) + t1_data["RCK_QTLY AVG OF TOTAL ASSETS.1"].apply(lambda x: x if np.isnan(x)==False else 0)
t2_data["assets_for_peer_group_assignment"] = t2_data["RCK_QTLY AVG OF TOTAL ASSETS"].apply(lambda x: x if np.isnan(x)==False else 0) + t2_data["RCK_QTLY AVG OF TOTAL ASSETS.1"].apply(lambda x: x if np.isnan(x)==False else 0)
asset_features_to_drop = ["RCK_QTLY AVG OF TOTAL ASSETS","RCK_QTLY AVG OF TOTAL ASSETS.1","RCRI_QTLY AVG OF TOTAL ASSETS","RCRI_QTLY AVG OF TOTAL ASSETS.1",]
t1_data.drop(asset_features_to_drop,axis=1,inplace=True)
t2_data.drop(asset_features_to_drop,axis=1,inplace=True)

fiveBillion = 5 * 10**9
t1_data = t1_data[t1_data["assets_for_peer_group_assignment"]<fiveBillion]
t2_data = t2_data[t2_data["assets_for_peer_group_assignment"]<fiveBillion]

# drop features with little variance in the t1 data - WHY DOES THIS TAKE SO LONG
def variance(ary):
    return sum([(a - ary.mean())**2 for a in list(ary)])/(len(ary)-1)

print("Original number of features in T1 data: " + str(len(t1_data.columns)))
for i, tc in enumerate(t1_data.columns):
    if i % 10 == 0:
        print("Starting on " + str(i))
    try:
        if len(set(t1_data[tc]))<3 or variance(t1_data[tc]) < .8 * (1 - .8):
            t1_data.drop(tc,axis=1,inplace=True)
    except:
        pass
print("Number of features in T1 data after dropping low-variance features: " + str(len(t1_data.columns)))

# drop columns missing 25% or more of their data
for i, tc in enumerate(t1_data.columns):
    if i % 10 == 0:
        print("Starting on feature " + str(i+1))
    if len(t1_data[tc]) - len(t1_data[tc].dropna()) < .75 * len(t1_data[tc]):
        t1_data.drop(tc,axis=1,inplace=True)
    else:
        t1_data[tc].replace(np.nan,t1_data[tc].mean(),inplace=True)
print("Number of features dropped after losing features with too few values remaining: " + str(len(t1_data.columns)))

for tc in t1_data.columns:
    if tc not in t2_data.columns:
        t1_data.drop(tc,axis=1,inplace=True)
        
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# scale - THIS DOES NOT WORK FOR SOME REASON
t1_data_scaled = pd.DataFrame(data=StandardScaler().fit_transform(t1_data.values),columns=t1_data.columns,index=t1_data.index)



# Visualize the scatter of just 2 and see if that gives you an idea of k




# Kmeans; get the right k



# Figure out the features with the highest effect sizes from a Kruskal-wallis test on those clusters
def non_parametric_effect_size(h,n):
    # epsilon squared for kruskal-wallis test
    return h/((n**2-1)/(n+1))


# Train an SVM on the old thing and validate the clusters with the 50-50 test

# Classify

# Test the accuracy of the model on the testing set

# Make predictions for the newer dataframe

# Test that the features that make a difference are different



#
