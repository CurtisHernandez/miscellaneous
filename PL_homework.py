# GOOD MORNING, ELEANOR.  RUN EACH CHUNK AGAIN AS YOU GO.  SPLIT UP THE THING WHERE IT GETS RID OF LOW-VARIANCE.  SOMEHOW THERE ARE STILL NANS IN YOUR DATA.

import os

# two datasets with the same parameters, one from the third quarter of 2018 and one from the fourth quarter of 2018
path_t1 = "C:/Users/ekh9/Desktop/PL_interview/FFIEC CDR Call Bulk All Schedules 09302018"
path_t2 = "C:/Users/ekh9/Desktop/PL_interview/FFIEC CDR Call Bulk All Schedules 12312018"

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

def preliminaryDataWrangling(df):
    # while it would be more efficient overall to do this once all the dfs
    # are joined, I'm breaking it down for now so I can tell what's going on
    # with these redundant columns
    # if it's not numeric we don't care for now
    df = df._get_numeric_data()
    # fill up empty spots with nans
    df.fillna(np.nan,inplace=True)
    # get rid of infinities
    df.replace([np.inf, -np.inf],np.nan,inplace=True)
    # drop empty columns
    df.dropna(axis=1,how="all",inplace=True)
    return df

def consolidate(origdf,lstOfArys,newname):
    # currently only works for numerical values    
    tempdf = origdf[lstOfArys]
    counts = len(lstOfArys) - origdf[lstOfArys].isnull().sum(axis=1)
    if max(set(counts))==1:
#        print("There is at most one value for each of these rows, so they will be consolidated and appended to the dataframe")        
        #origdf[newname] = origdf[lstOfArys].replace(np.nan,0).sum(axis=1)        
        #tempdf[newname] = tempdf[lstOfArys].replace(np.nan,0).sum(axis=1)
        #origdf.drop(lstOfArys,axis=1,inplace=True)
        #origdf = pd.concat([origdf,tempdf],axis=1,join="outer")        
        for i in range(1,len(lstOfArys)):
            tempdf[lstOfArys[0]] = tempdf[lstOfArys[0]].combine_first(tempdf[lstOfArys[i]])
        origdf.drop(lstOfArys,axis=1,inplace=True)
        origdf[newname] = tempdf[lstOfArys[0]]
#    else:
#        print("There are rows where two or more of these columns are populated, so they will not be consolidated")
    

def groptai(strng):
    # gets rid of periods that aren't important
    # this is important for consolidating columns
    if strng.split(".")[-1].isdigit() == True:
        newstr = strng.replace(".","")[:-len(strng.split(".")[-1])] + "." + str(strng.split(".")[-1])
    else:
        newstr = strng.replace(".","")
    return newstr
       

def joinFrames(filenames):
    for i, filename in enumerate(filenames):
        designator = filename.split(" ")[10] # so I know which file it was from originally - there are some redundant column names
        if i==0:
            alldata = pd.DataFrame.from_csv(filename, sep="\t", header=1)
            alldata = preliminaryDataWrangling(alldata)
            alldata.columns = [designator + "_" + i for i in alldata.columns]            
        else:
            data = pd.DataFrame.from_csv(filename, sep="\t", header=1)
            data = preliminaryDataWrangling(data)
            data.columns = [groptai(d) for d in data.columns] # because we will be parsing by periods later            
            # review the columns for sets which might need to be consolidated
            maybe_to_consolidate = {key: [c for c in data.columns if key in c] for key in [d.split(".")[0] for d in data.columns]}
            for key in maybe_to_consolidate.keys():
                if len(maybe_to_consolidate[key]) > 1:
                    try:
                        consolidate(data,maybe_to_consolidate[key],key + "_consolidated")                        
                    except KeyError:
                        pass
                else:
                    continue            
            print("=======================")
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

# get rid of redundant columns
duplicate_column_names = list(set([c for c in t1_data.columns if list(t1_data.columns).count(c) > 1]))
t1_data.drop(duplicate_column_names,axis=1,inplace=True)

print("")
print("Number of observations at T1:")
print(len(t1_data))
print("Number of observations at T2:")
print(len(t2_data))
print("Number of features at T1:")
print(len(t1_data.columns))
print("Number of features at T2:")
print(len(t2_data.columns))

# just peer groups 12:15
t1_data = t1_data[t1_data["RCK_QTLY AVG OF TOTAL ASSETS_consolidated"]< 5 * 10**7]
t2_data = t2_data[t2_data["RCK_QTLY AVG OF TOTAL ASSETS_consolidated"]< 5 * 10**7]

# drop columns missing 20% or more of their data
for i, tc in enumerate(t1_data.columns):
    if i % 10 == 0:
        print("Starting on feature " + str(i+1))
    try:
        if len(t1_data[tc]) - len(t1_data[tc].dropna()) >= .80 * len(t1_data[tc]):        
            t1_data[tc].replace(np.nan,t1_data[tc].mean(),inplace=True)
    except KeyError:
        pass
print("Number of features after losing features with too few values remaining: " + str(len(t1_data.columns)))

# remove all other features with missing values
t1_data.dropna(axis=1,how="any",inplace=True)

for tc in t1_data.columns:
    if tc not in t2_data.columns:
        t1_data.drop(tc,axis=1,inplace=True)

from sklearn.preprocessing import StandardScaler

# scale
t1_data_scaled = pd.DataFrame(data=StandardScaler().fit_transform(t1_data.values),columns=t1_data.columns,index=t1_data.index)

# drop the low-variance features
for t1d in t1_data_scaled:
    if t1_data_scaled[t1d].std() ** 2 < .8 * (1 - .8): # variance threshold suggested by sklearn documentation
        t1_data_scaled.drop(t1d,axis=1,inplace=True)

# Visualize the scatter of just 2 and see if that gives you an idea of k
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

initial_pca = PCA(n_components=2).fit_transform(t1_data_scaled.values)
plt.scatter(initial_pca[:,0],initial_pca[:,1])
plt.show()
# does not look like it wants a cluster solution, looks largely covarying except for outliers

# gotta get rid of outliers
pca = pd.DataFrame(data=PCA(n_components=10).fit_transform(t1_data_scaled.values),columns=["PC" + str(i) for i in range(1,11)],index=t1_data_scaled.index)
from scipy.spatial import distance
centroid = list(pca.mean())
pca["distance_from_centroid"] = pca.apply(lambda row: distance.euclidean(centroid,(row[pca.columns])),axis=1) # calculate distance from centroid
average_distance_from_centroid = pca["distance_from_centroid"].mean()
std_distance_from_centroid = pca["distance_from_centroid"].std() # only need +3 bcause it's bigger than the avergae
pca["outliers"] = pca.apply(lambda row: 1 if row["distance_from_centroid"] > 3.29 * std_distance_from_centroid + average_distance_from_centroid else 0,axis=1)

print("Outliers constitute " + str(round(100 * pca["outliers"].sum()/len(pca),2)) + "% of the sample")

# maybe keep outliers for later inspection, but since it's so little of the sample, dropping for now
outliers_IDs = list(pca[pca["outliers"]==1].index)
outliers_pca = pca.loc[outliers_IDs]
outliers_t1_data_scaled = t1_data.loc[outliers_IDs]
outliers = pd.concat([outliers_pca,outliers_t1_data_scaled],axis=1,join="inner")
outliers.drop("distance_from_centroid",inplace=True)

pca.drop(outliers_IDs,axis=0,inplace=True)
t1_data_scaled.drop(outliers_IDs,axis=0,inplace=True)

# cluster; get the right k
from sklearn.cluster import AgglomerativeClustering, SpectralClustering
from sklearn.metrics import silhouette_score

cluster_solution_scores = {k: {clusterType: {dataType: {"labels": [], "silhouette": []} 
                            for dataType in ["pca","raw"]}
                            for clusterType in ["agglomerative","kernel-based"]}
                            for k in range(2,11)}

for k in range(2,11):
    print("Starting on a " + str(k) + "-cluster solution")
    cluster_solution = SpectralClustering(n_clusters=k).fit(pca.values)        
    cluster_solution_scores[k]["kernel-based"]["pca"]["labels"] = cluster_solution.labels_
    cluster_solution_scores[k]["kernel-based"]["pca"]["silhouette"] = silhouette_score(pca,cluster_solution.labels_,metric="euclidean")
    
for k in range(2,11):
    print("Silhouette score for " + str(k) + " clusters: " + str(round(cluster_solution_scores[k]["agglomerative"]["raw"]["silhouette"],2)))



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
