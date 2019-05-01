# -*- coding: utf-8 -*-
"""
Created on Mon Apr 29 15:53:27 2019

@author: ekh9
"""
# https://cdr.ffiec.gov/public/PWS/DownloadBulkData.aspx


# automate a pipeline where you smush things together if they are in redundant columns (like the assets and assets.1 in rck)
# split up categorical variables and continuous
# do some feature selection on the continuous variables




# ideally I'd have a way to automate the scraping
# but right now I'm just downloading it by hand like a sucker

# first want to get a look at the data
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_selection import VarianceThreshold

#path = "C:/Users/ekh9/Desktop/PL_interview/FFIEC CDR Bulk All UBPR Stats 2018"
path = "C:/Users/ekh9/Desktop/PL_interview/FFIEC CDR Call Bulk All Schedules 12312018"
path = "/Users/eleanorhanna/Desktop/job_search_20192020/industry/PrecisionLender_homework/FFIEC CDR Call Bulk All Schedules 12312018"

# read in files
filenames = os.listdir(path)
filenames.remove("Readme.txt")
filenames.remove("FFIEC CDR Call Bulk POR 12312018.txt") # summary, not needed for this analysis
filenames.remove("FFIEC CDR Call Schedule NARR 12312018.txt") # just notes

# join the tables
for i, filename in enumerate(filenames):
    designator = filename.split(" ")[4] # so I know which file it was from originally - there are some redundant names
    if i==0:
        megadf = pd.DataFrame.from_csv(path + "/" + filename, sep="\t", header=1)
        megadf.dropna(axis=1,how="all",inplace=True)
        megadf.columns = [designator + "_" + i for i in megadf.columns]
    else:
        data = pd.DataFrame.from_csv(path + "/" + filename, sep="\t", header=1)
        data.dropna(axis=1,how="all",inplace=True)
        data.columns = [designator + "_" + i for i in data.columns]
        if filename=='FFIEC CDR Call Schedule RIE 12312018.txt': # the index is all strings
            data["idx"] = data.index
            for di in list(data.index):
                try:
                    data.loc[di,"idx"] = np.int64(di)
                except ValueError:
                    print(di)
                    data.drop(di,axis=0,inplace=True)
            data.set_index('idx',inplace=True)
        megadf = pd.concat([megadf,data],axis=1,join="inner")
        
# check to make sure these columns are redundant
x = megadf[["RCK_QTLY AVG OF TOTAL ASSETS","RCRI_QTLY AVG OF TOTAL ASSETS"]]
print(len(x.dropna(axis=0,how="all"))==len(x[x["RCK_QTLY AVG OF TOTAL ASSETS"]==x["RCRI_QTLY AVG OF TOTAL ASSETS"]]))
y = megadf[["RCK_QTLY AVG OF TOTAL ASSETS.1","RCRI_QTLY AVG OF TOTAL ASSETS.1"]]
print(len(y.dropna(axis=0,how="all"))==len(y[y["RCK_QTLY AVG OF TOTAL ASSETS.1"]==y["RCRI_QTLY AVG OF TOTAL ASSETS.1"]]))

megadf.drop(axis=1,labels=["RCRI_QTLY AVG OF TOTAL ASSETS","RCRI_QTLY AVG OF TOTAL ASSETS.1"],inplace=True)

# are these the same value but split into two columns?
rckNans_hdr1 = megadf[np.isnan(megadf["RCK_QTLY AVG OF TOTAL ASSETS"])==False]
rckNans_hdr2 = megadf[np.isnan(megadf["RCK_QTLY AVG OF TOTAL ASSETS.1"])==False]

# is every observation in just one or the other?
print(len([r for r in rckNans_hdr1.index if r in rckNans_hdr2.index]))
print(len([r for r in rckNans_hdr2.index if r in rckNans_hdr1.index]))

# yes...so I will combine them into one
megadf["assets_for_peer_group_assignment"] = megadf["RCK_QTLY AVG OF TOTAL ASSETS"].apply(lambda x: x if np.isnan(x)==False else 0) + megadf["RCK_QTLY AVG OF TOTAL ASSETS.1"].apply(lambda x: x if np.isnan(x)==False else 0)

# check for nans
count_nan = len(megadf) - megadf["assets_for_peer_group_assignment"].count()

# let's see what I even have in terms of peer group options
bins = [0,50000000,100000000,300000000,1000000000,3000000000]
pg_hist, bin_edges = np.histogram(megadf["assets_for_peer_group_assignment"],bins)
fig, ax = plt.subplots()
ax.bar(range(len(pg_hist)),pg_hist,width=1)
ax.set_xticks([i for i,j in enumerate(pg_hist)])
ax.set_xticklabels(['PG12:15','PG8:11','PG4:7','PG3','PG2','PG1'])
############ ADD AXIS LABELS
plt.show()

# so the vast majority of banks have less than $50 million in total assets last quarter on average
smallBanks = megadf[megadf["assets_for_peer_group_assignment"]<50000000]

# let's try to reduce this down
# first I will get rid of low-variance features

def variance(ary):
    return sum([(a - ary.mean())**2 for a in ary])/(len(ary)-1)

cat_features = []
cont_features = []
wtf_features = []

for i, sc in enumerate(smallBanks.columns):
    sample = smallBanks.iloc[0, i]
    if isinstance(sample,(bool,str)) or str(sample).lower()=="true" or str(sample).lower()=="false":
        cat_features.append(sc)
    elif isinstance(sample,(int,np.int64,float)):
        cont_features.append(sc)
    else:
        wtf_features.append(sc)

#for sc in smallBanks.columns:
#    if variance(smallBanks)
