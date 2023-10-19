import csv
cluster = 'n6253'
input_table_base = 'NGC_6253_clean_gaia_data.csv'
input_table_check = cluster + '_stragglers_table.csv'
csvfile = input_table_base
# ID and probability (blank lists)
CID = []
mprob = []
parallax = []
num_check = 0
with open(input_table_base) as csvfile:
    dreader = csv.reader(csvfile)
    for row in dreader:
        if row[1] == 'probability':
            continue
        CID.append(row[0])
        mprob.append(float(row[1]))
        parallax.append(float(row[6]))
        num_check += 1


# temporary debug/confirmation
# print(CID)
# print(mprob)
# print( 'number of stars = ',len(CID) )


import tabulate
cid2 = []
mprob2 = []
num_check2 = 0
with open(input_table_check) as csvfile:
    dreader = csv.reader(csvfile)
    for row in dreader:
        try:
            if row[0] == 'ID':
                continue
        except:
            continue
        cid2.append(row[0])
        mprob2.append(float(row[1]))
        num_check2 += 1
finalcid = []
finalmprob = []
finalparal = []
num_check3 = 0
for i in range(len(cid2)):
    if cid2[i] in CID:
        finalcid.append(cid2[i])
        cid_index = CID.index(cid2[i])
        finalmprob.append([mprob[cid_index],mprob2[i]])
        finalparal.append(parallax[cid_index])
    else:
        #print(cid2[i], "is not in the 6253 clean data!")
        num_check3 += 1
print("Number of Simbad Stars: ",num_check,"; Number of GAIA Stars (cut):", num_check2, "; Number of GAIA stars not in Simbad star data:",num_check3)
flist = []
for i in range(len(finalcid)):
    flist.append([finalcid[i],finalmprob[i][0],finalmprob[i][1],finalparal[i]])
from operator import itemgetter, attrgetter
n_list = sorted(flist,key=itemgetter(1),reverse=True)
n_list.insert(0,["ID","Probability (Simbad)","Probability (Methodology)","Parallax (arcsec)"])
import tabulate
with open(cluster + '_stragglers_Comparison.dat', "w") as f:
      f.write(tabulate.tabulate(n_list))