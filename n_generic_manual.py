import matplotlib.pyplot as plt
import numpy as np
from pylab import *
import math
import astropy.units as u
from astropy.coordinates import SkyCoord
from astroquery.gaia import Gaia
from astroquery.simbad import Simbad
from sklearn.ensemble import RandomForestClassifier
import os
import matplotlib.patches as mpatches

#remove the # in the parameter areas for your cluster.
ngc_#### = {
    "cluster_name" : "NGC_####",
    "name" : "NGC ####",
    "filename" : "n####",
    "center" : [###, ###], #e.g. for ngc_2420, this is PROPER MOTIONS [right ascension,declination.]
    "good_radius" : ###, # in miliarcseconds
    "bad_radius" : ###, # in miliarcseconds
    "parallax" : [###,###], # in miliarcseconds (angle parallax)
    "use_search" : False, # means you have a data file for ngc 2420. do True and the following if you don't.
    "coord" : SkyCoord(ra=#ra, dec=#dec, unit=(u.degree, u.degree), frame='icrs') #Note, if the above is True this field may be null or even undefined.
}
cluster_params = ngc_663

# read cone search results from a file
inputfilename = cluster_params.get("filename") + '_cone.csv' #File that must be provided if use_search is false.
modelfilename = cluster_params.get("filename") + '_model_file.data' #File that is generated if this script has been run before (OK if it doesnt exist.)
input_table_base = cluster_params.get("cluster_name") + '_clean_gaia_data.csv' #File that is ALWAYS NEEDED. Training set probabilities.

ID=[];ra=[];dec=[];plx=[];eplx=[];pmra=[];epmra=[];pmdec=[];epmdec=[];g=[];bp=[];rp=[];bprp=[];rv=[];teff=[];ag=[];ebprp=[];vflag=[]
if cluster_params["use_search"] == True: 
     import Gaias_Data_Collect as gdc
     astrtble = gdc.return_table_with_data(skyfile = cluster_params.get("filename") + 'data.dat', skyfile2 = cluster_params.get("cluster_name") + 'data_sim.dat', coord = cluster_params["coord"],mode=0)
     # import csv
     ID = astrtble["DESIGNATION"][astrtble['parallax'] > 0]
     ra = astrtble['ra'][astrtble['parallax'] > 0]
     dec = astrtble['dec'][astrtble['parallax'] > 0]
     plx = astrtble['parallax'][astrtble['parallax'] > 0]
     eplx = astrtble['parallax_error'][astrtble['parallax'] > 0]
     pmra = astrtble['pmra'][astrtble['parallax'] > 0]
     epmra = astrtble['pmra_error'][astrtble['parallax'] > 0]
     pmdec = astrtble['pmdec'][astrtble['parallax'] > 0]
     epmdec = astrtble['pmdec_error'][astrtble['parallax'] > 0]
     g = astrtble['phot_g_mean_mag'][astrtble['parallax'] > 0]
     bp = astrtble['phot_bp_mean_mag'][astrtble['parallax'] > 0]
     rp = astrtble['phot_rp_mean_mag'][astrtble['parallax'] > 0]
     rv = astrtble['radial_velocity'][astrtble['parallax'] > 0]
     #bprp = astrtble['bp_rp'][astrtble['parallax'] > 0]
     bprp = np.subtract(bp,rp)
else:
    import csv
    ID=[];ra=[];dec=[];plx=[];eplx=[];pmra=[];epmra=[];pmdec=[];epmdec=[];g=[];bp=[];rp=[];bprp=[];rv=[];teff=[];ag=[];ebprp=[];vflag=[]
    with open(inputfilename,'r',newline='') as csvfile:
        cr = csv.reader(csvfile)
        ict = 0
        for row in cr:
            if ict == 0:
                columnnames = row
                ict = ict + 1
            else:
                ict = ict + 1
                tmplist=[];
                #ID.append(row[0])
                tmplist.append(row[0])
                try:
                        #ra.append(float(row[1]))  # The downside of csv format is
                        tmplist.append(float(row[1]))
                        if float(row[1]) <= -999.0:
                            continue
                except:                        # having to cast the floats
                        #ra.append(-999.)          # instead of simply reading them
                        continue
                try:                           # in as floats to start with.
                        #dec.append(float(row[2]))
                        tmplist.append(float(row[2]))
                        if float(row[2]) <= -999.0:
                            print("test")
                            continue
                except:
                        #dec.append(-999.)
                        continue #by doing this, anything that'd write a -999 value will be skipped.
                try:
                        #plx.append(float(row[3]))
                        tmplist.append(float(row[3]))
                        if float(row[3]) <= -999.0:
                            print("test1")
                            continue
                except:
                        #plx.append(-999.)
                        continue
                # try:
                #      #eplx.append(float(row[4]))
                #      tmplist.append(float(row[4]))
                # except:
                #      #eplx.append(-999.)
                #      continue
                try:
                        #pmra.append(float(row[5]))
                        tmplist.append(float(row[5]))
                        if float(row[5]) <= -999.0:
                            print("test2")
                            continue
                except:
                        #pmra.append(-999.)
                        continue
                # try:
                #      #epmra.append(float(row[6]))
                #      tmplist.append(float(row[6]))
                # except:
                #      #epmra.append(-999.)
                #      continue
                try:
                        #pmdec.append(float(row[7]))
                        tmplist.append(float(row[7]))
                        if float(row[7]) <= -999.0:
                            print("test3")
                            continue
                except:
                        #pmdec.append(-999.)
                        continue
                # try:
                #      #epmdec.append(float(row[8]))
                #      tmplist.append(float(row[8]))
                # except:
                #      #epmdec.append(-999.)
                #      continue
                try:
                        #g.append(float(row[9]))
                        tmplist.append(float(row[9]))
                        if float(row[9]) <= -999.0:
                            print("test4")
                            continue
                except:
                        #g.append(-999.)
                        continue
                try:
                        #bprp.append(float(row[12]))
                        tmplist.append(float(row[12]))
                        if float(row[12]) <= -999.0:
                            print("test5")
                            continue
                except:
                        #bprp.append(-999.)
                        continue
                ID.append(tmplist[0])
                ra.append(tmplist[1])
                dec.append(tmplist[2])
                plx.append(tmplist[3])
                pmra.append(tmplist[4])
                pmdec.append(tmplist[5])
                g.append(tmplist[6])
                bprp.append(tmplist[7])

ra = np.array(ra); dec=np.array(dec); pmra=np.array(pmra); pmdec=np.array(pmdec)
g = np.array(g); bprp=np.array(bprp); plx=np.array(plx)
jtrain = np.zeros(len(ID),dtype=int) # 0 for not in training list, 1 for in the list
ptrain = np.zeros(len(ID),dtype=float) # store probabilities here
colors = ['']*len(ID)

c_path = os.getcwd()
n_path = os.path.join(c_path,cluster_params.get("cluster_name"))
if not os.path.exists(n_path):
     os.mkdir(n_path)
rcParams.update({'axes.titlesize':'large'})
# 1. plot proper motions to draw a generous circle for member/nonmember
plot(pmra,pmdec,'b+')
xlabel(r'$\mu_\alpha$ (mas yr$^{-1}$)')
ylabel(r'$\mu_\delta$ (mas ys$^{-1}$)')
title('Proper motions of ' + cluster_params["name"])
cluster_params.get("cluster_name")
plt.savefig(cluster_params.get("cluster_name") + '/' + cluster_params.get("cluster_name") + '_fig_1_pm.png',dpi=200)
clf()

import csv
csvfile = input_table_base
# ID and probability (blank lists)
CID = []
mprob = []
with open(input_table_base) as csvfile:
    dreader = csv.reader(csvfile)
    for row in dreader:
        if row[1] == 'probability':
            continue
        CID.append(row[0])
        mprob.append(float(row[1]))

sizes = []
alphas = []
ptraintrue = np.zeros(len(ID),dtype=float) # store probabilities here
for i in range(len(ra)):
     try:
          test_index = CID.index(ID[i])
          if(mprob[test_index] > 0.10):
               if(mprob[test_index] < 0.50):
                    jtrain[i] = 0
                    ptrain[i] = max(min(1,mprob[test_index]),0)
                    colors[i] = 'xkcd:light gray'
                    ptraintrue[i]=0
                    sizes.append(1)
                    continue
               jtrain[i] = 1
               ptrain[i] = max(min(1,mprob[test_index]),0)
               colors[i] = 'xkcd:black'
               ptraintrue[i]=max(min(1,mprob[test_index]),0)
               sizes.append(2.5)
          else:
               jtrain[i] = 1
               ptrain[i] = max(min(1,mprob[test_index]),0)
               ptraintrue[i]=1
               colors[i] = 'xkcd:booger'
               sizes.append(1)
     except ValueError:
        d = math.sqrt( (pmra[i] + cluster_params["center"][0])**2 + (pmdec[i] + cluster_params["center"][1])**2 )   
        if d < cluster_params["good_radius"]:
            jtrain[i] = 1          # "1" simply means "in our training set"
            ptrain[i] = 1.0        # probability (crudely assigned)
            colors[i] = 'xkcd:black'
            sizes.append(2.5)
        elif d > cluster_params["bad_radius"]:
            jtrain[i] = 1
            ptrain[i] = 0.0
            colors[i] = 'xkcd:booger'
            sizes.append(0.01)
        else:
            colors[i] = 'xkcd:light gray'
            sizes.append(1)
            ptraintrue[i]=0
     

# hone our selection with a cut on parallax distance
for i in range(len(ra)):
     d = plx[i]
     if (d > cluster_params["parallax"][0]) or (d < cluster_params["parallax"][1]):   # also manually selected
          try:
               test_index = CID.index(ID[i])
               if(mprob[test_index] > 0.70):
                    continue
               else:
                    colors[i] = 'xkcd:booger'
                    sizes[i] = 0.01
                    jtrain[i] = 1
                    ptrain[i] = 0.0
          except:
               colors[i] = 'xkcd:booger'
               sizes[i] = 0.01
               jtrain[i] = 1
               ptrain[i] = 0.0
               ptraintrue[i]=1

nlstbp = []
nlstg = []
nlstc = []
nlstpr = []
nlstpd = []
nlstra = []
nlstdc = []
for i in range(len(bprp)):
     if colors[i] == 'xkcd:booger':
          continue
     nlstbp.append(bprp[i])
     nlstg.append(g[i])
     nlstc.append(colors[i])
     nlstpr.append(pmra[i])
     nlstpd.append(pmdec[i])
     nlstra.append(ra[i])
     nlstdc.append(dec[i])

handles=[mpatches.Patch(color='xkcd:aquamarine', label=r'$\geq$'+'90%'),mpatches.Patch(color='xkcd:pale green', label=r'$\geq$'+'80%'),mpatches.Patch(color='xkcd:pale yellow', label=r'$\geq$'+'70%'),mpatches.Patch(color='xkcd:mustard yellow', label=r'$\geq$'+'60%'),mpatches.Patch(color='xkcd:light red', label=r'$\geq$'+'50%'),
               mpatches.Patch(color='xkcd:rust', label=r'$\geq$'+'40%'),mpatches.Patch(color='xkcd:puce', label=r'$\geq$'+'30%'),mpatches.Patch(color='xkcd:plum', label=r'$\geq$'+'20%'),mpatches.Patch(color='xkcd:royal purple', label=r'$\geq$'+'10%')]
handles2=[mpatches.Patch(color='xkcd:black', label=r'$\geq$'+'50%'),mpatches.Patch(color='xkcd:booger', label=r'$\equal$'+'0%'),mpatches.Patch(color='xkcd:gray', label=r'$\equal$'+'??%')]
handles3=[mpatches.Patch(color='xkcd:black', label=r'$\geq$'+'50%'),mpatches.Patch(color='xkcd:gray', label=r'$\equal$'+'??%')]
# plot training data
#fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(12, 6))
#fig, axis = plt.subplots(nrows=2, ncols=2)
scatter(ra, dec, c=colors, s=2)
title("Star Membership Inclusion: \n" + cluster_params["name"] + " Right Ascension v. Declination")
ylabel(r'Declination (deg)')
xlabel(r'Right Ascension (deg)')
legend(handles=handles2)
plt.savefig(cluster_params.get("cluster_name") + '/' + cluster_params.get("cluster_name") + '_fig_2_radecbf.png',dpi=200)
clf()

scatter(nlstra, nlstdc, c=nlstc, s=2)
title("Star Membership Inclusion: \n" + cluster_params["name"] + " Right Ascension v. Declination")
ylabel(r'Declination (deg)')
xlabel(r'Right Ascension (deg)')
legend(handles=handles2)
plt.savefig(cluster_params.get("cluster_name") + '/' + cluster_params.get("cluster_name") + '_fig_2_radecbf_flt.png',dpi=200)
clf()

scatter(pmra, pmdec, c=colors, s=2)
title("Star Membership Inclusion: " + cluster_params["name"] + "\nProper Motions Right Ascension v. Proper Motions Declination")
ylabel(r'Proper Motions Declination (mas yr$^{-1}$)')
xlabel(r'Proper Motions Right Ascension (mas yr$^{-1}$)')
legend(handles=handles2)
plt.savefig(cluster_params.get("cluster_name") + '/' + cluster_params.get("cluster_name") + '_fig_3_pmbf.png',dpi=200)
clf()

scatter(nlstpr, nlstpd, c=nlstc, s=2)
title("Star Membership Inclusion: " + cluster_params["name"] + "\nProper Motions Right Ascension v. Proper Motions Declination")
ylabel(r'Proper Motions Declination (mas yr$^{-1}$)')
xlabel(r'Proper Motions Right Ascension (mas yr$^{-1}$)')
legend(handles=handles2)
plt.savefig(cluster_params.get("cluster_name") + '/' + cluster_params.get("cluster_name") + '_fig_3_pmbf_flt.png',dpi=200)
clf()

comp1=plx[(ptrain < 0.5)]
comp2=plx[(ptrain > 0.5)]
hist([comp1,comp2],bins=100,range=(0.0,cluster_params["parallax"][0]*2.5),stacked=True,color=['xkcd:light gray','xkcd:crimson'])
title("Parallax of " + cluster_params["name"] + ".\n(Red is the estimated members.)")
xlabel('Parallax (miliarcsec)')
ylabel('Number of Stars')
plt.savefig(cluster_params.get("cluster_name") + '/' + cluster_params.get("cluster_name") + '_fig_4_pxbf.png',dpi=200)
clf()

scatter(bprp,g,c=colors, s=2)
xlim([-1,4])
ylim([22.0,6.0])
xlabel('BP - RP (magnitude)')
ylabel('Gaia Filter Apparent Magnitude')
title(cluster_params["name"]+" Color Magnitude Diagram\nEstimated Probabilities")
legend(handles=handles2)
plt.savefig(cluster_params.get("cluster_name") + '/' + cluster_params.get("cluster_name") + '_fig_5_bprp.png',dpi=200)
clf()

scatter(nlstbp,nlstg,c=nlstc, s=2)
xlim([-1,4])
ylim([22.0,6.0])
xlabel('BP - RP (magnitude)')
ylabel('Gaia Filter Apparent Magnitude')
title(cluster_params["name"]+" Color Magnitude Diagram\nEstimated Probabilities")
legend(handles=handles3)
plt.savefig(cluster_params.get("cluster_name") + '/' + cluster_params.get("cluster_name") + '_fig_5_bprp_flt.png',dpi=200)
clf()

foo = jtrain[ptrain > 0.5]
print('Before random forest, we have ',len(foo),'members')


# Random Forest
import time
# prepare the data: stack the arrays
data = np.stack((ra,dec,pmra,pmdec,plx),axis=-1)
#import joblib as job
import pickle as pick
try:
     o_rfc = pick.load(open(modelfilename,'rb'))
     fresh = False
except:
     o_rfc = NAN
     fresh = True
### recipe: https://stackabuse.com/random-forest-algorithm-with-python-and-scikit-learn/
score = 0
iterations = 0
rfmem = []
max_score = 0
n_rfc = any
bests = [0, 0, 0, 0, 0]
#SEED = 188232   # a random number
SEED = randint(1,100000)
est = randint(10,360)
dep = randint(5,24)
rfc = RandomForestClassifier(n_estimators=est,max_depth=dep,random_state=SEED)
#rfc = RandomForestClassifier(n_estimators=5,max_depth=2,random_state=SEED)
### Train the RF 
data2 = np.stack((ra[jtrain==1],dec[jtrain==1],pmra[jtrain==1],pmdec[jtrain==1],plx[jtrain==1]),axis=-1)
ytrain = ptrain[jtrain==1]
ytrain2 = np.zeros(len(ytrain),dtype=int)
for i in range(len(ytrain)):
     if ytrain[i] < 0.50:
          ytrain2[i] = 0
     else:
          ytrain2[i] = 1
     #ytrain2[i] = ytrain[i]
data3 = np.stack((ra[ptraintrue>0],dec[ptraintrue>0],pmra[ptraintrue>0],pmdec[ptraintrue>0],plx[ptraintrue>0]),axis=-1)
ytrain3 = ptrain[ptraintrue>0]
ytrain4 = np.zeros(len(ytrain3),dtype=int)
for i in range(len(ytrain3)):
     if ytrain3[i] < 0.50:
          ytrain4[i] = 0
     else:
          ytrain4[i] = 1
     #ytrain4[i] = ytrain3[i]
if fresh == True: 
     beegscore = 0.998
else:
     try:
          beegscore = max(0.95,o_rfc.score(data3,ytrain4))
     except:
          beegscore = 0.998
redo = 1
print("Score to Beat:",max(beegscore,0.95))

#Weighting the sample
sampletrain = np.zeros(len(ytrain3),dtype=int)
for i in range(len(ytrain3)):
     sampletrain[i] = min(2,max(0,1+4*((ytrain3[i]+0.5)**2)))

skip = False # debug
skip_if_Not_Fresh = False
if fresh == False and skip_if_Not_Fresh == True:
     skip = True
while score < beegscore and iterations < 20:
     if skip == True:
          break
     iterations += 1
     # Random Forest setup:
     #SEED = 188232   # a random number
     if redo == 1:
          SEED=randint(1,100000)
          #rfc = RandomForestClassifier(n_estimators=5,max_depth=2,random_state=SEED)
          est = randint(10,500)
          dep = randint(5,60)
          rfc = RandomForestClassifier(n_estimators=est,max_depth=dep,random_state=SEED)
     else:
          SEED=randint(1,100000)
          #est = randint(10,177)
          #dep = randint(4,8)
          rfc.set_params(n_estimators=est,max_depth=dep,random_state=SEED)
     rfc.fit(data2,ytrain2)
     score = rfc.score(data3,ytrain4,sampletrain)
     time.sleep(1)
     print("Iteration:",iterations,"; Score:",score, "; Estimators:",est,"; Depth:",dep,"; Last RFC Kept", ["true","false"][redo])
     redo = 0 #By default, keep the old classifier (don't fix what isn't borked)
     if(score > max_score):
          n_rfc = rfc
          max_score = score
          bests[0] = iterations
          bests[1] = score
          bests[2] = est
          bests[3] = dep
          bests[4] = SEED
          if(score > beegscore):
               break
     else:
          redo = 1
          #Create a completely new classifier.

print("Best RFC: Iteration:",bests[0],"; Score:",bests[1], "; Estimators:",bests[2],"; Depth:",bests[3],"; Seed:",bests[4])
class DEVSkipTrue(Exception):
     "[DEV] Skip set to true, despite there being no saved model."
     pass
#Save training set
if fresh == True:
     #job.dump(n_rfc, filename=modelfilename)
     pick.dump(n_rfc,open(modelfilename,'wb'))
     if skip == True:
          raise DEVSkipTrue
else:
     if bests[1] > beegscore: 
          #job.dump(n_rfc, filename=modelfilename)
          pick.dump(n_rfc,open(modelfilename,'wb'))
     if skip == True:
          n_rfc = o_rfc
#Essentially, replace training set if it doesn't exist, or if this training set is better than the old one.

# Predict using this training set
rfmem = n_rfc.predict(data)

rfmem2 = [] ; memcount=0; noncount=0
sizes = []

### map the colors.  gm[] is full of zeros and ones. Let's make a gm2 with colors in it.
rfmem2 = []   # empty 'list' object
rfmemtest = n_rfc.predict_proba(data)
rfmem2test = []
snum = 1
for i in range(len(rfmemtest)):
     if rfmemtest[i][snum] >= 0.90:
          rfmem2test.append('xkcd:aquamarine')
          memcount = memcount + 1
          sizes.append(2.25)
     elif rfmemtest[i][snum] >= 0.80:
          rfmem2test.append('xkcd:pale green')
          sizes.append(2)
          memcount = memcount + 1
     elif rfmemtest[i][snum] >= 0.70:
          rfmem2test.append('xkcd:pale yellow')
          sizes.append(1.75)
          memcount = memcount + 1
     elif rfmemtest[i][snum] >= 0.60:
          rfmem2test.append('xkcd:mustard yellow')
          sizes.append(1.5)
          memcount = memcount + 1
     elif rfmemtest[i][snum] >= 0.50:
          rfmem2test.append('xkcd:light red')
          sizes.append(1.35)
          memcount = memcount + 1
     elif rfmemtest[i][snum] >= 0.40:
          rfmem2test.append('xkcd:rust')
          sizes.append(1)
          noncount = noncount + 1
     elif rfmemtest[i][snum] >= 0.30:
          rfmem2test.append('xkcd:puce')
          sizes.append(0.75)
          noncount = noncount + 1
     elif rfmemtest[i][snum] >= 0.20:
          rfmem2test.append('xkcd:plum')
          sizes.append(0.5)
          noncount = noncount + 1
     elif rfmemtest[i][snum] >= 0.10:
          rfmem2test.append('xkcd:royal purple')
          sizes.append(0.35)
          noncount = noncount + 1
     else:
          rfmem2test.append('xkcd:navy blue')
          sizes.append(0.25)
          noncount = noncount + 1
rfmem2 = rfmem2test
n_rfmem = []
n_bprp = []
n_g = []
n_pmra = []
n_pmdec = []
n_ra = []
n_dec = []
for i in range(len(rfmem2)):
     if rfmem2[i] == 'xkcd:navy blue':
          continue
     if rfmem2[i] == 'xkcd:royal purple':
          continue
     if rfmem2[i] == 'xkcd:plum':
          continue
     if rfmem2[i] == 'xkcd:puce':
          continue
     if rfmem2[i] == 'xkcd:rust':
          continue
     n_rfmem.append(rfmem2[i])
     n_bprp.append(bprp[i])
     n_g.append(g[i])
     n_pmra.append(pmra[i])
     n_pmdec.append(pmdec[i])
     n_ra.append(ra[i])
     n_dec.append(dec[i])
n_probs = []
for i in range(len(rfmem2)):
     n_probs.append(rfmemtest[i][snum] * 100)

print('After random forest, we have ',memcount,'members')

#fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(12, 6))
#fig, axis = plt.subplots(nrows=2, ncols=2)
scatter(ra,dec, c=rfmem2, s=sizes)
title("Star Membership Inclusion: \n" + cluster_params["name"] + " Right Ascension v. Declination")
ylabel(r'Declination (deg)')
xlabel(r'Right Ascension (deg)')
legend(handles=handles)
plt.savefig(cluster_params.get("cluster_name") + '/' + cluster_params.get("cluster_name") + '_fig_6_radecaf.png',dpi=200)
clf()

scatter(n_ra,n_dec, c=n_rfmem, s=2)
title("Star Membership Inclusion: \n" + cluster_params["name"] + " Right Ascension v. Declination")
ylabel(r'Declination (deg)')
xlabel(r'Right Ascension (deg)')
legend(handles=handles)
plt.savefig(cluster_params.get("cluster_name") + '/' + cluster_params.get("cluster_name") + '_fig_6_radecaf_flt.png',dpi=200)
clf()

scatter(pmra, pmdec, c=rfmem2, s=sizes)
title("Star Membership Inclusion: " + cluster_params["name"] + "\nProper Motions Right Ascension v. Proper Motions Declination")
ylabel(r'Proper Motions Declination (mas yr$^{-1}$)')
xlabel(r'Proper Motions Right Ascension (mas yr$^{-1}$)')
legend(handles=handles)
plt.savefig(cluster_params.get("cluster_name") + '/' + cluster_params.get("cluster_name") + '_fig_7_pmaf.png',dpi=200)
clf()

scatter(n_pmra, n_pmdec, c=n_rfmem, s=2)
title("Star Membership Inclusion: " + cluster_params["name"] + "\nProper Motions Right Ascension v. Proper Motions Declination")
ylabel(r'Proper Motions Declination (mas yr$^{-1}$)')
xlabel(r'Proper Motions Right Ascension (mas yr$^{-1}$)')
legend(handles=handles)
plt.savefig(cluster_params.get("cluster_name") + '/' + cluster_params.get("cluster_name") + '_fig_7_pmaf_flt.png',dpi=200)
clf()

#show()

#fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(12, 6))
hist(n_probs,bins=100,range=(0.0,100.0),stacked=True,color='xkcd:red')
title("Probability Distributions of " + cluster_params["name"] + ".\n(From 0% to 100%)")
xlabel('Probability')
ylabel('Number of Stars')
plt.savefig(cluster_params.get("cluster_name") + '/' + cluster_params.get("cluster_name") + '_fig_11_probdist.png',dpi=200)
clf()

n2_probs = []
for i in range(len(n_probs)):
     if(n_probs[i] == 0):
          continue
     n2_probs.append(rfmemtest[i][snum] * 100)
hist(n2_probs,bins=100,range=(0.0,100.0),stacked=True,color='xkcd:red')
title("Probability Distributions of " + cluster_params["name"] + ".\n(From 0% to 100%) (0% Excluded)")
xlabel('Probability')
ylabel('Number of Stars')
plt.savefig(cluster_params.get("cluster_name") + '/' + cluster_params.get("cluster_name") + '_fig_11_probdist_flt.png',dpi=200)
clf()

n2_probs = []
for i in range(len(n_probs)):
     if(n_probs[i] <= 5):
          continue
     n2_probs.append(rfmemtest[i][snum] * 100)
hist(n2_probs,bins=100,range=(0.0,100.0),stacked=True,color='xkcd:red')
title("Probability Distributions of " + cluster_params["name"] + ".\n(From 0% to 100%) (0% Excluded)")
xlabel('Probability')
ylabel('Number of Stars')
plt.savefig(cluster_params.get("cluster_name") + '/' + cluster_params.get("cluster_name") + '_fig_11_probdist_flt_2.png',dpi=200)
clf()


#fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(12, 6))
comp1=plx[rfmem <0.2]
comp2=plx[rfmem >= 0.2]
hist([comp1,comp2],bins=100,range=(0.0,cluster_params["parallax"][0]*2.5),stacked=True,color=['xkcd:light gray','xkcd:red'])
title("Parallax of " + cluster_params["name"] + ".\n(Red is the machine-estimated members.)")
xlabel('Parallax (miliarcsec)')
ylabel('Number of Stars')
plt.savefig(cluster_params.get("cluster_name") + '/' + cluster_params.get("cluster_name") + '_fig_8_pxaf.png',dpi=200)
clf()


scatter(bprp,g,c=rfmem2, s=sizes)
xlim([-1,4])
ylim([22.0,6.0])
xlabel('BP - RP (magnitude)')
ylabel('Gaia Filter Apparent Magnitude')
title(cluster_params["name"]+" Color Magnitude Diagram - Probabilities")
legend(handles=handles)
plt.savefig(cluster_params.get("cluster_name") + '/' + cluster_params.get("cluster_name") + '_fig_9_bprpaf.png',dpi=200)
clf()

scatter(n_bprp,n_g,c=n_rfmem, s=2)
xlim([-1,4])
ylim([22.0,6.0])
xlabel('BP - RP (magnitude)')
ylabel('Gaia Filter Apparent Magnitude')
title(cluster_params["name"]+" Color Magnitude Diagram - Probabilities")
legend(handles=handles)
plt.savefig(cluster_params.get("cluster_name") + '/' + cluster_params.get("cluster_name") + '_fig_10_bprpaf_filt_10.png',dpi=200)
clf()




#and now I single out the stragglers. Or try to.
stragglers_data = []
stragglers_data.append(["ID",'In ' + cluster_params["cluster_name"] + '?',"BP-RP","G (mag)","PLX (arcsecs)","Alt Prob"])
for i in range(len(data)):
     # if bprp[i] < 1:
     #      continue
     # if g[i] > 15.5:
     #      continue
     test_list = []
     test_list.append(ID[i])
     test_list.append(rfmemtest[i][snum])
     test_list.append(bprp[i])
     test_list.append(g[i])
     test_list.append(plx[i])
     test_list.append(rfmemtest[i][[1,0][snum]])
     stragglers_data.append(test_list)
import csv
with open(cluster_params["filename"] + '_stragglers_table.csv','w') as f:
     write = csv.writer(f)
     write.writerows(np.array(stragglers_data))
print( 'normal stop' )
