import matplotlib.pyplot as plt
import numpy as np
from pylab import *
import math
import astropy.units as u
from astropy.coordinates import SkyCoord
from astroquery.gaia import Gaia
from astroquery.simbad import Simbad
from sklearn.ensemble import RandomForestClassifier

cluster = 'NGC_6253'
clus2 = 'n6253'

# parameters: center = -4.58 -5.28  , goodradius = 0.17, definitelybadradius = 0.35
#center = [-1.2132, -2.037]
center = [-4.537, -5.280]
goodradius = 0.17
definitelybadradius = 0.35
paralax = [0.68,0.48]

# read cone search results from a file
inputfilename = clus2 + '_cone.csv'
modelfilename = clus2 + '_model_file.data'
input_table_base = cluster + '_clean_gaia_data.csv'

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
               ID.append(row[0])
               try:
                    ra.append(float(row[1]))  # The downside of csv format is
               except:                        # having to cast the floats
                    ra.append(-999.)          # instead of simply reading them
               try:                           # in as floats to start with.
                    dec.append(float(row[2]))
               except:
                    dec.append(-999.)
               try:
                    plx.append(float(row[3]))
               except:
                    plx.append(-999.)
               try:
                    eplx.append(float(row[4]))
               except:
                    eplx.append(-999.)
               try:
                    pmra.append(float(row[5]))
               except:
                    pmra.append(-999.)
               try:
                    epmra.append(float(row[6]))
               except:
                    epmra.append(-999.)
               try:
                    pmdec.append(float(row[7]))
               except:
                    pmdec.append(-999.)
               try:
                    epmdec.append(float(row[8]))
               except:
                    epmdec.append(-999.)
               try:
                    g.append(float(row[9]))
               except:
                    g.append(-999.)
               try:
                    bprp.append(float(row[12]))
               except:
                    bprp.append(-999.)


ra = np.array(ra); dec=np.array(dec); pmra=np.array(pmra); pmdec=np.array(pmdec)
g = np.array(g); bprp=np.array(bprp); plx=np.array(plx)
jtrain = np.zeros(len(ID),dtype=int) # 0 for not in training list, 1 for in the list
ptrain = np.zeros(len(ID),dtype=float) # store probabilities here
colors = ['']*len(ID)


# 1. plot proper motions to draw a generous circle for member/nonmember
plot(pmra,pmdec,'b+')
xlabel(r'$\mu_\alpha$ (mas yr$^{-1}$)')
ylabel(r'$\mu_\delta$ (mas ys$^{-1}$)')
title('Proper motions')
show()

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
          if(mprob[test_index] > 0):
               if(mprob[test_index] < 0.65):
                    jtrain[i] = 0
                    ptrain[i] = max(min(1,mprob[test_index]),0)
                    colors[i] = 'xkcd:light gray'
                    sizes.append(1)
                    continue
               jtrain[i] = 1
               ptrain[i] = max(min(1,mprob[test_index]),0)
               colors[i] = 'xkcd:black'
               ptraintrue[i]=max(min(1,mprob[test_index]),0)
               sizes.append(2.5)
          else:
               jtrain[i] = 1
               ptrain[i] = 0
               ptraintrue[i]=0
               colors[i] = 'xkcd:light gray'
               sizes.append(1)
     except ValueError:
          d = math.sqrt( (pmra[i] + center[0])**2 + (pmdec[i] + center[1])**2 )   
          if d < goodradius:
               jtrain[i] = 1          # "1" simply means "in our training set"
               ptrain[i] = 1.0        # probability (crudely assigned)
               colors[i] = 'xkcd:black'
               sizes.append(2.5)
          elif d > definitelybadradius:
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
     if (d > paralax[0]) or (d < paralax[1]):   # also manually selected
          try:
               test_index = CID.index(ID[i])
               if(mprob[test_index] > 0):
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



# plot training data
#fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(12, 6))
fig, axis = plt.subplots(nrows=2, ncols=2)
axis[0,0].scatter(ra, dec, c=colors, s=2)
axis[0,0].set_title("RA - dec")
axis[0,0].set_ylabel(r'Dec (deg)')
axis[0,0].set_xlabel(r'R.A. (deg)')
axis[0,1].scatter(pmra, pmdec, c=colors, s=2)
axis[0,1].set_title("pmRA - pmDEC")
axis[0,1].set_ylabel(r'pmdec (mas yr$^{-1}$)')
axis[0,1].set_xlabel(r'pmRA (mas yr$^{-1}$)')

#print(len(plx),' should equal',len(jtrain))

comp1=plx[(ptrain < 0.5)]
comp2=plx[(ptrain > 0.5)]
axis[1,0].hist([comp1,comp2],bins=100,range=(0.0,1.0),stacked=True,color=['xkcd:light gray','xkcd:crimson'])
axis[1,0].set_title("Parallax")
axis[1,0].set_xlabel('Parallax (miliarcsec)')
axis[1,0].set_ylabel('N')

axis[1,1].scatter(bprp,g,c=colors, s=2)
axis[1,1].set_xlim([-1,4])
axis[1,1].set_ylim([22.0,8.0])
axis[1,1].set_xlabel('BP-RP')
axis[1,1].set_ylabel('G (mag)')
plt.subplots_adjust(hspace=0.4,wspace=0.4)
show()

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
SEED=randint(1,100000)
est = randint(10,360)
dep = randint(5,24)
rfc = RandomForestClassifier(n_estimators=est,max_depth=dep,random_state=SEED)
#rfc = RandomForestClassifier(n_estimators=5,max_depth=2,random_state=SEED)
### Train the RF 
data2 = np.stack((ra[jtrain==1],dec[jtrain==1],pmra[jtrain==1],pmdec[jtrain==1],plx[jtrain==1]),axis=-1)
ytrain = ptrain[jtrain==1]
ytrain2 = np.zeros(len(ytrain),dtype=int)
for i in range(len(ytrain)):
     if ytrain[i] < 0.5:
          ytrain2[i] = 0
     else:
          ytrain2[i] = 1
data3 = np.stack((ra[ptraintrue>0],dec[ptraintrue>0],pmra[ptraintrue>0],pmdec[ptraintrue>0],plx[ptraintrue>0]),axis=-1)
ytrain3 = ptraintrue[ptraintrue>0]
ytrain4 = np.zeros(len(ytrain3),dtype=int)
for i in range(len(ytrain3)):
     if ytrain3[i] < 0.5:
          ytrain4[i] = 0
     else:
          ytrain4[i] = 1
if fresh == True: 
     beegscore = 0.998
else: 
     beegscore = o_rfc.score(data3,ytrain4)
redo = 1
print("Score to Beat:",beegscore)

skip = True # debug
while score < beegscore and iterations < 15:
     if skip == True:
          break
     iterations += 1
     # Random Forest setup:
     #SEED = 188232   # a random number
     if redo == 1:
          SEED=randint(1,100000)
          #rfc = RandomForestClassifier(n_estimators=5,max_depth=2,random_state=SEED)
          est = randint(10,360)
          dep = randint(5,24)
          rfc = RandomForestClassifier(n_estimators=est,max_depth=dep,random_state=SEED)
     else:
          SEED=randint(1,100000)
          #est = randint(10,177)
          #dep = randint(4,8)
          rfc.set_params(n_estimators=est,max_depth=dep,random_state=SEED)
     rfc.fit(data2,ytrain2)
     score = rfc.score(data3,ytrain4)
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

print('After random forest, we have ',memcount,'members')

#fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(12, 6))
fig, axis = plt.subplots(nrows=2, ncols=2)
axis[0,0].scatter(ra,dec, c=rfmem2, s=sizes)
axis[0,0].set_title("RA - dec")
axis[0,0].set_ylabel(r'Dec (deg)')
axis[0,0].set_xlabel(r'R.A. (deg)')

axis[0,1].scatter(pmra, pmdec, c=rfmem2, s=sizes)
axis[0,1].set_title("pmRA - pmDEC")
axis[0,1].set_ylabel(r'pmdec (mas yr$^{-1}$)')
axis[0,1].set_xlabel(r'pmRA (mas yr$^{-1}$)')

#show()


#fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(12, 6))
comp1=plx[rfmem <0.2]
comp2=plx[rfmem >= 0.2]
axis[1,0].hist([comp1,comp2],bins=100,range=(0.0,1.0),stacked=True,color=['xkcd:light gray','xkcd:red'])
axis[1,0].set_title("Parallax")
axis[1,0].set_xlabel('Parallax (miliarcsec)')
axis[1,0].set_ylabel('N')

axis[1,1].scatter(bprp,g,c=rfmem2, s=sizes)
axis[1,1].set_xlim([-1,4])
axis[1,1].set_ylim([22.0,8.0])
axis[1,1].set_xlabel('BP - RP (magnitude) (Blue Photometry minus Red Photometry)')
axis[1,1].set_ylabel('Gaia Filter Apparent Magnitude')
plt.subplots_adjust(hspace=0.4,wspace=0.4)
show()

n_rfmem = []
n_bprp = []
n_g = []
for i in range(len(rfmem2)):
     if rfmem2[i] == 'xkcd:navy blue':
          continue
     #if rfmem2[i] == 'xkcd:royal purple':
     #     continue
     #if rfmem2[i] == 'xkcd:plum':
     #     continue
     #if rfmem2[i] == 'xkcd:puce':
     #     continue
     #if rfmem2[i] == 'xkcd:rust':
     #     continue
     n_rfmem.append(rfmem2[i])
     n_bprp.append(bprp[i])
     n_g.append(g[i])

fig = plt.scatter(n_bprp,n_g,c=n_rfmem, s=2)
xlim([-1,4])
ylim([22.0,8.0])
xlabel('BP - RP (magnitude) (Blue Photometry minus Red Photometry)')
ylabel('Gaia Filter Apparent Magnitude')
title("Color Magnitude Diagram - Probabilities")
import matplotlib.patches as mpatches
legend(handles=[mpatches.Patch(color='xkcd:aquamarine', label='>=90%'),mpatches.Patch(color='xkcd:pale green', label='>=80%'),mpatches.Patch(color='xkcd:pale yellow', label='>=70%'),mpatches.Patch(color='xkcd:mustard yellow', label='>=60%'),mpatches.Patch(color='xkcd:light red', label='>=50%'),
               mpatches.Patch(color='xkcd:rust', label='>=40%'),mpatches.Patch(color='xkcd:puce', label='>=30%'),mpatches.Patch(color='xkcd:plum', label='>=20%'),mpatches.Patch(color='xkcd:royal purple', label='>=10%')])
show()

#and now I single out the stragglers. Or try to.
stragglers_data = []
stragglers_data.append(["ID",'In ' + cluster + '?',"BP-RP","G (mag)","PLX (arcsecs)"])
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
     stragglers_data.append(test_list)
import csv
with open(clus2 + '_stragglers_table.csv','w') as f:
     write = csv.writer(f)
     write.writerows(np.array(stragglers_data))
print( 'normal stop' )