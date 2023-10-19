import matplotlib.pyplot as plt
import numpy as np
from pylab import *

import astropy.units as u
from astropy.coordinates import SkyCoord
from astroquery.gaia import Gaia
from astroquery.simbad import Simbad

from sklearn.mixture import GaussianMixture
from sklearn.mixture import *
# if error here, "pip install scikit-learn"
from sklearn.ensemble import RandomForestClassifier


# perform a Gaia cone search and get the data
#coord = SkyCoord(ra=117.465, dec=-17.225, unit=(u.degree, u.degree), frame='icrs') # Ruprecht 37
coord = SkyCoord(ra=130.054, dec=19.621, unit=(u.degree, u.degree), frame='icrs') # M44 = NGC 2632
#coord = SkyCoord(ra=290.221, dec=37.778, unit=(u.degree, u.degree), frame='icrs') # NGC 6791

radius = u.Quantity(0.45, u.deg)


Gaia.MAIN_GAIA_TABLE = "gaiadr3.gaia_source"  # Reselect Data Release 3, 3 is NOT default?
Gaia.ROW_LIMIT = -1  # -1 is "unlimited." <-- be careful with that.
jj = Gaia.cone_search_async(coord, radius)
r = jj.get_results()
# "r" is a data structure of type "Table" from astropy.table

# Here are a few methods for dealing with a "Table"
# More information at https://docs.astropy.org/en/stable/table/
#r.pprint() # print bits of the table
#print(r.info) # list the column names (and more, in a pretty format)
#print(r.colnames)  # list the column names crudely

# Columns from a "Table" can be isolated into a 1-d array by name.
# Here are the useful ones for Gaia. These are case-sensitive.
#r['DESIGNATION']    # Gaia ID. note that lower case 'designation' will cause an error.
#r['ra']
#r['ra_error']
#r['dec']
#r['dec_error']
#r['parallax']
#r['parallax_error']
#r['pmra']
#r['pmra_error']
#r['pmdec']
#r['pmdec_error']
#r['phot_g_mean_mag']
#r['phot_g_mean_mag_error']
#r['phot_bp_mean_mag']
#r['phot_bp_mean_mag_error']
#r['phot_rp_mean_mag']
#r['phot_rp_mean_mag_error']
#r['radial_velocity']
#r['radial_velocity_error']

# for example, plot proper motions
#plot(r['pmra'],r['pmdec'],'b+')
#xlabel(r'$\mu_\alpha$')
#ylabel(r'$\mu_\delta$')
#show()

# Fit a multiGaussian
# explanation here: https://scikit-learn.org/stable/modules/mixture.html
# Prepare data array: (1) make a 2-d array and also (2) strip out NaNs
plx = r['parallax']
#data = np.stack((r['ra'][~np.isnan(plx)],r['dec'][~np.isnan(plx)],r['pmra'][~np.isnan(plx)],r['pmdec'][~np.isnan(plx)],1000.0/r['parallax'][~np.isnan(plx)]),axis=-1)
data = np.stack((r['pmra'][(~np.isnan(plx)) & (plx > 1)],r['pmdec'][(~np.isnan(plx))& (plx > 1)],1000.0/r['parallax'][(~np.isnan(plx))& (plx > 1)]),axis=-1)

data = np.array(data)  # convert to numpy array (apparently, this is NOT redundant)

# columns are:  ra, dec, pmra, pmdec, 1000/parallax

#print(data.shape)
#print(data)


# MultiGaussian setup:

# user guess as to where the cluster and field are centered
#clusterfield=np.array([[130.05,19.62,-36.4,-13.0,185.0],[130.05,19.62,-1,-2,300.0]])
clusterfield=np.array([[-36.4,-13.0,185.0],[-1,-2,300.0]])

#clusterfield=np.array([[130.05,130.05],[19.62,19.62],[-36.4,-1],[-13.0,-2],[185.0,300.0]])

gm = GaussianMixture(n_components=2,init_params='random_from_data',random_state=32,means_init=clusterfield,covariance_type='full',n_init=41).fit_predict(data)

# map the colors.  gm[] is full of zeros and ones. Let's make a gm2 with colors in it.
gm2 = []   # empty 'list' object
for i in range(len(gm)):
     if gm[i] == 0:
          gm2.append('xkcd:dark gray')
     else:
          gm2.append('xkcd:light purple')

fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(12, 6))
ax1.scatter(r['ra'][(~np.isnan(plx))& (plx > 1)],r['dec'][(~np.isnan(plx))& (plx > 1)], c=gm2)
ax1.set_title("RA - dec")
ax1.set_ylabel(r'Dec (deg)')
ax1.set_xlabel(r'R.A. (deg)')

ax2.scatter(data[:, 0], data[:, 1], c=gm2)
ax2.set_title("pmRA - pmDEC")
ax2.set_ylabel(r'pmdec (mas yr$^{-1}$)')
ax2.set_xlabel(r'pmRA (mas yr$^{-1}$)')

show()


fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(12, 6))
comp1=data[:,2][gm==0]
comp2=data[:,2][gm==1]
ax1.hist([comp1,comp2],bins=50,range=(0.0,400.0),stacked=True,color=['xkcd:dark gray','xkcd:light purple'])
ax1.set_title("Parallax dist (pc)")
ax1.set_xlabel('Distance (pc)')
ax1.set_ylabel('N')

color = r['phot_bp_mean_mag'][(~np.isnan(plx))& (plx > 1)] - r['phot_rp_mean_mag'][(~np.isnan(plx))& (plx > 1)]
mag = r['phot_rp_mean_mag'][(~np.isnan(plx))& (plx > 1)] #- 5.0*np.log10(1000.0/r['parallax'][(~np.isnan(plx))& (plx > 1)]) + 5.0
ax2.scatter(color,mag,c=gm2)
ax2.set_xlim([-2,5])
ax2.set_ylim([21.0,1.0])
ax2.set_xlabel('BP-RP')
ax2.set_ylabel('RP (mag)')

show()


# Random Forest

# recipe: https://stackabuse.com/random-forest-algorithm-with-python-and-scikit-learn/
# Random Forest setup:
SEED = 88   # a random number
rfc = RandomForestClassifier(n_estimators=5,max_depth=2,random_state=SEED)
# Train the RF with the first 100 of the results of the Gaussian Mixture model
data2 = np.copy(data[0:100,:]) ; y_train = np.copy(gm[0:100])
rfc.fit(data2,y_train)
#rfc.fit(data,gm)
# Predict using this training set
rfmem = rfc.predict(data)

# map the colors.  gm[] is full of zeros and ones. Let's make a gm2 with colors in it.
rfmem2 = []   # empty 'list' object
for i in range(len(rfmem)):
     if rfmem[i] == 0:
          rfmem2.append('xkcd:navy blue')
     else:
          rfmem2.append('xkcd:pale green')

fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(12, 6))
ax1.scatter(r['ra'][(~np.isnan(plx))& (plx > 1)],r['dec'][(~np.isnan(plx))& (plx > 1)], c=rfmem2)
ax1.set_title("RA - dec")
ax1.set_ylabel(r'Dec (deg)')
ax1.set_xlabel(r'R.A. (deg)')

ax2.scatter(data[:, 0], data[:, 1], c=rfmem2)
ax2.set_title("pmRA - pmDEC")
ax2.set_ylabel(r'pmdec (mas yr$^{-1}$)')
ax2.set_xlabel(r'pmRA (mas yr$^{-1}$)')

show()


fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(12, 6))
comp1=data[:,2][gm==0]
comp2=data[:,2][gm==1]
ax1.hist([comp1,comp2],bins=50,range=(0.0,400.0),stacked=True,color=['xkcd:navy blue','xkcd:pale green'])
ax1.set_title("Parallax dist (pc)")
ax1.set_xlabel('Distance (pc)')
ax1.set_ylabel('N')

color = r['phot_bp_mean_mag'][(~np.isnan(plx))& (plx > 1)] - r['phot_rp_mean_mag'][(~np.isnan(plx))& (plx > 1)]
mag = r['phot_rp_mean_mag'][(~np.isnan(plx))& (plx > 1)] #- 5.0*np.log10(1000.0/r['parallax'][(~np.isnan(plx))& (plx > 1)]) + 5.0
ax2.scatter(color,mag,c=rfmem2)
ax2.set_xlim([-2,5])
ax2.set_ylim([21.0,1.0])
ax2.set_xlabel('BP-RP')
ax2.set_ylabel('RP (mag)')

show()



print( 'normal stop' )
