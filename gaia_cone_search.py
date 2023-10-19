import matplotlib.pyplot as plt
import numpy as np
from pylab import *

import astropy.units as u
from astropy.coordinates import SkyCoord
from astroquery.gaia import Gaia

# Uses Astroquery to do a Gaia DR3 cone search
# User should edit "coord", "radius", and "outputfilename" for each cluster.

# perform a cone search and get the data
#coord = SkyCoord(ra=117.465, dec=-17.225, unit=(u.degree, u.degree), frame='icrs') # Ruprecht 37
#coord = SkyCoord(ra=130.054, dec=19.621, unit=(u.degree, u.degree), frame='icrs') # M44 = NGC 2632
#coord = SkyCoord(ra=290.221, dec=37.778, unit=(u.degree, u.degree), frame='icrs') # NGC 6791
#coord = SkyCoord(ra=80.5310, dec =+45.4420, unit=(u.degree, u.degree), frame='icrs') #Be18
#coord = SkyCoord(ra=295.3270, dec =+40.1900, unit=(u.degree, u.degree), frame='icrs') #NGC 6819
#coord = SkyCoord(ra=254.7780,dec=-52.7120,unit=(u.degree, u.degree), frame='icrs') #NGC 6253
#coord = SkyCoord("07h38m24.5s +21 34 30", unit=(u.hourangle,u.deg), frame='icrs') #NGC 2420
coord = SkyCoord("01h46m20.6s +61 12 43", unit=(u.hourangle,u.deg), frame='icrs') #NGC 663
outputfilename = "n663_cone.csv"
radius = u.Quantity(0.25, u.deg)

# OK, user can fall asleep, now. Data should be retrieved, then written to output file in csv format.

Gaia.MAIN_GAIA_TABLE = "gaiadr3.gaia_source"  # Reselect Data Release 3, 3 is NOT default???
Gaia.ROW_LIMIT = -1  # -1 is "unlimited." <-- be careful with that.
j = Gaia.cone_search_async(coord, radius)
r = j.get_results()
# "r" is a data structure of type "Table" from astropy.table

# Here are some methods for dealing with a "Table"
# More information at https://docs.astropy.org/en/stable/table/

#r.pprint() # print bits of the table
print(r.info) # list the column names (and more, in a pretty format)
#print(r.colnames)  # list the column names crudely

# Extract a column into a 1-d array by name:
name = r['DESIGNATION']
ra = r['ra']
#r['ra_error']
dec = r['dec']
#r['dec_error']
plx = r['parallax']
plxe = r['parallax_error']
pmra = r['pmra']
pmrae = r['pmra_error']
pmdec = r['pmdec']
pmdece = r['pmdec_error']
g = r['phot_g_mean_mag']
#ge = r['phot_g_mean_mag_error']
bp = r['phot_bp_mean_mag']
#bpe = r['phot_bp_mean_mag_error']
rp = r['phot_rp_mean_mag']
#rpe = r['phot_rp_mean_mag_error']
bprp = r['bp_rp']   # BP - RP color, as observed (no dust correction)
# things likely to not exist for every star:
rv = r['radial_velocity']  # km/s
teff= r['teff_gspphot'] # Teff in K
Ag = r['ag_gspphot']  # extinction in G band
Ebr = r['ebpminrp_gspphot']  # color excess E(BP-RP)
vflag = r['phot_variable_flag'] # integer. Probably equals one if Gaia thinks the star is variable

print('Found ',len(pmra),' stars.')

# plot proper motions
#plot(r['pmra'],r['pmdec'],'b+')
#xlabel(r'$\mu_\alpha$')
#ylabel(r'$\mu_\delta$')
#show()

# filter for NaN (IEEE Not A Number) in parallax (no use in saving these data)
nanfilt = isnan(plx)

# write cone search results to a file
import csv
with open(outputfilename,'w',newline='') as csvfile:
     cw = csv.writer(csvfile, quoting=csv.QUOTE_MINIMAL)
     cw.writerow(['Gaia ID','RA','DEC','parallax','error in plx','PM(RA)','error in pmra','PM(dec)','error in pmdec','g magnitude','BP','RP','BP-RP','rv','Teff','Ag extinction','E(BP-RP)','variability flag'])
     for i in range(len(ra)):
          if nanfilt[i] == False:
               cw.writerow([name[i],ra[i],dec[i],plx[i],plxe[i],pmra[i],pmrae[i],pmdec[i],pmdece[i],g[i],bp[i],rp[i],bprp[i],rv[i],teff[i],Ag[i],Ebr[i],vflag[i]])

print( 'normal stop' )
