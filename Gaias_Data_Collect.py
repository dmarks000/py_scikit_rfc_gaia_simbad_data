#import matplotlib.pyplot as plt
#import numpy as np
from pylab import *
#from sklearn.ensemble import RandomForestRegressor as rf

import astropy.units as u
import astropy.coordinates as a_coord
from astropy.table import Table as t
from astropy.coordinates import SkyCoord
from astroquery.gaia import Gaia
from astroquery.simbad import Simbad

# if error here, try 'pip install astroquery'
# perform a cone search and get the data
#skyfile = 'ruprecht37data.dat'; skyfile2 = 'ruprecht37data_sim.dat'; coord = SkyCoord(ra=117.465, dec=-17.225, unit=(u.degree, u.degree), frame='icrs') # Ruprecht 37
#skyfile = 'm44data.dat'; skyfile2 = 'm44data_sim.dat'; coord = SkyCoord(ra=130.054, dec=19.621, unit=(u.degree, u.degree), frame='icrs') # M44 = NGC 2632
#skyfile = 'ngc6791data.dat'; skyfile2 = 'ngc6791data_sim.dat'; coord = SkyCoord(ra=290.221, dec=37.778, unit=(u.degree, u.degree), frame='icrs') # NGC 6791
#note, for some reason, for ngc6791 specifically, pinging simbad for 'rot' results in an error. Can remove 'rot' from custom_query.add_votable_fields() to fix this.
#instead of polling gaia over and over, try to store the data into a file for quick reference
def return_table_with_data(skyfile, skyfile2,coord,mode=0) -> t:
    if (mode<=1):
        try:
            test_table = t.read(skyfile, format='ascii')
            r = test_table
            #success, file found, thus now we read!
            #note: when changing coords, it'd be desirable to change the stored data, thus DELETE gaiadata.dat
            #else you'll be working off of old coords :(
        except:
            radius = u.Quantity(0.75, u.deg)
            Gaia.MAIN_GAIA_TABLE = "gaiadr3.gaia_source"  # Reselect Data Release 3, 3 is NOT default?
            Gaia.ROW_LIMIT = -1  # -1 is "unlimited." <-- be careful with that.
            j = Gaia.cone_search_async(coord, radius)
            r = j.get_results() #Queries Gaia.
            r.write(skyfile, format='ascii',overwrite=True) #Writes the table to a file to be freely retrieved with t.read(filename, format='ascii')
    if(mode>=1):
        try:
            r_sim =  t.read(skyfile2,format='ascii')
            #same idea as before. Reads the data as a table object to this file. This is the simbad data table.
            # print("test :)")
        except:
            radius = u.Quantity(0.5, u.deg)
            #Gaia.ROW_LIMIT = -1  # -1 is "unlimited." <-- be careful with that.
            #j = Gaia.cone_search_async(coord, radius)
            custom_query = Simbad() #Simbad queries are special python objects. This initializes the object.
            #custom_query.remove_votable_fields('coordinates') #This removes the query data we don't need.
            custom_query.add_votable_fields('membership','ra','dec','parallax','parallax_error','pmra_error','pmdec_error','phot_g_mean_mag','phot_bp_mean_mag','phot_rp_mean_mag','pmra','pmdec','rvz_radvel','rv_value','rvz_qual') #This adds the query data we do need. Some more specification needed to figure out what else we want.
            r_sim = t(custom_query.query_region(coord,radius),masked=False)#THIS DOES THE QUERY. All of this was done to ensure we're making 1 query to Simbad, with ALL of the data we need.
            r_sim.write(skyfile2, format='ascii',overwrite=True, fast_writer=False)#Finally, this writes the query to the file system. This is important, as on subsequent run-throughs we won't be querying anything (Technically offline with our data.)
    try:
        if(mode==0):
            return r
        if(mode==1):
            return r_sim, r
        if(mode==2):
            return r_sim
    except:
        return NaN