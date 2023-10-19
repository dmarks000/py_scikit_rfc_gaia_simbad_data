import matplotlib.pyplot as plt
import numpy as np
from pylab import *
from sklearn.ensemble import RandomForestRegressor as rf
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_regression as mr
from astropy.table import Table as t
import astropy.units as u
from astropy.coordinates import Angle, SkyCoord
from Gaias_Data_Collect import *
from RadialFunsies import radial_funsies
from sklearn.mixture import GaussianMixture
from time import process_time
import pickle
import csv

test = 'ngc6253'

def r_print(mode,astr_table: t):
    if mode == 0:
        astr_table.pprint() # print bits of the table
    elif mode == 1:
        print(astr_table.info) # list the column names (and more, in a pretty format)
    elif mode == 2:
        print(astr_table.colnames) # list the column names crudely
    elif mode == 3:
        print(astr_table['MAIN_ID'])

def prepare_rf_list_inputoutput(r: t,r_sim: t):
    #the preparer
    r_sim_rv_append = []
    #get rid of the first/second row decor
    #Essentially, everything's gotta just be a number or --.
    skipme = []
    increasor = 0
    for i in r_sim['RV_VALUE']:
        if (i == 'RV_VALUE'):
            continue
        if (i == '--------'):
            continue
        if (type(i) != np.float64):
            skipme.append(increasor)
            increasor += 1
            continue
        r_sim_rv_append.append(i)
        increasor += 1

    SUUUUPER_REGRESSOR = []
    SUUUUPER_REGRESSOR.append(r_sim_rv_append)
    ra_to_deg_float = []
    increasor = 0
    for i in r_sim['RA']:
        if increasor in skipme:
            increasor += 1
            continue
        increasor += 1
        ra_to_deg_float.append(Angle(i,unit='deg').value)
    dec_to_deg_float = []
    increasor = 0
    for i in r_sim['DEC']:
        if increasor in skipme:
            increasor += 1
            continue
        increasor += 1
        dec_to_deg_float.append(Angle(i,unit='deg').value)
    SUUUUPER_REGRESSOR.append(ra_to_deg_float)
    SUUUUPER_REGRESSOR.append(dec_to_deg_float)
    #Ideally there'd be a command to convert this to a float so that we can just insert a list into the regressor.
    # SUUUUPER_REGRESSOR.append(r_sim['RA'])
    # SUUUUPER_REGRESSOR.append(r_sim['DEC'])
    # SUUUUPER_REGRESSOR.append(r_sim['FLUX_B'])
    # SUUUUPER_REGRESSOR.append(r_sim['FLUX_V'])
    # SUUUUPER_REGRESSOR.append(r_sim['FLUX_K'])
    # SUUUUPER_REGRESSOR.append(r_sim['SP_TYPE'])
    # SUUUUPER_REGRESSOR.append(r_sim['RVZ_RADVEL'])
    # SUUUUPER_REGRESSOR.append(r_sim['ROT_qual'])
    # SUUUUPER_REGRESSOR.append(r_sim['ROT_upVsini'])
    # SUUUUPER_REGRESSOR.append(r_sim['ROT_Vsini'])
    #print(SUUUUPER_REGRESSOR[0])
    result_list = []
    increasor = 0
    for i in r_sim["MEMBERSHIP"]:
        if increasor in skipme:
            increasor += 1
            continue
        increasor += 1
        result_list.append(i)
    indexer = 0
    input_list = []
    while indexer < len(result_list):
        index_list = []
        index_list.append(SUUUUPER_REGRESSOR[0][indexer])
        index_list.append(SUUUUPER_REGRESSOR[1][indexer])
        index_list.append(SUUUUPER_REGRESSOR[2][indexer])
        input_list.append(input_list)
        indexer += 1
    # print(input_list)
    # print(input_list.shape)
    # print(input_list[0].shape)
    # print(result_list.shape)
    # in_array = np.array(index_list)
    # out_array = np.array(result_list)
    # print(in_array.shape)
    # print(out_array.shape)
    return input_list, result_list

def run_rf_model(input,output,retry_model,skyfile3):
    t_model_fit_process_before = process_time()
    try:
        if retry_model == 0:
            raise TypeError("Model file exists, but retrying model anyways.")
        regr = pickle.load(skyfile3)
        # as it turns out, it takes a LONG time to train a model! Thus, this should help: load the previously saved training model.
        # As a additional note, THIS IS FOOLISH!!!!
        # This is because using pickle is pretty unsecure, though I'm using it for short term testing. In the long term, it might be far better
        # to test and switch to a format that sklearn recommends, like skops.io
        # https://scikit-learn.org/stable/model_persistence.html
        t_model_fit_process_load = process_time()
        print("Time (Fit Process Before):",t_model_fit_process_before,"(Fit Process Load):", t_model_fit_process_load)
    except:
        # Model doesn't exist, so now we try to make a new one instead.
        regr = rf(max_depth = 2, random_state = 42) #random_state = 42 is pretty crucial for some reason.
        # regr.fit(r['radial_velocity'],r['']) #uuuh does gaia have membership probabilities? need to use simbad to generate I guess
        regr.fit(input,output)
        pickle.dump(regr, skyfile3)
        t_model_fit_process_fit = process_time()
        print("Time (Fit Process Before):",t_model_fit_process_before,"(Fit Process Fit):", t_model_fit_process_fit)
    #at this point the machine is learning!
    return regr

def make_prepared_predicted(r: t):
    #building the data to detect- must be in the same order as above
    not_so_super_regressor = []
    not_so_super_regressor.append(r['radial_velocity'][2])
    not_so_super_regressor.append(Angle(r['ra'][0],unit='deg').value)
    not_so_super_regressor.append(Angle(r['dec'][0],unit='deg').value)
    # not_so_super_regressor.append(r['b'])
    # not_so_super_regressor.append(r['v'])
    return not_so_super_regressor

def old_function(r,r_sim,skyfile3):
    #radial_funsies(r,r_sim) uncomment this :)
    # now for random forest stuff!!!

    regr = run_rf_model(prepare_rf_list_inputoutput(r,r_sim),0,skyfile3)


    test_regress = regr.predict(make_prepared_predicted(r))
    print(test_regress)
    #r_print(2, r_sim)
    # t_model_fit_process_predict = process_time()
    t_stop = process_time()
    print("Normal Stop; Time (Total):", t_stop)

def gaussian(data: np.array,clusterfield,fig, axis,r,plx,t_para):
    GaussMix = GaussianMixture(n_components=2,init_params='random_from_data',random_state=26,means_init=clusterfield,covariance_type='full',n_init=60)
    gm = GaussMix.fit_predict(data)
    #gmtest = GaussMix.predict_proba(data)

    #map the colors.  gm[] is full of zeros and ones. Let's make a gm2 with colors in it.
    gm2 = []   # empty 'list' object
    for i in range(len(gm)):
        if gm[i] == 1:
            gm2.append('xkcd:dark gray')
        else:
            gm2.append('xkcd:light purple')
    axis[0,0]
    #ax1.scatter(r['ra'][(~np.isnan(plx)) & (plx > (1000/t_para * 0.5)) & (~np.isnan(rot)) & (rot > 1)],r['dec'][(~np.isnan(plx)) & (plx > (1000/t_para * 0.5)) & (~np.isnan(rot)) & (rot > 1)], c=gm2)
    axis[0,0].scatter(r['ra'][(~np.isnan(plx)) & (plx > (1000/t_para * 0.5))],r['dec'][(~np.isnan(plx)) & (plx > (1000/t_para * 0.5))], c=gm2)
    axis[0,0].set_title("RA - dec (Gaussian)")
    axis[0,0].set_ylabel(r'Dec (deg)')
    axis[0,0].set_xlabel(r'R.A. (deg)')

    axis[0,1].scatter(data[:, 0], data[:, 1], c=gm2)
    axis[0,1].set_title("pmRA - pmDEC (Gaussian)")
    axis[0,1].set_ylabel(r'pmdec (mas yr$^{-1}$)')
    axis[0,1].set_xlabel(r'pmRA (mas yr$^{-1}$)')
    #plt.figure()

    comp1=data[:,2][gm==0]
    comp2=data[:,2][gm==1]
    #print("test:", clusterfield[0][2]*2)
    axis[1,0].hist([comp1,comp2],bins=100,range=(0.0,clusterfield[0][2]*2),stacked=True,color=['xkcd:light purple','xkcd:dark gray'])
    axis[1,0].set_title("Parallax dist (pc) (Gaussian)")
    axis[1,0].set_xlabel('Distance (pc)')
    axis[1,0].set_ylabel('N')

    color = r['phot_bp_mean_mag'][(~np.isnan(plx)) & (plx > (1000/t_para * 0.5))] - r['phot_rp_mean_mag'][(~np.isnan(plx)) & (plx > (1000/t_para * 0.5))]
    mag = r['phot_rp_mean_mag'][(~np.isnan(plx)) & (plx > (1000/t_para * 0.5))] #- 5.0*np.log10(1000.0/r['parallax'][(~np.isnan(plx))& (plx > (1000/t_para * 0.5))]) + 5.0
    #color = r['phot_bp_mean_mag'][(~np.isnan(plx)) & (plx > (1000/t_para * 0.5)) & (~np.isnan(rot)) & (rot > 1)] - r['phot_rp_mean_mag'][(~np.isnan(plx)) & (plx > (1000/t_para * 0.5)) & (~np.isnan(rot)) & (rot > 1)]
    #mag = r['phot_rp_mean_mag'][(~np.isnan(plx)) & (plx > (1000/t_para * 0.5)) & (~np.isnan(rot)) & (rot > 1)] #- 5.0*np.log10(1000.0/r['parallax'][(~np.isnan(plx))& (plx > (1000/t_para * 0.5))]) + 5.0
    axis[1,1].scatter(color,mag,c=gm2)
    axis[1,1].set_xlim([-2,5])
    axis[1,1].set_ylim([21.0,1.0])
    axis[1,1].set_xlabel('BP-RP')
    axis[1,1].set_ylabel('RP (mag)')
    axis[1,1].set_title("(Gaussian)")
    plt.subplots_adjust(hspace=0.4,wspace=0.4)
    #plt.figure()
    t_gm = process_time()
    print("Time (Gaussian):", t_gm)
    return gm

def random_forest(data,gm,clusterfield,fig, axis,r,plx,t_para):
    # Random Forest

    # recipe: https://stackabuse.com/random-forest-algorithm-with-python-and-scikit-learn/
    # Random Forest setup:
    SEED = 88   # a random number
    rfc = RandomForestClassifier(n_estimators=15,max_depth=3,random_state=SEED)
    # Train the RF with the first 100 of the results of the Gaussian Mixture model
    data2 = np.copy(data[0:120,:]) ; y_train = np.copy(gm[0:120])
    rfc.fit(data2,y_train)
    #rfc.fit(data,gm)
    # Predict using this training set
    rfmem = rfc.predict(data)
    rfmemtest = rfc.predict_proba(data)

    # map the colors.  gm[] is full of zeros and ones. Let's make a gm2 with colors in it.
    rfmem2 = []   # empty 'list' object
    # for i in range(len(rfmem)):
    #      if rfmem[i] == 0:
    #           rfmem2.append('xkcd:navy blue')
    #      else:
    #           rfmem2.append('xkcd:pale green')
    rfmem2test = []
    snum = 0
    for i in range(len(rfmemtest)):
        #print(rfmemtest[i])
        if rfmemtest[i][snum] >= 0.90:
            rfmem2test.append('xkcd:aquamarine')
        elif rfmemtest[i][snum] >= 0.80:
            rfmem2test.append('xkcd:pale green')
        elif rfmemtest[i][snum] >= 0.70:
            rfmem2test.append('xkcd:pale yellow')
        elif rfmemtest[i][snum] >= 0.60:
            rfmem2test.append('xkcd:mustard yellow')
        elif rfmemtest[i][snum] >= 0.50:
            rfmem2test.append('xkcd:light red')
        elif rfmemtest[i][snum] >= 0.40:
            rfmem2test.append('xkcd:rust')
        elif rfmemtest[i][snum] >= 0.30:
            rfmem2test.append('xkcd:puce')
        elif rfmemtest[i][snum] >= 0.20:
            rfmem2test.append('xkcd:plum')
        elif rfmemtest[i][snum] >= 0.10:
            rfmem2test.append('xkcd:royal purple')
        else:
            rfmem2test.append('xkcd:navy blue')
    rfmem2 = rfmem2test

    fig, axis = plt.subplots(nrows=2, ncols=2)
    axis[0,0].scatter(r['ra'][(~np.isnan(plx)) & (plx > (1000/t_para * 0.5))],r['dec'][(~np.isnan(plx)) & (plx > (1000/t_para * 0.5))], c=rfmem2)
    #ax1.scatter(r['ra'][(~np.isnan(plx)) & (plx > (1000/t_para * 0.5)) & (~np.isnan(rot)) & (rot > 1)],r['dec'][(~np.isnan(plx)) & (plx > (1000/t_para * 0.5)) & (~np.isnan(rot)) & (rot > 1)], c=rfmem2)
    axis[0,0].set_title("RA - dec (Random Forest)")
    axis[0,0].set_ylabel(r'Dec (deg)')
    axis[0,0].set_xlabel(r'R.A. (deg)')

    axis[0,1].scatter(data[:, 0], data[:, 1], c=rfmem2)
    axis[0,1].set_title("pmRA - pmDEC (Random Forest)")
    axis[0,1].set_ylabel(r'pmdec (mas yr$^{-1}$)')
    axis[0,1].set_xlabel(r'pmRA (mas yr$^{-1}$)')

    #plt.figure()



    comp1=data[:,2][gm==0]
    comp2=data[:,2][gm==1]
    axis[1,0].hist([comp1,comp2],bins=100,range=(0.0,clusterfield[0][2]*2.2),stacked=True,color=['xkcd:aquamarine','xkcd:navy blue'])
    axis[1,0].set_title("Parallax dist (pc) (Random Forest)")
    axis[1,0].set_xlabel('Distance (pc)')
    axis[1,0].set_ylabel('N')

    #plt.figure()

    color = r['phot_bp_mean_mag'][(~np.isnan(plx)) & (plx > (1000/t_para * 0.5))] - r['phot_rp_mean_mag'][(~np.isnan(plx)) & (plx > (1000/t_para * 0.5))]
    mag = r['phot_rp_mean_mag'][(~np.isnan(plx)) & (plx > (1000/t_para * 0.5))] #- 5.0*np.log10(1000.0/r['parallax'][(~np.isnan(plx))& (plx > (1000/t_para * 0.5))]) + 5.0
    #color = r['phot_bp_mean_mag'][(~np.isnan(plx)) & (plx > (1000/t_para * 0.5)) & (~np.isnan(rot)) & (rot > 1)] - r['phot_rp_mean_mag'][(~np.isnan(plx)) & (plx > (1000/t_para * 0.5)) & (~np.isnan(rot)) & (rot > 1)]
    #mag = r['phot_rp_mean_mag'][(~np.isnan(plx)) & (plx > (1000/t_para * 0.5)) & (~np.isnan(rot)) & (rot > 1)] #- 5.0*np.log10(1000.0/r['parallax'][(~np.isnan(plx))& (plx > (1000/t_para * 0.5))]) + 5.0
    axis[1,1].scatter(color,mag,c=rfmem2)
    axis[1,1].set_xlim([-2,5])
    axis[1,1].set_ylim([21.0,1.0])
    axis[1,1].set_xlabel('BP-RP')
    axis[1,1].set_ylabel('RP (mag)')
    axis[1,1].set_title('(Random Forest)')
    plt.subplots_adjust(hspace=0.4,wspace=0.4)

    ###########
    #Want: List of 'cold' stars with probabilities.

    # resultant_List = rfmemtest
    # ids = r['DESIGNATION']
    # Nice_And_Neat_Table = []
    # test_ids = r['DESIGNATION'][(~np.isnan(plx)) & (plx > (1000/t_para * 0.5))]
    # for i in range(len(test_ids)):
    #     #if i == 0:
    #     #    continue
    #     # if(r['parallax'][i]):
    #     #     continue
    #     if(mag[i] >= 15):
    #         continue
    #     if(color[i] <= 1):
    #         continue
    #     templist = []
    #     templist.append(test_ids[i])
    #     templist.append(resultant_List[i])
    #     Nice_And_Neat_Table.append(templist)
    #print(Nice_And_Neat_Table)
    return rfmemtest

def main(whatculookinat: string_):
    #skyfile = 'ngc6791data.dat'; skyfile2 = 'ngc6791data_sim.dat'; skyfile3 = 'ngc6791modeldata.dat' ; coord = SkyCoord(ra=290.221, dec=37.778, unit=(u.degree, u.degree), frame='icrs') # NGC 6791
    #skyfile = 'ngc2324data.dat'; skyfile2 = 'ngc2324data_sim.dat'; skyfile3 = 'ngc2324modeldata.dat' ; coord = SkyCoord("07h04m07.9s +01 02 46.0", unit=(u.hourangle,u.deg), frame='icrs') # NGC 2324
    #skyfile = '47tucdata.dat'; skyfile2 = '47tucdata_sim.dat'; skyfile3 = '47tucmodeldata.dat' ; coord = SkyCoord("00h24m05.359s -72 04 53.20", unit=(u.hourangle,u.deg), frame='icrs') # NGC 104/47 Tuc
    #skyfile = 'pal6data.dat'; skyfile2 = 'pal6data_sim.dat'; skyfile3 = 'pal6modeldata.dat' ; coord = SkyCoord("17h43m42.20s -26 13 21.0", unit=(u.hourangle,u.deg), frame='icrs') # Pal 6
    #skyfile = 'berkeley18data.dat'; skyfile2 = 'berkeley18data_sim.dat'; skyfile3 = 'berkeley18modeldata.dat' ; coord = SkyCoord("05h22m07.4s +45 26 31", unit=(u.hourangle,u.deg), frame='icrs') # berkeley 18
    #skyfile = 'ngc2420data.dat'; skyfile2 = 'ngc2420data_sim.dat'; skyfile3 = 'ngc2420modeldata.dat' ; coord = SkyCoord("07h38m24.5s +21 34 30", unit=(u.hourangle,u.deg), frame='icrs') # NGC 2420
    sky_list = []
    sky_list.append(['ngc6253',[SkyCoord("16h59m06.7s -52 42 43", unit=(u.hourangle,u.deg), frame='icrs'),1776,np.array([[-4.537,-5.280,1776],[-8,8,1300]])]]) # NGC 6253)
    sky_list.append(['ruprecht37',[SkyCoord(ra=117.465, dec=-17.225, unit=(u.degree, u.degree), frame='icrs'),5128,np.array([[-1.700,2.417,5128],[-40,0,1000]])]]) # ruprecht37)
    sky_list.append(['m67',[SkyCoord("08h51m23.0s +11 48 50", unit=(u.hourangle,u.deg), frame='icrs'),875.0,np.array([[-10.97,-2.9396,875.0],[-1,-2,500.0]])]]) # m67)
    sky_list.append(['m44',[SkyCoord(ra=130.054, dec=19.621, unit=(u.degree, u.degree), frame='icrs'),185.0,np.array([[-36.4,-13.0,185.0],[-1,-2,300.0]])]]) # m44)

    skyfile = ''
    skyfile2 = ''
    skyfile3 = ''
    coord = SkyCoord
    #more manual stuff:
    #pmra pmdec parallax (parsecs)
    clusterfield = np.array

    
    skyfile = whatculookinat + 'data.dat'
    skyfile2 = whatculookinat + 'data_sim.dat'
    skyfile3 = whatculookinat + 'modeldata.dat'
    for i in range(len(sky_list)):
        if (sky_list[i][0] == whatculookinat):
            coord = sky_list[i][1][0]
            t_para = sky_list[i][1][1]
            clusterfield = sky_list[i][1][2]
            break

    t_start = process_time()
    r = return_table_with_data(skyfile=skyfile,skyfile2=skyfile2,coord=coord,mode=0)
    t_table_return = process_time()
    print("Time (startup):",t_start,"(Table Return):", t_table_return)
    #old_function(r,r_sim)

    plx = r['parallax']
    #rot = r['radial_velocity'] #test using radial velocity
    #data = np.stack((r['ra'][~np.isnan(plx)],r['dec'][~np.isnan(plx)],r['pmra'][~np.isnan(plx)],r['pmdec'][~np.isnan(plx)],1000.0/r['parallax'][~np.isnan(plx)]),axis=-1)
    #data = np.stack((r['pmra'][(~np.isnan(plx)) & (plx > (1000/t_para * 0.5)) & (~np.isnan(rot)) & (rot > 1)],r['pmdec'][(~np.isnan(plx)) & (plx > (1000/t_para * 0.5)) & (~np.isnan(rot)) & (rot > 1)],1000.0/r['parallax'][(~np.isnan(plx)) & (plx > (1000/t_para * 0.5)) & (~np.isnan(rot)) & (rot > 1)],r['radial_velocity'][(~np.isnan(plx)) & (plx > (1000/t_para * 0.5)) & (~np.isnan(rot)) & (rot > 1)]),axis=-1)
    data = np.stack((r['pmra'][(~np.isnan(plx)) & (plx > (1000/t_para * 0.5))],r['pmdec'][(~np.isnan(plx)) & (plx > (1000/t_para * 0.5))],1000.0/r['parallax'][(~np.isnan(plx)) & (plx > (1000/t_para * 0.5))]),axis=-1)
    data = np.array(data)  # convert to numpy array (apparently, this is NOT redundant)

    fig, axis = plt.subplots(nrows=2, ncols=2)
    # maxnum=0
    # sumnum = 0
    # cntnum = 0
    # plx = r['parallax']
    # for i in plx:
    #     if(np.isreal(i)):
    #         maxnum=max(1000.0/i,maxnum)
    #         sumnum += 1000.0/i
    #         cntnum += 1
    # avgnum = sumnum/cntnum
    # print("max: ",maxnum,"avg: ",avgnum, "cnt:", cntnum)
    print("Time (Data):", process_time())
    gm = gaussian(data,clusterfield,fig, axis,r,plx,t_para)
    rf = random_forest(data,gm,clusterfield,fig, axis,r,plx,t_para)
    print("Time (Training):", process_time())
    
    # # infile = 'Berkeley_18clean.csv'
    # infile = 'NGC_6253clean.csv'

    # # ID and probability (blank lists)
    # CID = []
    # mprob = []

    # with open(infile) as csvfile:
    #      dreader = csv.reader(csvfile)
    #      for row in dreader:
    #           CID.append(row[0])
    #           mprob.append(float(row[1]))
    

    # # temporary debug/confirmation
    # # print(CID)
    # # print(mprob)
    # # print( 'number of stars = ',len(CID) )
    # rfprep = []
    # referencelist = []
    # design = r['DESIGNATION']
    # for i in design:
    #     if np.isnan(plx[design.index_column(i)]):
    #         continue
    #     if plx[design.index_column(i)] > (1000/t_para * 0.5):
    #         continue
    #     referencelist += i

    # rfprep = [] * len(referencelist)
    # for i in range(len(CID)):
    #     #i is now a given Gaia ID
    #     if(CID[i] in referencelist): #this is very slow. Replace for speed!
    #         testlist = []
    #         testlist.append(CID[i])
    #         testlist.append(mprob[i])
    #         rfprep[referencelist.index(CID[i])] = testlist
    # #rf prep is now an ordered list
    
    # overall_Table = []
    # for i in range(len(rfprep)):
    #     testlist = []
    #     testlist.append(rfprep[i][0])
    #     testlist.append(rfprep[i][1])
    #     testlist.append(gm[i][0])
    #     testlist.append(gm[i][1])
    #     testlist.append(rf[i][0])
    #     testlist.append(rf[i][1])
    #     overall_Table.append()
    # from tabulate import tabulate
    # with open(whatculookinat + 'output_table.dat', 'w') as f:
    #     f.write(tabulate(overall_Table))
    
    print("Normal Stop; Time (Total):", process_time())
    show()

if __name__ == "__main__":
    main(test)