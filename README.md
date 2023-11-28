# py_scikit_rfc_gaia_simbad_data
Utilizing sci-kit-learn's random forest classifier to catagorize membership probabilities of star clusters, via gaia/simbad data.

#Script explanation:


Gaia_Data_Analysis.py:

This is a initial version of utilizing the random forest method. It's fairly early, and is mostly self-contained (utilizes Gaia_Data_Collect), and puts everything through analysis.

Gaia_Simbad_Cross.py:

Takes a csv file of SIMBAD probabilties, lists the number of stars and does a list of their ID's and probabilities. (The latter two are in order.)

Gaia_Data_Collect.py:

This is a data collection script that obtains data from GAIA and/or SIMBAD. By default, it always creates/loads a file to store the obtained data. Takes in a couple parameters.
'skyfile' is the GAIA data file.
'skyfile2' is the SIMBAD data file.
'coord' is a coordinate of SkyCoord data type.