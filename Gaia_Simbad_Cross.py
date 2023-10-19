import csv

#infile = 'Berkeley_18clean.csv'
infile = 'NGC_6253clean.csv'

# ID and probability (blank lists)
CID = []
mprob = []

with open(infile) as csvfile:
     dreader = csv.reader(csvfile)
     for row in dreader:
          CID.append(row[0])
          mprob.append(float(row[1]))


# temporary debug/confirmation
print(CID)
print(mprob)
print( 'number of stars = ',len(CID) )