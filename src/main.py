import subprocess
from GenerateCSV import analytics
import pandas as pd

#####################################################################################################################################
#################### YOU ARE EXPECTED TO PREPROCESS THE DATA TO BE AN NP.NDARRAY BY USING .VALUES AS SHOWN BELOW ####################
############################### ALSO SPECIFY THE COLUMN WHICH IS THE TARGET VARIABLE IN THE CODE BELOW ##############################
################################### IF THE TARGET VARIABLE IS CATEGORICAL, FACTORISE IT #############################################
#####################################################################################################################################

def main():
    data = pd.read_csv('data/raw/iris.data', header=None)
    X = data.drop(data.columns[4], axis=1).values                           
    y = pd.factorize(data[4])[0]

    sklearnDTCdf = analytics(X, y)
    ownDTCdf = analytics(X, y, 'own')

    sklearnDTCdf.to_csv('results/LibraryDecisionTreeClassifier_results.csv', index=False)
    ownDTCdf.to_csv('results/OwnDecisionTreeClassifier_results.csv', index=False)

#####################################################################################################################################

if __name__ == '__main__':
    main()