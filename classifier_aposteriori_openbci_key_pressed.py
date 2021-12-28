"""############################################"""
""" Classify incoming EEG signals in real-time """
"""############################################"""



###############
### Imports ###
###############

import numpy as np
import random
import csv
import scipy.signal
import math
import time
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
import pandas as pd
import helpers as hp
import sys
import genetic as gen
import argparse
import time
import pandas as pd
import brainflow
from brainflow.board_shim import BoardShim, BrainFlowInputParams, LogLevels, BoardIds
from brainflow.data_filter import DataFilter, FilterTypes, AggOperations
import firebasemonitoring as fbm
from imblearn.over_sampling import SMOTE
from sklearn.feature_selection import VarianceThreshold
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
sm = SMOTE()



##################################
### Measurement specifications ###
##################################

dino_frame_counter = 0
production_time = 10 # in seconds
samplingRate = 125
sensor_count = 32



########################
### Global variables ###
########################

clf=RandomForestClassifier(n_estimators=10)
X_indices = []
X_indices_real = []
data_saved = []

GLOBAL_CORR_LIMIT = 0.2



##############################################
### Model calibration after initialization ###
##############################################

def calibrateModel(data_raw, events_raw, doc):

    if not fbm.check_initialization_status():
        return 0,0

    fbm.setTrainingStatusOn()


    data, events = hp.removeArtifacts(data_raw, events_raw)


    c = list(zip(data, events))
    random.shuffle(c)

    data_raw, events_raw = zip(*c)



    hp.monitor("data_raw", data_raw)

    data_raw_transposed = []

    for i in data_raw:
        data_raw_transposed.append(np.transpose(i))
    
    data_raw = data_raw_transposed

    hp.monitor("data_raw_transposed", data_raw_transposed)
    
    # Load global variables
    global clf
    global X_indices
    global data_saved
    global X_indices_real
    global GLOBAL_CORR_LIMIT
    


    data, events = hp.removeArtifacts(data_raw, events_raw)
    
    # Get Extreme Points
    mini, maxi = hp.extremePointsCorrelation(data, events, 10)
    
    # Get frequency bands
    cores_real_numbers = hp.getFrequenciesPredefined(data)
    
    X_reduced_res = hp.concatenateFeatures(cores_real_numbers, mini, maxi)
    fbm.monitor_value(doc, "len_extracted_features", len(X_reduced_res[0]))

    

    df = []
    for row in X_reduced_res:
        df.append(row)


    dfn = pd.DataFrame(df)
        
    filename = str(time.time())
    dfn.to_csv('data/Patients/' + filename + '.csv',sep=',', float_format='%.15f') # write with precision .15




    
    print("Starting genetic algorithm")

    # Manually transpose array, because np.tranpose does not work here (why ever)...
    c = 00
    X_reduced_res_transposed = []
    len_row_zero = 0
    events_ref = []

    while c < len(X_reduced_res[0]):
        X_reduced_res_transposed_row = []
        for row in X_reduced_res:
            i = 0
            for val in row:
                if i == c:
                    X_reduced_res_transposed_row.append(val)
                i += 1

        if c == 0: # in some cases the row contains too few values. If so, this must be detected to prevent training from crashing.
            len_row_zero = len(X_reduced_res_transposed_row)

        if len(X_reduced_res_transposed_row) == len_row_zero:
            X_reduced_res_transposed.append(X_reduced_res_transposed_row)
            #events_ref.append(events[c])

        c += 1



    #sel = VarianceThreshold(threshold=(.9 * (1 - .9)))
    #X_reduced_res_transposed_high_variance = sel.fit_transform(X_reduced_res_transposed)

    #print('HUHUHUHUHUHU', len(X_reduced_res_transposed_high_variance[0]))

    #selector = SelectKBest(chi2, k=20)

    #X_indices_real = selector.fit_transform(np.transpose(X_reduced_res_transposed_high_variance), events)
    #cols = selector.get_support(indices=True)

    #print('HUHUHUHUHUHU', cols)
    X_indices_real = gen.simulateEvolution(X_reduced_res_transposed, 70, 220, 80, 20, 0.5, (len(events) - 25), 250, events, doc)
    
    print("Ended genetic algorithm")
    print(X_indices_real)
    
    X_reduced_res_real = []
    
    for row in X_reduced_res:
        res_row = []
        c = 0
        for i in row:
            if c in X_indices_real:
                res_row.append(i)

            c += 1
        X_reduced_res_real.append(res_row)
    
    # ------------------------------------------------------------------------------------------------------------------------------------- #
    
    X_indices = X_indices_real

    X_train, target = hp.generateTrainingSet(X_reduced_res_real, events)
    
    print("X Train: ", X_train)
    print("Events", events, target)
    print("X indices: ", X_indices)
    print("step 7 / 7")

    #hp.write([X_train, target], ["Input_data_edited", "targets"], "JLS")

    #print('jojojojojojojojdsfogfdjodfdjfgodfjdfogdfjfdogdjgodgjfdo', len(X_train), len(target), len(X_reduced_res_real))
    #print('pampa', len(X_indices_real[0]))

    #X_train, target = sm.fit_resample(X_train, target)



    #hp.monitor("Ufgepassiti2", X_train)

    cols = []
    for i in X_train[0]:
        cols.append(i)


    #df = pd.DataFrame(X_train, columns=cols)
    df = X_train
    df_target = pd.DataFrame(target)
    #print("Guck doch mal", df)


    #print("SHAPEjgh", df.shape)

    #df.to_csv('jls04.csv',sep=',', float_format='%.15f') # write with precision .15
    #df_target.to_csv('estc1.csv',sep=',', float_format='%.15f') # write with precision .15


    X_train_new = X_train
    X_test_new = X_train[78:]

    target_new = target
    target_test_new = target[78:]


    clf.fit(X_train_new, target_new)
    #print("Accuracy:",metrics.accuracy_score(clf.predict(X_test_new), target_test_new))
    print("Predictions: ", clf.predict(X_test_new))
    print("X_INDICES aotl", X_indices)

    return clf, X_indices












##############################################
### Model calibration after initialization ###
##############################################

def calibrateModel_adv(data_raw, events_raw, doc):

    if not fbm.check_advancedtraining_status():
        return 0,0

    fbm.setTrainingStatusOn()

    hp.monitor("data_raw", data_raw)

    data_raw_transposed = []

    for i in data_raw:
        data_raw_transposed.append(np.transpose(i))
    
    data_raw = data_raw_transposed

    hp.monitor("data_raw_transposed", data_raw_transposed)
    
    # Load global variables
    global clf
    global X_indices
    global data_saved
    global X_indices_real
    global GLOBAL_CORR_LIMIT
    


    data, events = hp.removeArtifacts(data_raw, events_raw)


    c = list(zip(data, events))
    random.shuffle(c)

    data, events = zip(*c)


    
    # Get Extreme Points
    mini, maxi = hp.extremePointsCorrelation(data, events, 10)
    
    # Get frequency bands
    cores_real_numbers = hp.getFrequenciesPredefined(data)
    
    X_reduced_res = hp.concatenateFeatures(cores_real_numbers, mini, maxi)
    fbm.monitor_value(doc, "len_extracted_features", len(X_reduced_res[0]))

    

    df = []
    for row in X_reduced_res:
        df.append(row)


    dfn = pd.DataFrame(df)
        
    filename = str(time.time())
    dfn.to_csv('data/Patients/' + filename + '_adv.csv',sep=',', float_format='%.15f') # write with precision .15




    
    print("Starting genetic algorithm")

    # Manually transpose array, because np.tranpose does not work here (why ever)...
    c = 00
    X_reduced_res_transposed = []
    len_row_zero = 0
    events_ref = []

    while c < len(X_reduced_res[0]):
        X_reduced_res_transposed_row = []
        for row in X_reduced_res:
            i = 0
            for val in row:
                if i == c:
                    X_reduced_res_transposed_row.append(val)
                i += 1

        if c == 0: # in some cases the row contains too few values. If so, this must be detected to prevent training from crashing.
            len_row_zero = len(X_reduced_res_transposed_row)

        if len(X_reduced_res_transposed_row) == len_row_zero:
            X_reduced_res_transposed.append(X_reduced_res_transposed_row)
            #events_ref.append(events[c])

        c += 1



    #sel = VarianceThreshold(threshold=(.9 * (1 - .9)))
    #X_reduced_res_transposed_high_variance = sel.fit_transform(X_reduced_res_transposed)

    #print('HUHUHUHUHUHU', len(X_reduced_res_transposed_high_variance[0]))

    #selector = SelectKBest(chi2, k=20)

    #X_indices_real = selector.fit_transform(np.transpose(X_reduced_res_transposed_high_variance), events)
    #cols = selector.get_support(indices=True)

    #print('HUHUHUHUHUHU', cols)
    X_indices_real = gen.simulateEvolution(X_reduced_res_transposed, 50, 192, 60, 20, 0.5, (len(events) - 1), 250, events, doc)
    
    print("Ended genetic algorithm")
    print(X_indices_real)
    
    X_reduced_res_real = []
    
    for row in X_reduced_res:
        res_row = []
        c = 0
        for i in row:
            if c in X_indices_real:
                res_row.append(i)
            c += 1
        X_reduced_res_real.append(res_row)
    
    # ------------------------------------------------------------------------------------------------------------------------------------- #
    
    X_indices = X_indices_real

    X_train, target = hp.generateTrainingSet(X_reduced_res_real, events)
    print("X Train: ", X_train)
    print("Events", events, target)
    print("X indices: ", X_indices)
    print("step 7 / 7")

    hp.write([X_train, target], ["Input_data_edited", "targets"], "JLS")

    print('jojojojojojojojdsfogfdjodfdjfgodfjdfogdfjfdogdjgodgjfdo', len(X_train), len(target), len(X_reduced_res_real))
    #print('pampa', len(X_indices_real[0]))

    #X_train, target = sm.fit_resample(X_train, target)



    #hp.monitor("Ufgepassiti2", X_train)
    print("Guck doch mal -1", X_indices_real[0])
    print("Guck doch mal 0", X_train[2])

    cols = []
    for i in X_train[0]:
        cols.append(i)

    #df = pd.DataFrame(X_train, columns=cols)
    df = X_train
    df_target = pd.DataFrame(target)
    #print("Guck doch mal", df)

    #print("SHAPEjgh", df.shape)

    df.to_csv('data/Patients/' + filename + '_reduced_adv.csv',sep=',', float_format='%.15f') # write with precision .15
    #df.to_csv('jls04.csv',sep=',', float_format='%.15f') # write with precision .15
    #df_target.to_csv('estc1.csv',sep=',', float_format='%.15f') # write with precision .15


    clf.fit(X_train, target)
    print("Accuracy:",metrics.accuracy_score(clf.predict(X_train), target))
    print("Predictions: ", clf.predict(X_train))

    return clf, X_indices


















##############################
### Real time predictions ###
##############################

def get_input_data(data_raw, X_indices):

    data_raw_transposed = []

    for i in data_raw:
        data_raw_transposed.append(np.transpose(i))

    data = data_raw_transposed # Data already split...

    hp.monitor("Ufgepassiti", data)

    print("ge_input_data step 1")
    # Get Extreme Points
    mini, maxi = hp.extremePointsCorrelationMain(data, 10)
    print("ge_input_data step 2")
    # Get frequency bands
    cores_real_numbers = hp.getFrequenciesPredefined(data)
    print("ge_input_data step 3")
    hp.monitor("mini", mini)
    hp.monitor("cores_real_numbers", cores_real_numbers)
    # Concatenate both lists to one prediction list
    X_predict = hp.concatenateFeaturesMain(np.transpose(cores_real_numbers), mini, maxi, X_indices)
    print("ge_input_data step 4")

    return X_predict



######################
### Data Streaming ###
######################

def calibrate(board, events_initial, interval_time, doc):

    # Monitor length of training events
    #fbm.monitor_value(doc, "training_events", len(events_initial))

    # Define local variables
    init_count = 0
    input_data = []
    input_data_latest = []
    datapoints_per_interval_already_monitored = False
    input_length_limit = 124

    i_stepper = 4



    ##########################
    ### Handle data stream ###
    ##########################

    for i in range(0, len(events_initial) + production_time):
        
        if init_count < len(events_initial): # If we are in initialization phase
            
            if events_initial[init_count] == 1:
                print("YES")
            else:
                print("No")

            init_count += 1
                
        else: # If we are in production phase
            #df = pd.DataFrame(map(float(input_data)))
            #df.to_csv('file2.csv', index=False, header=False)
            
            input_data_formatted = []
            hp.monitor("Ufgepassiti", input_data)

            for row in input_data:
                latest_sensor_length = -10
                false_sensor = False
                for sensor in row:
                    if latest_sensor_length != -10:
                        if len(sensor) != latest_sensor_length:
                            false_sensor = True
                            #print("Ich mache jetzt was")
                        latest_sensor_length = len(sensor)
                
                if not false_sensor:
                    input_data_formatted.append(row)

            return calibrateModel(input_data, events_initial, doc) # Train ML model with data from initilization

        time.sleep(interval_time)
        input_data_latest = board.get_board_data()

        combined_input_sensor = []
        for sensor in input_data_latest:
            c = 0
            combined_input_row = []
            for val in sensor:
                if c < input_length_limit:
                    combined_input_row.append(val)
                c += 1
            combined_input_sensor.append(combined_input_row)
        
        input_data.append(combined_input_sensor)
        print("PRINZ", len(input_data_latest[0]))
        

        #hp.monitor("input_data raw", input_data)
        if not datapoints_per_interval_already_monitored:
            fbm.monitor_value(doc, "datapoints_per_interval", len(input_data_latest[0]))
            datapoints_per_interval_already_monitored = True

        i_stepper += 1
        if i_stepper == 4:
            if not fbm.check_initialization_status():
                return 0,0
            i_stepper = 0

    #DataFilter.write_file(input_data, 'openbci_test.csv', 'w')  # use 'a' for append mode










def calibrate_adv(board, events_initial, interval_time, doc):

    # Monitor length of training events
    #fbm.monitor_value(doc, "training_events", len(events_initial))

    # Define local variables
    init_count = 0
    input_data = []
    input_data_latest = []
    datapoints_per_interval_already_monitored = False
    input_length_limit = 124

    i_stepper = 4



    ##########################
    ### Handle data stream ###
    ##########################

    for i in range(0, len(events_initial) + production_time):
        
        if init_count < len(events_initial): # If we are in initialization phase
            
            if events_initial[init_count] == 1:
                print("YES")
            else:
                print("No")

            init_count += 1
                
        else: # If we are in production phase
            #df = pd.DataFrame(map(float(input_data)))
            #df.to_csv('file2.csv', index=False, header=False)
            
            input_data_formatted = []
            hp.monitor("Ufgepassiti", input_data)

            for row in input_data:
                latest_sensor_length = -10
                false_sensor = False
                for sensor in row:
                    if latest_sensor_length != -10:
                        if len(sensor) != latest_sensor_length:
                            false_sensor = True
                            #print("Ich mache jetzt was")
                        latest_sensor_length = len(sensor)
                
                if not false_sensor:
                    input_data_formatted.append(row)

            return calibrateModel_adv(input_data, events_initial, doc) # Train ML model with data from initilization

        time.sleep(interval_time)
        input_data_latest = board.get_board_data()

        combined_input_sensor = []
        for sensor in input_data_latest:
            combined_input_row = []
            c = 0
            for val in sensor:
                if c < input_length_limit:
                    combined_input_row.append(val)
                c += 1
            combined_input_sensor.append(combined_input_row)
        
        input_data.append(combined_input_sensor)
        print("PRINZ", len(input_data_latest[0]))

        #hp.monitor("input_data raw", input_data)
        if not datapoints_per_interval_already_monitored:
            fbm.monitor_value(doc, "datapoints_per_interval", len(input_data_latest[0]))
            datapoints_per_interval_already_monitored = True

        i_stepper += 1
        if i_stepper == 4:
            if not fbm.check_initialization_status():
                return 0,0
            i_stepper = 0

    DataFilter.write_file(input_data, 'openbci_test.csv', 'w')  # use 'a' for append mode












       

# sudo python3 classifier_aposteriori_openbci.py --serial-port /dev/cu.usbserial-DM03H7U4 --board-id 2