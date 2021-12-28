"""#############################"""
""" Neurogram EEG Questionnaire """
"""#############################"""



###############
### Imports ###
###############

import classifier_aposteriori_openbci_key_pressed as cl
import argparse
import helpers as hp
import brainflow
from brainflow.board_shim import BoardShim, BrainFlowInputParams, LogLevels, BoardIds
from brainflow.data_filter import DataFilter, FilterTypes, AggOperations
import time
import firebasemonitoring as fbm
import keyboard
from multiprocessing import Process
import atexit
import joblib
import random
import pickle
import pandas as pd



######################################
### Application specific variables ###
######################################

interval_time = 1
#doc = fbm.init("dino_statistics", str(time.time()))
doc = fbm.init("dino_application", "Test_Four")
#events = [0,1,0,1,0,1,0,0,0,0,1,1,1,1,0,0,0,0,1,1,1,0,0,0,1,1,1,0,0,0,0,0,0,0,0,1,1,1,0,0,0,0,1,1,1,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,0,0,0,0,0,1,1,1,0,0,0,0]
#events = [0,1,0,1,0,1,0,0,0,0,1,1,1,1,0,0,0,0,1,1,1,0,0,0,1,1,1,0,0,0,0,0,0,0,0,1,1,1,0,0,0,0,1,1,1,0,0,0,0,1,1,0,0,0,1,1,0,0,1,1,1,0,0,0,0,0,1,1,1,0,0,0,0]
#events = [0,0,1,1,0,0,1,1,0,0,1,1,0,0,1,1,0,0,1,1,0,0,1,1,0,0,1,1,0,0,1,1,0,0,1,1,0,0,1,1,0,0,1,1,0,0,1,1,0,0,1,1,0,0,1,1,0,0,1,1,0,0,1,1,0,0,1,1,0,0,1,1,0,0,1,1,0,0,1,1,0,0,1,1,0,0,1,1,0,0,1,1,0,0,1,1,0,0,1,1,0,0,1,1,0,0,1,1,0,0,1,1,0,0,1,1,0,0,1,1,0,0,1,1,0,0,1,1,0,0,1,1,0,0,]
#events = [0,0,1,1,0,0,1,1,0,0,1,1,0,0,1,1,0,0,1,1,0,0,1,1,0,0,1,1,0,0,1,1,0,0,1,1,0,0,1,1,0,0,1,1,0,0,1,1,0,0,1,1,0,0,1,1,0,0,1,1,0,0,1,1,0,0,1,1,0,0,1,1,0,0]
#events = [0,1,0,1,0,1,0,0,0,0,1,1,1,1,0,0,0,0,1,1,1,0,0,0,1,1,1,0,0,0,0,0,0,0,0,1,1,1,0,0,0,0,1,1,1,0,0,0,0,0,0,0,0,0,1,1,1,0,0,0,0,1,0,1,0,1,0,0,0,0,1,1,1,1,0,0,0,0,1,1,1,0,0,0,1,1,1,0,0,0,0,0,0,0,0,1,1,1,0,0,0,0,1,1,1,0,0,0,0,0,0,1,1,1,0,0,0,0,1,1,1,0,0,0,0]
#events = [1,0,1,0,1,0,0,0,1,1,0,0,0]
#events = [0,1,0,1,0]


events = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]



#events = [0,1,1]
##events = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1]
#events = [0,0,0,1,1,1,0,0,0,1,1,1,0,0,0,1,1,1,0,0]
#events = [0,1,0,1,0,1,0,0,0,0,1,1,1,1,0,0,0,0,1,1,1,0,0,0,1,1,1,0,0,0,0,0,0,0,0,1,1,1,0,0,0,0]

#####################################
### Application runtime variables ###
#####################################

X_indices = []
pred_time_already_monitored = 0
predictions = []
tick_counter_limit = 1
tick_counter_limit_cl = 60
combined_input = []


input_length_limit = 124

jump = -1



###################
### Board stuff ###
###################


parser = argparse.ArgumentParser()
parser.add_argument('--timeout', type=int, help='timeout for device discovery or connection', required=False,
                    default=0)
parser.add_argument('--ip-port', type=int, help='ip port', required=False, default=0)
parser.add_argument('--ip-protocol', type=int, help='ip protocol, check IpProtocolType enum', required=False,
                    default=0)
parser.add_argument('--ip-address', type=str, help='ip address', required=False, default='')
parser.add_argument('--serial-port', type=str, help='serial port', required=False, default='')
parser.add_argument('--mac-address', type=str, help='mac address', required=False, default='')
parser.add_argument('--other-info', type=str, help='other info', required=False, default='')
parser.add_argument('--streamer-params', type=str, help='streamer params', required=False, default='')
parser.add_argument('--serial-number', type=str, help='serial number', required=False, default='')
parser.add_argument('--board-id', type=int, help='board id, check docs to get a list of supported boards',
                    required=True)
parser.add_argument('--file', type=str, help='file', required=False, default='')
args = parser.parse_args()
params = BrainFlowInputParams()
params.ip_port = args.ip_port
params.serial_port = args.serial_port
params.mac_address = args.mac_address
params.other_info = args.other_info
params.serial_number = args.serial_number
params.ip_address = args.ip_address
params.ip_protocol = args.ip_protocol
params.timeout = args.timeout
params.file = args.file
board = BoardShim(args.board_id, params)
board.prepare_session()
board = hp.initialize_board(args)
"""

params = BrainFlowInputParams()
board_id = BoardIds.SYNTHETIC_BOARD.value
sampling_rate = BoardShim.get_sampling_rate(board_id)
board = BoardShim(board_id, params)
board.prepare_session()

"""









def do_initialization():

    fbm.setinitstatustrue()
    model, X_indices = cl.calibrate(board, events, interval_time, doc)
    if model != 0:
        fbm.setinitdone()

    fbm.setTrainingStatusOff()

    model_filename = "models/latestpatientmodel.sav"
    indices_filename = "models/latestmodelindices.csv"

    pickle.dump(model, open(model_filename, 'wb'))
    pd.DataFrame(X_indices).to_csv(indices_filename,sep=',', float_format='%.15f')

    fbm.updateLastModelTraining()

    return model, X_indices




def do_initialization_adv():

    fbm.setadvancedtrainingstatustrue()
    model, X_indices = cl.calibrate_adv(board, events_adv, interval_time, doc)
    if model != 0:
        fbm.setinitdone()

    fbm.setTrainingStatusOff()




inputs_while_in_prod = []


def do_classification(model, X_indices):

    fbm.setclassificationstatustrue()
    input_length_limit = 124

    ticks = 0
    tick_counter = 0   
    global tick_counter_limit
    global combined_input
    global jump
    global predictions
    global pred_time_already_monitored
    global inputs_while_in_prod

    while True:

            #######################################  
            ### Make predictions after interval ###
            #######################################

            #fps.tick(50)
            if ticks == 0:
                wegschmeissen = board.get_board_data()

            time.sleep(0.5)
            ticks += 1


            if ticks % 2 == 0:
                #print("ticki", ticks, X_indices)
                pred_start = time.time()
                print("get latest input started")
                latest_input = board.get_board_data()

                combined_input_sensor = []

                for sensor in latest_input:
                    combined_input_row = []
                    c = 0
                    for val in sensor:
                        if c < input_length_limit:
                            combined_input_row.append(val)
                        c += 1
                    combined_input_sensor.append(combined_input_row)
                

                combined_input.append(combined_input_sensor)


                print("PRINZ", len(latest_input[0]))
                print("get latest input finished")
                if ticks == 10:
                    ticks = 0
                    input_data = cl.get_input_data(combined_input, X_indices)
                    inputs_while_in_prod.append(input_data)
                    hp.monitor("input_data GHAS", input_data)
                    latest_input = []
                    combined_input = []
                    #print("INPUT DATA ZERRRRRROOOOO", input_data, model.predict([input_data]))
                    #print("DFID", input_data)
                    #print("DFID2", input_data[0])
                    ones = 0
                    zeros = 0
                    jump = 0


                    ic = 0
                    print(len(input_data), len(input_data[0]), "qweqwirqwqweoiqqweiqweqwoeuiquweioweuqwioweuqwioeuqweioqeuqweioqweuqwoieueiqweuqwieqweuqwioeuqweqwioeqwiu")
                    
                    for i in input_data:
                        if ic >= 0:
                            print("BEFORE SUB PREDICTION LEN I", len(i))
                            try:
                                pr = model.predict([i])[0]

                            except:
                                pr = 0
                            print("SUB PREDICTION: ", pr)
                            if pr == 0:
                                zeros += 1
                            else:
                                ones += 1
                        ic += 1

                    jump = ones

                    predictions.append(jump)
                    print("THIS IS THE FUCKING PREDICTION RIGHT NOW MANNNN", jump)

                    fbm.updatePred(jump)
                    wegschmeissen = board.get_board_data()


            """
            if tick_counter == tick_counter_limit_cl:
                tick_counter = 0
                inputs_while_in_prod_filename = "data/inputs_prod_reduced_" + str(time.time()) + ".csv"
                pd.DataFrame(inputs_while_in_prod).to_csv(inputs_while_in_prod_filename,sep=',', float_format='%.15f')
                if not fbm.check_classification_status():
                    return
            """

            tick_counter += 1        







def do_classification_youtube(model, X_indices):

    fbm.setyoutubestatustrue()
    input_length_limit = 124

    ticks = 0
    tick_counter = 0   
    global tick_counter_limit
    global combined_input
    global jump
    global predictions
    global pred_time_already_monitored

    while True:

            #######################################  
            ### Make predictions after interval ###
            #######################################

            #fps.tick(50)
            if ticks == 0:
                wegschmeissen = board.get_board_data()

            time.sleep(0.5)
            ticks += 1


            if ticks % 2 == 0:
                #print("ticki", ticks, X_indices)
                pred_start = time.time()
                print("get latest input started")
                latest_input = board.get_board_data()

                combined_input_sensor = []

                for sensor in latest_input:
                    c = 0
                    combined_input_row = []
                    for val in sensor:
                        if c < input_length_limit:
                            combined_input_row.append(val)
                        c += 1
                    combined_input_sensor.append(combined_input_row)
                

                combined_input.append(combined_input_sensor)

                print("PRINZ", len(latest_input[0]))
                print("get latest input finished")
                if ticks == 6:
                    ticks = 0
                    input_data = cl.get_input_data(combined_input, X_indices)
                    hp.monitor("input_data GHAS", input_data)
                    latest_input = []
                    combined_input = []
                    #print("INPUT DATA ZERRRRRROOOOO", input_data, model.predict([input_data]))
                    #print("DFID", input_data)
                    #print("DFID2", input_data[0])
                    ones = 0
                    zeros = 0
                    jump = 0

                    for i in input_data:
                        print("BEFORE SUB PREDICTION LEN I", len(i))
                        try:
                            pr = model.predict([i])[0]
                        except:
                            pr = 0
                        print("SUB PREDICTION: ", pr)
                        if pr == 0:
                            zeros += 1
                        else:
                            ones += 1

                    jump = ones

                    predictions.append(jump)
                    print("THIS IS THE FUCKING PREDICTION RIGHT NOW MANNNN", jump)

                    fbm.updatePredYoutube(jump)
                    wegschmeissen = board.get_board_data()

                if pred_time_already_monitored < 2:
                    pred_end = time.time()
                    fbm.monitor_value(doc, "prediction_time", (pred_end -  pred_start))
                    pred_time_already_monitored += 1

        
            if tick_counter == tick_counter_limit_cl:
                tick_counter = 0
                if not fbm.check_youtube_status():
                    return

            tick_counter += 1     













if __name__ == "__main__":
    model = []
    X_indices = []

    application_reaction_time = 600
    application_reaction_time_adv = 50

    try:
        if fbm.initsession_platform():
            model_filename = "models/latestpatientmodel.sav"
            indices_filename = "models/latestmodelindices.csv"

            model = pickle.load(open(model_filename, 'rb'))
            X_indices_data = pd.read_csv(indices_filename)
            X_indices_raw = X_indices_data.values.tolist()


            for row in X_indices_raw:
                X_indices.append(row[1])

            print("X_INDICES bitte aufpassen: ", X_indices)

        board.start_stream()
        fbm.initsession()

        startts, classts, advts, youts = fbm.getStarttime()
        #classts = fbm.getClasstime()
        ticker_start = 0
        class_ticker_start = 0

        while True:
            if time.time() * 1000 > classts - application_reaction_time and time.time() * 1000 < classts + application_reaction_time:
                do_classification(model, X_indices)
            elif time.time() * 1000 > startts - application_reaction_time and time.time() * 1000 < startts + application_reaction_time:
                model, X_indices = do_initialization()
            elif time.time() * 1000 > advts - application_reaction_time and time.time() * 1000 < advts + application_reaction_time:
                model, X_indices = do_initialization()
            elif time.time() * 1000 > youts - application_reaction_time and time.time() * 1000 < youts + application_reaction_time:
                do_classification_youtube(model, X_indices)
            time.sleep(0.05)
            ticker_start += 1
            if ticker_start == 20:
                startts, classts, advts, youts = fbm.getStarttime()
                wegschmeissen = board.get_board_data()
                ticker_start = 0




    finally:
        fbm.stopsession()

board.stop_stream()


board.release_session() 


def exit_handler():
    fbm.stopsession()

atexit.register(exit_handler)


# sudo python3 questionnaire.py --serial-port /dev/cu.usbserial-DM03H7U4 --board-id 2