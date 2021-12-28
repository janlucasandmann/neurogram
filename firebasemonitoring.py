"""#########################"""
""" Firebase Monitoring API """
"""#########################"""



###############
### Imports ###
###############

import firebase_admin
from firebase_admin import credentials
from firebase_admin import firestore
import time



########################
### Authentification ###
########################

def init(col_name, doc_name):
    # Use a service account

    cred = credentials.Certificate('private-key.json')
    firebase_admin.initialize_app(cred)

    db = firestore.client()

    collection = db.collection(col_name)
    doc = collection.document(doc_name)

    doc.set({
        "len_extracted_features": 0,
        "training_events": 0,
        "genetic_preferences": {},
        "prediction_time": 0,
        "genetic_algorithm_training_time": 0,
        "datapoints_per_interval": 0,
        "genetic_datapoint_set": [],
        "test_predictions": []
    })

    return doc



##################################
### Update field in collection ###
##################################

def monitor_value(doc, key, value):
    doc.update({key: value})

def initsession_platform():
    db = firestore.client()

    collection_platform = db.collection("platform")
    doc_platform = collection_platform.document("platform")

    doc_platform.update({
        "connection": True,
        "ai_is_training": False,
    })

    doc = doc_platform.get()
    if doc.exists:
        print(f'Document data: {doc.to_dict()}')
        if int(doc.to_dict()["last_model_training"]) != 0:

            doc_platform.update({
                "initialization_done": True,
            })

            return True
    
    return False


def updateLastModelTraining():
    db = firestore.client()

    collection_platform = db.collection("platform")
    doc_platform = collection_platform.document("platform")

    doc_platform.update({
        "last_model_training": int(time.time()),
    })



def initsession():
    db = firestore.client()

    collection = db.collection("questionnaire")
    doc = collection.document("questionnaire")

    doc.update({
        "initialization_ready": True,
    })

    

def stopsession():
    db = firestore.client()

    collection = db.collection("questionnaire")
    doc = collection.document("questionnaire")

    collection_platform = db.collection("platform")
    doc_platform = collection_platform.document("platform")

    doc.update({
        "initialization_ready": False,
    })

    doc_platform.update({
        "connection": False,
        "initialization_done": False,
        "ai_is_training": False,
    })

def updatePred(jump):

    print("UPDATE PRED JUMP NUMBER: ", jump)

    db = firestore.client()

    collection = db.collection("questionnaire")
    doc = collection.document("questionnaire")

    answer = "none"
    brainactivity = "Nicht erhöht"
    security = 0

    if jump >= 2:
        answer = "Ja"
        brainactivity = "Erhöht"
    else:
        answer = "Nein"


    if jump == 5:
        security = 100
    if jump == 4:
        security = 84
    elif jump == 3:
        security = 71
    elif jump == 2:
        security = 25
    elif jump == 1:
        security = 73
    elif jump == 0:
        security = 100


    doc.update({
        "answer": answer,
        "certainty": security,
        "brainactivity": brainactivity
    })
    
    print("updatePred wurde aufgerufen!")
    print(answer, security, brainactivity)




def updatePredYoutube(jump):

    db = firestore.client()

    collection = db.collection("youtube")
    doc = collection.document("youtube")

    hand_up = False

    if jump == 5 or jump == 4 or jump == 3:
        hand_up = True


    doc.update({
        "hand_up": hand_up
    })
    
    print("updatePredYoutube wurde aufgerufen!")
    print(hand_up)




def save_dataset(df):

    db = firestore.client()

    collection = db.collection("datasets")
    doc = collection.document("test")

    doc.set({
        "data": df
    })


def getStarttime():
    db = firestore.client()

    collection = db.collection("platform")
    doc_ref = collection.document("platform")

    doc = doc_ref.get()
    if doc.exists:
        print(f'Document data: {doc.to_dict()}')
        return int(doc.to_dict()["initialization_startpoint"]), int(doc.to_dict()["classification_startpoint"]), int(doc.to_dict()["advancedtraining_startpoint"]), int(doc.to_dict()["youtube_startpoint"])



def getClasstime():
    db = firestore.client()

    collection = db.collection("platform")
    doc_ref = collection.document("platform")

    doc = doc_ref.get()
    if doc.exists:
        print(f'Document data: {doc.to_dict()}')
        return doc.to_dict()["classification_startpoint"]


def setinitdone():
    db = firestore.client()

    collection = db.collection("platform")
    doc = collection.document("platform")

    doc.update({
        "initialization_done": True,
    })

def setinitstatustrue():
    db = firestore.client()

    collection = db.collection("platform")
    doc = collection.document("platform")

    doc.update({
        "initialization_status": True,
    })


def setadvancedtrainingstatustrue():
    db = firestore.client()

    collection = db.collection("platform")
    doc = collection.document("platform")

    doc.update({
        "advancedtraining_status": True,
    })


def setclassificationstatustrue():
    db = firestore.client()

    collection = db.collection("platform")
    doc = collection.document("platform")

    doc.update({
        "classification_status": True,
    })


def setyoutubestatustrue():
    db = firestore.client()

    collection = db.collection("platform")
    doc = collection.document("platform")

    doc.update({
        "youtube_status": True,
    })




def check_classification_status():
    db = firestore.client()

    collection = db.collection("platform")
    doc_ref = collection.document("platform")

    doc = doc_ref.get()
    if doc.exists:
        print(f'Document data: {doc.to_dict()}')
        return doc.to_dict()["classification_status"]


def check_youtube_status():
    db = firestore.client()

    collection = db.collection("platform")
    doc_ref = collection.document("platform")

    doc = doc_ref.get()
    if doc.exists:
        print(f'Document data: {doc.to_dict()}')
        return doc.to_dict()["youtube_status"]



def check_initialization_status():
        db = firestore.client()

        collection = db.collection("platform")
        doc_ref = collection.document("platform")

        doc = doc_ref.get()
        if doc.exists:
            print(f'Document data: {doc.to_dict()}')
            return doc.to_dict()["initialization_status"]


def check_advancedtraining_status():
        db = firestore.client()

        collection = db.collection("platform")
        doc_ref = collection.document("platform")

        doc = doc_ref.get()
        if doc.exists:
            print(f'Document data: {doc.to_dict()}')
            return doc.to_dict()["advancedtraining_status"]


def setTrainingStatusOn():
    db = firestore.client()

    collection = db.collection("platform")
    doc = collection.document("platform")

    doc.update({
        "ai_is_training": True,
    })


def setTrainingStatusOff():
    db = firestore.client()

    collection = db.collection("platform")
    doc = collection.document("platform")

    doc.update({
        "ai_is_training": False,
    })


