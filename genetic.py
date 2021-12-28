""" Genetic algorithm for feature selection """

import random
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_hastie_10_2
from sklearn.ensemble import GradientBoostingClassifier
from sklearn import metrics
import numpy as np
import helpers as hp
import firebasemonitoring as fbm
import time

def generateDnaString(numberOfGenes, X_input):
    # X_input must be transposed!

    possibleGenes = len(X_input)
    geneNumbers = []
    dnaString = [] 

    i = 0
    while i < numberOfGenes:
        rnd = -1
        
        while rnd in geneNumbers or rnd == -1:
            rnd = random.randint(0, (possibleGenes - 1))

        geneNumbers.append(rnd)
        dnaString.append(X_input[rnd])

        i += 1

    return [geneNumbers, np.transpose(dnaString)]

def evaluateDnaString(dnaString, events, trainingNumber, estimators):
    # Create and train random forest model

    #clf=RandomForestClassifier(n_estimators=estimators)
    clf = GradientBoostingClassifier(n_estimators=estimators, learning_rate=1.0, max_depth=2, random_state=0)

    try:
        clf.fit(dnaString[:trainingNumber - 1], events[:trainingNumber - 1])

        prediction=clf.predict(dnaString[trainingNumber:])

        return metrics.accuracy_score(events[trainingNumber:], prediction)

    except:
        return 0

def killWeakSubjects(dnaStrings, killRate, trainingNumber, estimators, events, numberOfGenes):

    kill = len(dnaStrings) * killRate
    res = [] 

    for i in dnaStrings:
        #if (len(i[1][0]) == numberOfGenes):
        try:
            res.append(evaluateDnaString(i[1], events, trainingNumber, estimators))
        except:
            res.append(0)

    sortedDnaStrings = [x for _,x in sorted(zip(res,dnaStrings))]

    return sortedDnaStrings[len(sortedDnaStrings) - int(kill) + 1:], res

def sex(subjectOne, subjectTwo, numberOfKids, X_input, numberOfMutations):
    # X_input must be transposed!

    possibleGenes = len(X_input)
    kids = []
    i = 0

    while i < numberOfKids:
        kidDnaString = []
        kidGeneNumbers = []

        c = 0
        while c < len(subjectOne[0]):
            rnd = random.randint(0,1)

            if rnd == 0:
                kidDnaString.append(np.transpose(subjectOne[1])[c])
                kidGeneNumbers.append(subjectOne[0][c])
            else:
                kidDnaString.append(np.transpose(subjectTwo[1])[c])
                kidGeneNumbers.append(subjectTwo[0][c])

            c += 1

        kids.append(mutateGenes(kidGeneNumbers, kidDnaString, numberOfMutations, X_input))

        i += 1

    return kids

def mutateGenes(geneNumbers, dnaString, numberOfMutations, X_input):
    # X_input must be transposed!


    possibleGenes = len(X_input)

    i = 0
    while i < numberOfMutations:
        geneNumber = random.randint(0, len(geneNumbers) - 1)
        rnd = random.randint(0, possibleGenes - 1)

        geneNumbers[geneNumber] = rnd
        dnaString[geneNumber] = X_input[rnd]

        i += 1

    return [geneNumbers, np.transpose(dnaString)]

def simulateEvolution(X_input, numberOfGenes, numberOfSubjects, numberOfGenerations, numberOfMutations, killRate, trainingNumber, estimators, events, doc):
    # X_input must be transposed!

    fbm.monitor_value(doc, "genetic_preferences", {"numberOfGenes": numberOfGenes, "numberOfSubjects": numberOfSubjects, "numberOfGenerations": numberOfGenerations, "numberOfMutations": numberOfMutations, "killRate": killRate, "trainingNumber": trainingNumber, "estimators": estimators})
    subjects = []

    i = 0
    while i < numberOfSubjects:
        subjects.append(generateDnaString(numberOfGenes, X_input))
        
        
        i += 1

    k = 0
    correlation_scores_res = []

    training_start = time.time()

    while k < numberOfGenerations:
        decimatedSubjects, correlation_scores = killWeakSubjects(subjects, killRate, trainingNumber, estimators, events, numberOfGenes)
        
        correlation_scores_res.append(max(correlation_scores))
        

        kids = []
        subjects = decimatedSubjects

        c = 0
        if k < numberOfGenerations - 1:
            while c < len(decimatedSubjects[0]):
                new_kids = sex(decimatedSubjects[c], decimatedSubjects[len(decimatedSubjects) - c - 1], 2, X_input, numberOfMutations)

                for i in new_kids:
                    subjects.append(i)
                c += 1
        k += 1

    fbm.monitor_value(doc, "genetic_correlations", correlation_scores_res)
    training_end = time.time()
    fbm.monitor_value(doc, "genetic_algorithm_training_time", (training_end - training_start))
    fbm.monitor_value(doc, "genetic_datapoint_set", subjects[len(subjects) - 1][0])

    return subjects[len(subjects) - 1][0]