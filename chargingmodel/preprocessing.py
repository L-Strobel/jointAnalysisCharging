import random
from datetime import datetime as dt
from enum import Enum
from typing import NamedTuple, List
import math
import os
from os import listdir
from os.path import isfile, join

import pandas as pd
import numpy as np

# Data structures
class Location(Enum):
    HOME = "home"
    WORK = "work"
    PUBLIC = "public"

# Charging event
class Event(NamedTuple):
    start:int
    stop:int
    consumption:float
    pMax:float
    purpose: int

# Electric vehicle agent
class Agent(NamedTuple):
    name: str
    model: str
    events: List[Event]
    aggFac: int
    capacity: float
    evseGroup: int
    regionID: int

def getResidual(regionID, startDate=dt(2030, 1, 1), endDate=dt(2031, 1, 1)):
    directory = os.path.join(os.path.dirname(__file__), "input/residual-load")
    if regionID == "all":
        fns = [f for f in listdir(directory) if isfile(join(directory, f))]
    else:
        fns = [regionID + ".csv"]
    
    timeWindow = slice(startDate, endDate)
    residualLoad = None
    for fn in fns:
        df = pd.read_csv(directory + "/" + fn, sep=";", parse_dates=["TimeStamp"], index_col="TimeStamp")
        if residualLoad is None:
            residualLoad = df.loc[timeWindow, :].values # pylint: disable=no-member
        else:
            residualLoad += df.loc[timeWindow, :].values # pylint: disable=no-member
    # Drop last value if datetime end was in df
    if endDate in df.index: # pylint: disable=no-member
        residualLoad = residualLoad[:-1]
    return list(residualLoad.reshape(-1))

def getEVModel(N, shares, models):
    # Create a distribution of models
    # Deterministic
    cnts = [math.floor(N*share) for share in shares]
    
    while sum(cnts) < N:
        diffs = [cnts[i]-N*share for i, share in enumerate(shares)]
        cnts[diffs.index(min(diffs))] += 1
    
    rslt = []
    for i, cnt in enumerate(cnts):
        rslt.extend([models[i]] * cnt)
    return rslt

def getEVSEGroup(N, shares):
    # Creates equally distributed classes over N spaces.
    # shares: Frequency of each class
    # Example: N = 10, shares = [0.5, 0.3, 0.2]
    #   Result: [0, 1, 0, 2, 0, 1, 0, 2, 0, 1]
    if (sum(shares)) - 1 > 1e-6:
        raise ValueError("Shares don't add up!")
    
    n = N
    lsts = []
    for idx, share in enumerate(shares[:-1]):
        # Distribute one class over n spaces
        x = share/(1-sum(shares[:idx]))
        grp = [0]*n
        for j in range(n):
            if int(j%(1/x)) == 0:
                grp[j] = 1
        lsts.append(grp)
        n -= sum(lsts[-1])
    # Remaining agents
    lsts.append([1]*n)

    # Merge lists
    groups = [-1]*N
    for i in range(N):
        for j in range(len(lsts)):
            val = lsts[j].pop(0)
            if val == 1:
                groups[i] = j
                break
    return groups

def getPMax(loc, evseConfig, evseGroup, pMaxEV=None):
    # Public charging
    if loc == Location.PUBLIC:
        conf = evseConfig["public"]
        # Probabilities: [No station found, slow station, fast station]
        probs = [1 - conf["prob"], conf["prob"]*conf["slow"]["share"],
                 conf["prob"]*conf["fast"]["share"]]
        # Charging rates
        chrgRates = [0.0, conf["slow"]["power"], conf["fast"]["power"]]
        if abs(sum(probs) - 1) > 1e-6: 
            raise ValueError("Probabilities don't add up!")
        # Power at station
        pMax = random.choices(chrgRates, weights=probs)[0]
    else:
        groups = evseConfig["privat"]
        # Charging rate
        pMax = groups[evseGroup][loc.value]
    
    # Maximum is charging rate of EV
    if pMaxEV is not None:
        pMax = min(pMaxEV, pMax)
    return pMax

# Turn trips into parking events
def getEvents(carData, aggFac, evseConfig, evseGroup, endInt, config, consumption, pMaxEV=None):
    mapPurpLoc = {0: Location.WORK, 1: Location.PUBLIC, 2: Location.PUBLIC, 3: Location.PUBLIC, 4: Location.PUBLIC,
                  5: Location.PUBLIC, 6: Location.PUBLIC, 7: Location.HOME, 8: Location.HOME, 9: Location.PUBLIC}

    # Get events
    events = []
    savedCons = 0
    lastTimestep = 0
    currentLoc = Location.HOME # Start condition
    currentPurp = 7 # Same as above
    for dep, arr, purp, cons in zip(carData["Departure"],
                                    carData["Arrival"],
                                    carData["Purpose"],
                                    consumption):
        # Parking event happens in an instant -> save consumption and continue
        if lastTimestep == dep:
            savedCons += cons
            continue
        
        # Maximum charging rate at event
        pMaxEvent = getPMax(currentLoc, evseConfig, evseGroup, pMaxEV=pMaxEV) * config["sensiAdjP"]

        # Parking event
        events.append(Event(lastTimestep, dep, (cons + savedCons) * aggFac * config["sensiAdjCons"],
                            pMaxEvent * aggFac, currentPurp))

        # Loop memory
        lastTimestep = arr
        currentLoc = mapPurpLoc[purp]
        currentPurp = purp
        savedCons = 0
    
    # From last event to end of year
    pMaxEvent = getPMax(currentLoc, evseConfig, evseGroup, pMaxEV=pMaxEV) * config["sensiAdjP"]
    events.append(Event(lastTimestep, endInt, 0.0, pMaxEvent * aggFac, currentPurp))
    return events

def getAgents(scenario, config, regionID, startDate=dt(2030, 1, 1), endDate=dt(2031, 1, 1), resolution="15T", aggFac=1):
    # Container for all optimization relevant data of one vehicle.
    # Output: All units in MW or MWh (kW creates numerical problems).

    # Load behavior data
    usecols = ["Departure", "Arrival", "Purpose", "Vehicle_id",
               "MeanSpeed [km/h]", "Temperature [deg_C]", "Distance [km]"]
    dirname = os.path.dirname(__file__)
    pathBehave =  os.path.join(dirname, f"input/behavior/{regionID}.csv")
    dfBehave = pd.read_csv(pathBehave, sep=";", parse_dates=["Departure", "Arrival"],
                           usecols=usecols)

    # Aggregation Factor
    aggFac *= config["sensiAdjFS"]

    # Convert time to integers for faster indexing
    res_in_secs = int(pd.to_timedelta(resolution).total_seconds())
    def idxToInt(x):
        y = ((x - startDate).dt.total_seconds() / res_in_secs).round().astype(int)
        return y
    dfBehave["Departure"] = idxToInt(dfBehave["Departure"])
    dfBehave["Arrival"] = idxToInt(dfBehave["Arrival"])
    endInt = int(round((endDate - startDate).total_seconds() / res_in_secs)) 

    # IDs
    agentIDs = sorted(dfBehave["Vehicle_id"].unique())

    # EVSE groups
    evseConfig = config["scenarios"][scenario]
    evseGroups = getEVSEGroup(len(agentIDs), [i["share"] for i in evseConfig["privat"]])

    # EV-Models
    evConfig = config["eVModels"]
    models = getEVModel(len(agentIDs), [i['share'] for i in evConfig.values()], [i for i in evConfig.keys()])

    # Get agents
    agents = []
    for agentID in agentIDs:
        # Data of one EV
        carData = dfBehave[dfBehave["Vehicle_id"] == agentID]
        
        # ID
        modelID = models[agentID]

        # Cut events not in time window
        carData = carData[(carData.Departure >= 0) & (carData.Arrival < endInt)]

        # Calc consumption
        consumption = calcConsumption(carData['MeanSpeed [km/h]'].values,
                                      carData['Temperature [deg_C]'].values,
                                      carData['Distance [km]'].values,
                                      config['eVModels'][modelID])

        # Parameter
        capacity = config["eVModels"][modelID]["capacity_kWh"] * config["sensiAdjCap"]
        evseGroup = evseGroups[agentID]

        # Events
        events = getEvents(carData, aggFac / 1000, evseConfig, evseGroup, endInt, config, consumption)

        agent = Agent(name=regionID+"_"+f'{agentID:02}', events=events, model=modelID,
                      aggFac=aggFac, capacity=capacity * aggFac / 1000,
                      evseGroup=evseGroup, regionID=regionID)
        agents.append(agent)
    return agents

def calcConsumption(meanSpeed, temperature, distance, model):
    # Normal consumption
    cons_base = np.empty(len(meanSpeed))
    mask_s = (meanSpeed >= 34) & (meanSpeed <= 78)
    cons_base[mask_s] = model["consCITY_kWh/100km"] + (meanSpeed[mask_s]-34)*(model["consHWY_kWh/100km"] - model["consCITY_kWh/100km"]) / (78-34)
    cons_base[meanSpeed < 34] = model["consCITY_kWh/100km"]
    cons_base[meanSpeed > 78] = model["consHWY_kWh/100km"]
    # Percentage increase due to temperature
    cons_add = np.empty(len(meanSpeed))
    cons_add[temperature > 35] = model["extraCons35Deg_perc"]
    cons_add[temperature < -6] = model["extraCons-6Deg_perc"]
    mask_tr = (temperature <= 35) & (temperature >= 23)
    cons_add[mask_tr] = (temperature[mask_tr]-23) * model["extraCons35Deg_perc"] / (35-23)
    mask_tl = (temperature < 23) & (temperature >= -6)
    cons_add[mask_tl] = (23-temperature[mask_tl]) * model["extraCons-6Deg_perc"] / (23+6)
    # Return in kWh
    return cons_base*(1+cons_add/100)*distance/100
