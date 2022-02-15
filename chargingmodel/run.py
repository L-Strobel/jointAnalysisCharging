import json
import random
import argparse
import os

import pandas as pd

import chargingmodel.tools as tools
import chargingmodel.uncontrolled as uncontrolled
import chargingmodel.optNational as optNational
import chargingmodel.optCounty as optCounty

def getArgs():
    parser = argparse.ArgumentParser()
    # Ouput database name
    parser.add_argument("--dbName", default=None)
    # EVSE availability scenario
    parser.add_argument("--scenario", default="Realistic") # See config for all options
    # Strategy
    parser.add_argument("--strategy", default="Uncontrolled") # Uncontrolled, Opt_National, Opt_County
    # Run configuration
    parser.add_argument("--silent", dest='verbose', action='store_false') # No console printing
    parser.set_defaults(verbose=True)
    parser.add_argument("--n_worker", default=1, type=int) # Number of cores to use
    return parser.parse_args()

def run(dbName=None, scenario="Realistic", strategy="Uncontrolled", verbose=True, n_worker=1):
    # Only the public charging probability contains randomness currently
    random.seed(123)

    dirname = os.path.dirname(__file__)
    # Get config
    with open(os.path.join(dirname, 'config.json')) as f:
        config = json.load(f)

    # Prepare output sqlite database
    if dbName is None:
        dbName = os.path.join(dirname, f"output/{scenario}_{strategy}.db")
    tools.innitDB(dbName)

    # Get Region IDs (In the paper these regions where counties)
    regionData = pd.read_csv(os.path.join(dirname,"input/regions.csv"), sep=";")
    regionIDs = regionData["ID"].astype(str).values

    # Aggregation factors of one car in each county
    evPenetration = config['fleetsize'] / regionData['RegisteredCars'].sum()
    aggFactors = regionData['RegisteredCars']*evPenetration / 100 # 100: Simulated evs in mobility model

    # Run
    kwargs = {"scenario": scenario, "config": config, "dbName":dbName, "regionIDs": regionIDs,
              "aggFactors": aggFactors, "n_worker": n_worker, "verbose": verbose}
    if strategy == "Uncontrolled":
        uncontrolled.run(**kwargs)
    elif strategy == "Opt_National":
        optNational.run(**kwargs)
    elif strategy == "Opt_County":    
        optCounty.run(**kwargs)
    else: 
        print(f"Invalid strategy! {strategy}")
    
if __name__ == "__main__":
    run(**vars(getArgs()))
    
    
    

    
    
    
    
    
