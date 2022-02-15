from multiprocessing import Process

import chargingmodel.preprocessing as preprocessing
import chargingmodel.optimize as optimize
import chargingmodel.tools as tools

# Uncontrolled charging.
# Every agent charges immediately and as much as possible after arriving at a charging station.
# Runs for all regions in parrallel.
def run(*, scenario, config, dbName, regionIDs, aggFactors, n_worker, verbose):
    currentWork = []
    for i, regionID in enumerate(regionIDs):
        # Preprocess data
        agents = preprocessing.getAgents(scenario, config, regionID, aggFac=aggFactors[i])

        # Create process
        kwargs = {"agents": agents, "config": config,
                  "dbName": dbName}
        p = Process(target=runCounty, name=str(regionID),
                    kwargs=kwargs)
        currentWork.append(p)

        # Run porcesses, will wait for all to complete
        if len(currentWork) >= n_worker:
            tools.runProcesses(currentWork, verbose)
            currentWork = []
    # Run remaining processes
    if currentWork:
        tools.runProcesses(currentWork, verbose)

def runCounty(*, agents, config, dbName):
    demands, slacks = optimize.immediate(agents=agents,
                                         eta=config["chargingEfficiency"],
                                         SOCStart=config["SOCStart"],
                                         deltaT=0.25, verbose=False)

    # Save
    tools.saveDB(agents=agents, demands=demands, slacks=slacks, dbName=dbName)