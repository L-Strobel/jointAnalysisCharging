from multiprocessing import Process, Queue

import chargingmodel.preprocessing as preprocessing
import chargingmodel.optimize as optimize
import chargingmodel.tools as tools

# Uncontrolled charging.
# Every agent charges immediately and as much as possible after arriving at a charging station.
# Runs for all regions in parrallel.
def run(*, scenario, config, dbName, regionIDs, aggFactors, n_worker, verbose):
    queue = Queue()
    currentWork = []
    for i, regionID in enumerate(regionIDs):
        # Preprocess data
        agents = preprocessing.getAgents(scenario, config, regionID, aggFac=aggFactors[i])

        # Create process
        kwargs = {"agents": agents, "config": config,
                  "queue": queue}
        p = Process(target=runCounty, name=str(regionID),
                    kwargs=kwargs)
        currentWork.append(p)

        # Run porcesses, will wait for all to complete
        if len(currentWork) >= n_worker:
            results = tools.runProcesses(currentWork, verbose, queue)
            currentWork = []
            
            for result in results:
                tools.saveDB(agents=result[0], demands=result[1], slacks=result[2], dbName=dbName)

    # Run remaining processes
    if currentWork:
        results = tools.runProcesses(currentWork, verbose, queue)

        for result in results:
            tools.saveDB(agents=result[0], demands=result[1], slacks=result[2], dbName=dbName)


def runCounty(*, agents, config, queue):
    demands, slacks = optimize.immediate(agents=agents,
                                         eta=config["chargingEfficiency"],
                                         SOCStart=config["SOCStart"],
                                         deltaT=0.25)
    queue.put((agents, demands, slacks))