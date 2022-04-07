from multiprocessing import Process, Queue

import chargingmodel.preprocessing as preprocessing
import chargingmodel.optimize as optimize
import chargingmodel.tools as tools

# Optimize the charging for every region individually.
# Runs for all regions in parrallel.
def run(*, scenario, config, dbName, regionIDs, aggFactors, n_worker, verbose):
    # Set optimization horizon
    if config["horizon"] == "full":
        segments = False
    else:
        segments = tools.createSegmentation(config["horizon"])

    # Run for all chosen counties independently
    queue = Queue()
    currentWork = []
    for i, regionID in enumerate(regionIDs):
        # Preprocess data
        agents = preprocessing.getAgents(scenario, config, regionID, aggFac=aggFactors[i])
        residualLoad = preprocessing.getResidual(regionID)

        # Create process
        kwargs = {"agents": agents, "residualLoad": residualLoad,
                  "config": config, "queue": queue, "segments": segments}
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


def runCounty(*, agents, residualLoad, config, queue, segments):
    demands, slacks = optimize.optimizeChargingQP_smpl(agents=agents, residualLoad=residualLoad,
                                                      eta=config["chargingEfficiency"],
                                                      SOCStart=config["SOCStart"],
                                                      deltaT=0.25, verbose=False,
                                                      segments=segments)
    queue.put((agents, demands, slacks))
