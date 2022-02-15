from multiprocessing import Process

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
    currentWork = []
    for i, regionID in enumerate(regionIDs):
        # Preprocess data
        agents = preprocessing.getAgents(scenario, config, regionID, aggFac=aggFactors[i])
        residualLoad = preprocessing.getResidual(regionID)

        # Create process
        kwargs = {"agents": agents, "residualLoad": residualLoad,
                  "config": config, "dbName": dbName, "segments": segments}
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

def runCounty(*, agents, residualLoad, config, dbName, segments):
    demands, slacks = optimize.optimizeChargingQP_smpl(agents=agents, residualLoad=residualLoad,
                                                      eta=config["chargingEfficiency"],
                                                      SOCStart=config["SOCStart"],
                                                      deltaT=0.25, verbose=False,
                                                      segments=segments)

    # Save
    tools.saveDB(agents=agents, demands=demands, slacks=slacks, dbName=dbName)
