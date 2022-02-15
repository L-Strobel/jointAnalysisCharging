from multiprocessing import Process, Queue

import chargingmodel.preprocessing as preprocessing
import chargingmodel.optimize as optimize
import chargingmodel.tools as tools

# Yield successive n-sized chunks from lst.
def chunks(lst, n):
    for i in range(0, len(lst), n):
        yield lst[i:i + n]

# Optimize the charging for every region individually.
# Set n_worker =! 1 to run in parralel batches
# -> Speed up at the cost of slight errors.
def run(*, scenario, config, dbName, regionIDs, aggFactors, n_worker, verbose, batchsize=25):
    # Preprocess data
    if verbose:
        print("Processing...")
        cnt = 0

    residualLoad = preprocessing.getResidual("all")
    agents = []
    for i, regionID in enumerate(regionIDs):
        agents.extend(preprocessing.getAgents(scenario, config, regionID, aggFac=aggFactors[i]))
        # Progressbar
        if verbose:
            cnt += 1
            tools.tickProg(cnt/len(regionIDs))
    agents = sorted(agents, key=lambda x: x.capacity)

    if verbose:
        print("\nData preprocessed!")
    
    # Create agent batches for parallel computing
    batches = chunks(agents, batchsize)

    # Create segments if the optimization horizon is not the entire year
    if config["horizon"] == "full":
        segments = False
    else:
        segments = tools.createSegmentation(config["horizon"])

    # Run in batches
    queue = Queue()
    currentWork = []
    for i, batch in enumerate(batches):
        kwargs = {"agents": batch, "residualLoad": residualLoad,
                  "config": config, "dbName": dbName,
                  "queue": queue, "segments": segments}
        p = Process(target=runBatch, name="Batch_" + str(i),
                    kwargs=kwargs)
        currentWork.append(p)

        # Run porcesses, will wait for all to complete
        if len(currentWork) >= n_worker:
            results = tools.runProcesses(currentWork, verbose, queue)
            currentWork = []

            # Update residual load
            for demands in results:
                for demand in demands.values():
                    for t, load in demand.items():
                        residualLoad[t] += load
            
    # Run remaining processes
    if currentWork:  
        tools.runProcesses(currentWork, verbose, queue)

def runBatch(*, agents, residualLoad, config, dbName, queue, segments):   
    demands, slacks = optimize.optimizeChargingQP_smpl(agents=agents, residualLoad=residualLoad,
                                                      eta=config["chargingEfficiency"],
                                                      SOCStart=config["SOCStart"],
                                                      deltaT=0.25, verbose=False,
                                                      segments=segments)

    # Save
    tools.saveDB(agents=agents, demands=demands, slacks=slacks, dbName=dbName)

    # Back to main thread
    queue.put(demands)