import time
import sys
from datetime import datetime as dt
import itertools
import sqlite3

import pandas as pd

# Progressbar
def tickProg(perc):
    prog = int(perc * 40)
    sys.stdout.write("\r[%-40s] %d%%" % ('=' * prog, 2.5 * prog))
    sys.stdout.flush()

# Multithreading wrapper
def runProcesses(processes, verbose, queue=None):
    process_names = ', '.join('{}'.format(k.name) for k in processes)

    if verbose:
        print(f"Starting {process_names} ...")
        st = time.time()

    for process in processes:
        process.start()

    # Has to be between start and join
    if queue is not None:
        results = [queue.get() for _ in processes]
    else:
        results = None

    for process in processes:
        process.join()

    if verbose:
        print(f"{process_names} Done! In {time.time()-st:.2f} secs")
    
    return results

# Split the given time window in individual chunks.
# Options: "D": Day, "M": Month, "W": Week
def createSegmentation(horizon, start=dt(2030, 1, 1), end=dt(2031, 1, 1), resolution="15T"):
    res_in_secs = int(pd.to_timedelta(resolution).total_seconds())
    
    # Split method: Day, Month or Calendar Week
    if horizon == "D":
        splitFunc = dt.toordinal
    elif horizon == "M":
        splitFunc = lambda x: x.month
    elif horizon == "W":
        splitFunc = lambda x: x.isocalendar()[1] # Calendar week
    else:
        raise ValueError(f"Optimization horizon: {horizon} unknown!")
    
    # Result as index
    def idxToInt(x):
        return int(round((x - start).total_seconds() / res_in_secs)) 
    
    # Create time range
    timeFrame = pd.date_range(start, end, inclusive="left", freq=resolution).to_pydatetime() # pylint: disable=no-member
    # Split into segments of horizon length. Convert to integer.
    segments = [idxToInt(list(group)[-1]) + 1 for k, group in itertools.groupby(timeFrame, key=splitFunc)]
    return segments

# Create output database
def innitDB(dbName, startDate=dt(2030, 1, 1), endDate=dt(2031, 1, 1), resolution="15T"):
    conn = sqlite3.connect(dbName)
    c = conn.cursor()
    c.execute("DROP TABLE IF EXISTS Simulation")
    c.execute("""CREATE TABLE Simulation (
        StartDate timestamp,
        EndDate timestamp,
        Resolution text
    ) """)

    c.execute("DROP TABLE IF EXISTS Agents")
    c.execute("""CREATE TABLE Agents (
        AgentID text PRIMARY KEY,
        Region text,
        EVSEgroup integer,
        AggFac real,
        CapacityMWh real,
        CarType text,
        Slack real
    ) """)

    c.execute("DROP TABLE IF EXISTS TimeSeries")
    c.execute("""CREATE TABLE TimeSeries (
        Time timestamp,
        AgentID text,
        PowerMW real,
        FOREIGN KEY (AgentID)
            REFERENCES Agents (AgentID)
                ON DELETE CASCADE
    ) """)

    # Meta-Info about simulation
    c.execute("INSERT INTO Simulation VALUES (?, ?, ?)", (startDate, endDate, resolution))

    conn.commit()
    conn.close()

# Save results of one agent to database
def saveDB(*, agents, demands, slacks, dbName, startDate=dt(2030, 1, 1), endDate=dt(2031, 1, 1), resolution="15T"):
    # SQL - connect
    conn = sqlite3.connect(dbName)
    c = conn.cursor()

    # Meta-Info about agent
    metadata = []
    for agent in agents:
        metadata.append((agent.name, agent.regionID, int(agent.evseGroup), agent.aggFac,
                         agent.capacity, agent.model, slacks[agent.name]))
    c.executemany("INSERT INTO Agents VALUES (?, ?, ?, ?, ?, ?, ?)", metadata)

    # Time series output, saved sparse
    tsData = []
    timeCol = list(pd.date_range(startDate,endDate, inclusive="left", freq=resolution).to_pydatetime()) # pylint: disable=no-member
    for agent in agents:
        for idx, val in demands[agent.name].items():
            p = int(val * 1000 )/1000 # Floor to 3 decimals
            if p > 0:
                tsData.append((timeCol[idx], agent.name, p))
    c.executemany("INSERT INTO TimeSeries VALUES (?, ?, ?)", tsData)
    conn.commit()
    conn.close()