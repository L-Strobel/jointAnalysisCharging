import sys
import pandas as pd
import sqlite3
import os

def processRegionalLoad(dbName):
    dirname = os.path.dirname(__file__)
    regionData = pd.read_csv(os.path.join(dirname,"input/regions.csv"), sep=";")
    regionIDs = regionData["ID"].values

    # For progressbar
    cnt = 0
    length = len(regionIDs)
    prog_per_it = 40 / length

    conn = sqlite3.connect(dbName)
    c = conn.cursor()

    c.execute("DROP TABLE IF EXISTS Regions")
    c.execute("""CREATE TABLE IF NOT EXISTS Regions (
        Time timestamp,
        Type text,
        PowerMW real
    ) """)

    for regionID in regionIDs:
        c.execute("""INSERT INTO Regions (Time, PowerMW, Type)
            Select Time, SUM(PowerMW), Region
            From TimeSeries
            INNER JOIN Agents on Agents.AgentID = TimeSeries.AgentID
            Where Region = (?)
            GROUP BY Time""", (regionID, ))
        conn.commit()
        # Progressbar
        cnt += 1
        prog = int(cnt * prog_per_it)
        sys.stdout.write("\r[%-40s] %d%%" % ('=' * prog, 2.5 * prog))
        sys.stdout.flush()

    conn.close()

def processTotalLoad(dbName):
    conn = sqlite3.connect(dbName)
    c = conn.cursor()

    c.execute("DROP TABLE IF EXISTS Total")
    c.execute("""CREATE TABLE IF NOT EXISTS Total (
        Time timestamp,
        Type text,
        PowerMW real
    ) """)

    c.execute("""INSERT INTO Total (Time, PowerMW, Type)
        Select Time, SUM(PowerMW), 'Total'
        FROM TimeSeries
        GROUP BY Time """)

    conn.commit()
    conn.close()

def getProcessed(dbName, region="Total"):
    table = 'Total' if region == "Total" else 'Regions'
    
    conn = sqlite3.connect(dbName)

    c = conn.cursor()
    # Get time 
    c.execute("""SELECT *
                 FROM Simulation""")
    t = c.fetchone()
    index = pd.date_range(start=t[0], end=t[1], freq=t[2], closed="left")
    
    # Get power
    sql = f"""SELECT Time, PowerMW
              FROM {table}
              WHERE Type = '{region}'"""
    _df = pd.read_sql(sql, conn, index_col="Time", parse_dates="Time").reindex(index).fillna(0)
    
    conn.close()
    return _df