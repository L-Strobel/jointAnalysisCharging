import gurobipy as grb

import chargingmodel.tools as tools

# Adapt parking events to the optimization horizon
def getSegmentEvents(events, segEndings):
    segEvents, currentSet = [], []
    segIdx = 0
    curSegEnd = segEndings[0]

    for event in events:
        # Normal case
        if event.stop <= curSegEnd:
            currentSet.append(event)
        # Event split between horizonts
        elif event.start < curSegEnd:
            # Add first half to current segment
            currentSet.append(event._replace(stop = curSegEnd, consumption = 0.0))

            # Next Segment
            segEvents.append(currentSet)
            currentSet = []
            segIdx +=1 
            curSegEnd = segEndings[segIdx]

            # Add later half to new segment
            currentSet.append(event._replace(start = curSegEnd+1))
        else:
            # Next Segment
            segEvents.append(currentSet)
            currentSet = []
            segIdx +=1 
            curSegEnd = segEndings[segIdx]

            # Add to new segment
            currentSet.append(event)
    return segEvents

# Calculate the slack parameter. Essentially immediate() with less data storing.
def getSlack(*, agent, events, eBatStart, etaTimesDelta):
    slacks = {}

    # Starting condition
    eBatCurrently = eBatStart
    # Simulation loop
    for eventidx, event in enumerate(events):
        # Charge maximum possible
        eCharge = max(0, min(event.pMax * etaTimesDelta * (event.stop-event.start), agent.capacity - eBatCurrently))
        # Charged energy to battery
        eBatCurrently += eCharge
        # Consumption between charging events
        eBatCurrently -= event.consumption
        # Slack
        if eBatCurrently < 0:
            slack = abs(eBatCurrently)
            eBatCurrently = 0
        else:
            slack = 0
        slacks[eventidx] = slack
    mxEbatEnd = eBatCurrently
    return slacks, mxEbatEnd

# Define the constraints of one vehicle
def setAgentConstr(*, model, agent, events, slack, mxEbatEnd, eBatStart, etaTimesDelta, eBatGoal):
    # Variables
    eBatPre, pCharge, eBatPost = {}, {}, {}

    # Starting Value
    eBatPre[0] = eBatStart

    for eventIdx, event in enumerate(events):
        # Variables
        # Charging power of EV at time step. Limits: 0 <= pCharge <= P_max[t, car]
        for t in range(event.start, event.stop):
            pCharge[t] = model.addVar(ub=event.pMax)

        # Battery energy content before event. Limits: 0 <= eBat <= capacity[car]
        eBatPre[eventIdx + 1] = model.addVar(ub=agent.capacity)

        # Battery energy content after event. Limits: 0 <= eBat <= capacity[car]
        eBatPost[eventIdx] = model.addVar(ub=agent.capacity)

        # Constraints
        # E_bat after charging = E_bat before charging + charged energy
        model.addLConstr(lhs=grb.quicksum(pCharge[t] for t in range(event.start, event.stop)) * etaTimesDelta + eBatPre[eventIdx],
                         sense=grb.GRB.EQUAL,
                         rhs=eBatPost[eventIdx])

        # E_bat after driving = E_bat before driving - consumption + slack
        model.addLConstr(lhs=eBatPost[eventIdx] - event.consumption + slack[eventIdx],
                         sense=grb.GRB.EQUAL,
                         rhs=eBatPre[eventIdx + 1])
    # Time window energy constraint
    boundEbatEnd = min(mxEbatEnd, eBatGoal)
    model.addLConstr(lhs=boundEbatEnd, sense=grb.GRB.LESS_EQUAL, rhs=eBatPre[len(events)])
    return pCharge, eBatPre[len(events)]

# Full quadratic program
def optimizeChargingQP(*, agents, residualLoad, eta, SOCStart, deltaT):
    # Repeated calculations
    etaTimesDelta = eta * deltaT

    # Solution
    nTimesteps = len(residualLoad)
    demands = {agent.name: {} for agent in agents}
    slacks = {agent.name: 0 for agent in agents}

    # Optimization
    env = grb.Env()
    # Init model
    model = grb.Model(env=env)
    pCharge = {}
    for agent in agents:
        # Starting SOC
        eBatStart = SOCStart * agent.capacity

        slack, mxEbatEnd = getSlack(agent=agent, events=agent.events, eBatStart=eBatStart, etaTimesDelta=etaTimesDelta)
        slacks[agent.name] = sum(slack.values())

        # Model bounds and constraints
        pChargeAgent, _ = setAgentConstr(model=model, agent=agent, events=agent.events, slack=slack,
                                         mxEbatEnd=mxEbatEnd, eBatStart=eBatStart, etaTimesDelta=etaTimesDelta,
                                         eBatGoal=eBatStart)

        # Add pCharge var to set of fleet
        for t in range(nTimesteps):
            pCharge[t, agent.name] = pChargeAgent.get(t, 0)

    # Objective
    linCoefs = [2 * i for i in residualLoad]
    objective = grb.QuadExpr()
    for agent in agents:
        for event in agent.events:
            for t in range(event.start, event.stop):
                objective.addTerms(linCoefs[t], pCharge[t, agent.name])
    objective.add(grb.quicksum(grb.quicksum(pCharge[t, agent.name] for agent in agents)*grb.quicksum(pCharge[t, agent.name] for agent in agents) for t in range(nTimesteps)))
    model.ModelSense = grb.GRB.MINIMIZE
    model.setObjective(objective)

    # Calculation
    model.optimize()

    # Solution
    if model.status == 13:
        print(f"Solution suboptimal! Agent ID: {agent.name}")
    elif model.status > 2:
        raise AssertionError(f"Model terminated! Status: {model.status}")
    
    # Solution
    for agent in agents:
        for event in agent.events:
            for t in range(event.start, event.stop):
                p = pCharge[t, agent.name].X
                if p > 0:
                    demands[agent.name][t] = p
    return demands, slacks

# Approximation see paper
def optimizeChargingQP_smpl(*, agents, residualLoad, eta, SOCStart, deltaT, segments=False, verbose=True):
    # Repeated calculations
    etaTimesDelta = eta * deltaT

    # Solution
    demands = {agent.name: {} for agent in agents}
    slacks = {agent.name: 0 for agent in agents}
    
    # Sort agents by capacity.
    # First solving more restricted agents -> smaller obj val
    agents = sorted(agents, key=lambda x: x.capacity)

    # Optimization
    cnt = 0 # For progressbar
    env = grb.Env()

    for agent in agents:
        # Starting SOC
        eBatStart = SOCStart * agent.capacity
        # End SOC of every horizon
        eBatGoal = eBatStart

        # Segment events
        if segments == False:
            segEvents = [agent.events]
        else:
            segEvents = getSegmentEvents(agent.events, segments)
        for events in segEvents:
            # Init model
            model = grb.Model(env=env)

            # Get minimal slack
            slack, mxEbatEnd = getSlack(agent=agent, events=events, eBatStart=eBatStart, etaTimesDelta=etaTimesDelta)
            slacks[agent.name] += sum(slack.values())

            # Model bounds and constraints
            pCharge, endEBat = setAgentConstr(model=model, agent=agent, events=events, slack=slack, mxEbatEnd=mxEbatEnd,
                                              eBatStart=eBatStart, etaTimesDelta=etaTimesDelta, eBatGoal=eBatGoal)

            # Objective
            linCoefs = [2 * i for i in residualLoad]
            objective = grb.QuadExpr()
            ts = []
            for event in events:
                for t in range(event.start, event.stop):
                    objective.addTerms(linCoefs[t], pCharge[t])
                    ts.append(t)
            objective.add(grb.quicksum(pCharge[t]*pCharge[t] for t in ts))
            model.ModelSense = grb.GRB.MINIMIZE
            model.setObjective(objective)

            # Calculation
            model.optimize()

            # Check Success
            if model.status == 13:
                print(f"Solution suboptimal! Agent ID: {agent.name}")
            elif model.status > 2:
                raise AssertionError(f"Model terminated! Status: {model.status}, Agent ID: {agent.name}")
            
            # Solution
            for event in events:
                for t in range(event.start, event.stop):
                    p = pCharge[t].X
                    if p > 0:
                        demands[agent.name][t] = p

            # Update residual load
            for t, p in demands[agent.name].items():
                residualLoad[t] += p

            # End SOC
            eBatStart = endEBat.X

        # Progressbar
        if verbose:
            cnt += 1
            tools.tickProg(cnt/len(agents))

    return demands, slacks

# Uncontrolled charging
def immediate(*, agents, eta, SOCStart, deltaT, verbose=True):
    # Repeated calculations -> Energy to power
    etaTimesDelta = deltaT * eta

    # Solution
    demands = {agent.name: {} for agent in agents}
    slacks = {agent.name: 0 for agent in agents}

    cnt = 0 # For progressbar
    for agent in agents:
        # Starting condition
        eBatCurrently = agent.capacity * SOCStart

        # Simulation loop
        for event in agent.events:
            pMAx = event.pMax
            
            for t in range(event.start, event.stop):
                # Charge maximum possible
                eCharge = max(0, min(pMAx * etaTimesDelta, agent.capacity - eBatCurrently))
                # Charged energy to battery
                eBatCurrently += eCharge
                # Remember power
                if eCharge > 0:
                    demands[agent.name][t] = eCharge / etaTimesDelta
            # Consumption between charging events
            eBatCurrently -= event.consumption
            # Slack
            if eBatCurrently < 0:
                slacks[agent.name] += abs(eBatCurrently)
                eBatCurrently = 0

        # Progressbar
        if verbose:
            cnt += 1
            tools.tickProg(cnt/len(agents))
    return demands, slacks
