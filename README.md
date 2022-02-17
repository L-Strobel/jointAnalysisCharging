# Charging Model used in "Joint Analysis of Regional and National Power System Impacts of Electric Vehicles - A Case Study for Germany on the County Level in 2030"

For a detailed explanation of the methodology, see Chapter 3.5 in the paper.

The model is designed to calculate the uncontrolled and optimal load for a fleet of electric vehicles. Optimal load refers to the load that minimizes the sum of squares of the resulting netload. The residual load is expected to be separated into regions (in the paper counties) which can be optimized individually (Opt_County) or combined (Opt_National).

In theory, the model can be utilized for any region and corresponding subregions; however, it was created with a specific case study in mind. Therefore, some aspects are currently hard-coded (E.g., time step length (15min), time period (2030), number of agents per region (100))

# Input
The input is expected in *chargingmodel/input* and should be in the same form as the given dummy tables. Thus, same header, same time window (2030), same time step (15min), same separator (";").

The number of regions is flexible. Only the regions with a name corresponding to an ID in *regions.csv* will be considered in the model. The column *RegisteredCars* in this csv refers to the number of total cars registered in the region. The number of EVs allocated to each region is propotional to this number.

The dummy residual load inputs are generated by adding multiple sinus curves and Gaussian noise. They do not represent real-world data. For example, you can acquire suitable real-world data from https://transparency.entsoe.eu/ for different EU countries.

The dummy behavior profiles are generated with the mobility model described in the paper but without defining the agents' personal characteristics like age or employment status. The underlying probability distributions are created from the data of "Mobilität in Deutschland 2017" (http://www.mobilitaet-in-deutschland.de/). The meaning of the "Purpose" column derives from this dataset and is mapped to locations as follows:

    Purpose -> Next parking location
    0 -> WORK,
    1 -> PUBLIC
    2 -> PUBLIC
    3 -> PUBLIC
    4 -> PUBLIC
    5 -> PUBLIC
    6 -> PUBLIC
    7 -> HOME
    8 -> HOME
    9 -> PUBLIC

## Configuration Options:

*config.json* can be adjusted to introduce new EVSE availability scenarios, fleet compositions, and fleet sizes.

# Output:
The model output is stored in SQlite databases. They can be browsed with the "DB Browser for SQLite". Additionally, the package provides some processing options in *postprocessing.py* (see *example.ipynb*).

# Get started:

1. Install the package. Run (from here):
    ```
    pip install -e .
    ```

2. Install Gurobi and obtain a licence (https://www.gurobi.com/)

3. Run the model: see *example.ipynb* (Additional requirements: JupyterLab and Matplotlib)
