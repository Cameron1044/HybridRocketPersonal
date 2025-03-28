import numpy as np
import math as m
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import pandas as pd

from .Bernoulli import BernoulliModel
from .ZK import ZKmodel
from .Combustion import CombustionModel
from .utilities import ToEnglish, ToMetric

class Model():
    """
    Model class provides the core functionality for simulating the performance of the rocket,
    considering various factors such as combustion, blowdown, pressure differences, and more.
    """

    def __init__(self, initialInputs, ZK=True, iterations=1000, tspan=[0, 20]):
        """
        Initialize the Model class.

        Parameters:
        - initialInputs (dict): Input parameters provided by the user or main application.
        - ZK (bool): Flag to select between the ZK model or Bernoulli model for blowdown.
        - iterations (int): Number of data points across the specified time span.
        - tspan (list): Start and finish time for iterations.

        Returns:
        None.
        """
        self.initialInputs = initialInputs  # Initial Input Dictionary
        self.iterations = iterations        # Number of Iterations
        self.tspan = tspan                  # Time Span
        self.ZK = ZK                        # Flag for ZK Model
        self.initialValues()                # Calculate Initial Values

    def initialValues(self):
        """
        Calculate and set initial conditions based on provided inputs.

        Parameters:
        None.

        Returns:
        None.
        """
        # Initialize combustion model using initial inputs
        self.combustion = CombustionModel(self.initialInputs)

        if self.initialInputs["GeometryType"] == "Cylindrical":
            mf_i = self.initialInputs['rho_fuel'] * np.pi * self.initialInputs['L_fuel'] * (self.initialInputs['OD_fuel']**2 - self.initialInputs['ID_fuel']**2) / 4
            r_i = self.initialInputs['ID_fuel'] / 2         # Initial Radius of the Fuel Grain Port [m]
        elif self.initialInputs["GeometryType"] == "Custom":
            geodf = pd.read_csv('src/Regression/burnback_table.csv')
            cross_section_area = geodf['area'].iloc[0]
            circle_area = np.pi * (self.initialInputs['OD_fuel']**2) / 4
            mf_i = self.initialInputs['rho_fuel'] * self.initialInputs['L_fuel'] * (circle_area - cross_section_area)
            self.r_final = geodf['r'].iloc[-1]
            r_i = 0
        elif self.initialInputs["GeometryType"] == "Helical":
            mf_i = self.initialInputs['rho_fuel'] * self.initialInputs['HelixVolume']
            r_i = self.initialInputs['HelixPortDiameter'] / 2         # Initial Radius of the Fuel Grain Port [m]

        if self.ZK:
            # Using the ZK model for blowdown
            self.blowdown = ZKmodel(self.initialInputs)

            # Set initial conditions
            Ti = self.blowdown.T_tank                       # Initial Temperature of the Oxidizer Tank [K]
            ng_i = self.blowdown.n_go                       # Initial Number of Gas Molecules [kmol]
            nl_i = self.blowdown.n_lo                       # Initial Number of Liquid Molecules [kmol]
            Pc_i = self.initialInputs["P_int"]              # Initial Pressure of the Combustion Chamber [Pa]
            self.mt_i = nl_i * self.blowdown.MW_N2O + mf_i  # Initial Total Mass of the Fuel + Oxidizer [kg]
            self.y0 = [Ti, ng_i, nl_i, mf_i, Pc_i, r_i, 0]  # Initial Conditions for the ODE45 Solver
        else:
            # Using the Bernoulli model for blowdown
            self.blowdown = BernoulliModel(self.initialInputs)
            
            # For calculations of initial mass of liquid oxidizer
            liquidOxidizerFraction = (1 - self.blowdown.ullageFraction)
            densityN2OLiquid = self.blowdown.densityN2OLiquid(self.initialInputs["T_tank"])
            # Set other initial conditions
            mo_i = self.blowdown.V_tank * liquidOxidizerFraction * densityN2OLiquid # Initial Mass of Liquid Oxidizer [kg]
            Po_i = self.blowdown.P_tank                                             # Initial Pressure of the Oxidizer Tank [Pa]
            Pc_i = self.initialInputs["P_int"]                                      # Initial Pressure of the Combustion Chamber [Pa]
            Vu_i = self.blowdown.V_tank * self.blowdown.ullageFraction              # Initial Volume of the Ullage [m^3]
            self.mt_i = mo_i + mf_i                                                 # Initial Total Mass of the Fuel + Oxidizer [kg]
            self.y0 = [mo_i, mf_i, Po_i, Pc_i, Vu_i, r_i, 0]                        # Initial Conditions for the ODE45 Solver

    def termination_event_radius(self, t, y):
        """
        Termination event to stop the integration when port radius exceeds half of the fuel outer diameter.

        Parameters:
        - t (float): Time.
        - y (list): Current state values.

        Returns:
        float: Difference between current radius and half of the fuel outer diameter.
        """
        r = y[5]
        if self.initialInputs["GeometryType"] == "Cylindrical":
            return r - self.initialInputs['OD_fuel'] / 2
        elif self.initialInputs["GeometryType"] == "Custom":
            return r - self.r_final
        elif self.initialInputs["GeometryType"] == "Helical":
            return r - self.initialInputs['OD_fuel'] / 2

    termination_event_radius.terminal = True

    def termination_event_oxidizer_mass(self, t, y):
        """
        Termination event to stop the integration when oxidizer mass reaches zero.

        Parameters:
        - t (float): Time.
        - y (list): Current state values.

        Returns:
        float: Current oxidizer mass.
        """
        m_o = y[2] if self.ZK else y[0]
        return m_o

    termination_event_oxidizer_mass.terminal = True

    def termination_event_pressure_difference(self, t, y):
        """
        Termination event to stop the integration when chamber pressure exceeds oxidizer tank pressure.

        Parameters:
        - t (float): Time.
        - y (list): Current state values.

        Returns:
        float: Difference between chamber pressure and oxidizer tank pressure.
        """
        if self.ZK:
            To, ng, nl, Pc = y[0], y[1], y[2], y[4]
            _, _, _, P_o = self.blowdown.ZKModel(To, ng, nl, Pc)
        else:
            Pc, P_o = y[3], y[2]
        return Pc - P_o

    termination_event_pressure_difference.terminal = True

    def model_zk(self, t, To, ng, nl, mf, Pc, r, I):
        """
        Initiates the ZK Model which wraps around the different Hybrid rocket Models.

        Parameters:
        - To (float): Temperature.
        - ng (float): Number of gas molecules.
        - nl (float): Number of liquid molecules.
        - mf (float): Mass of fuel.
        - Pc (float): Chamber pressure.
        - r (float): Radius.
        - I (float): Impulse.

        Returns:
        tuple: Differentiated values of temperature, gas molecules, liquid molecules, fuel mass, pressure, radius, and thrust.
        """

        dTo, dng_dt, dnl_dt, Po = self.blowdown.ZKModel(To, ng, nl, Pc)
        dmo_dt = dnl_dt * self.blowdown.MW_N2O
        if self.initialInputs['ColdFlow']:
            dr_dt, dmf_dt, dPc_dt, T, OF = 0, 0, 0, 0, 0
            T_chmb, M_chmb, gamma = self.combustion.T_chmb, self.combustion.M_chmb, self.combustion.gamma
        else:
            dr_dt, dmf_dt, dPc_dt, T, OF, T_chmb, M_chmb, gamma = self.combustion.combustionModel(dmo_dt, Pc, r)
        return dTo, dng_dt, dnl_dt, dmf_dt, dPc_dt, dr_dt, T, Po, OF, T_chmb, M_chmb, gamma

    def model(self, t, mo, mf, Po, Pc, Vu, r, I):
        """
        Initiates the Bernoulli Model which wraps around the different Hybrid rocket Models.

        Parameters:
        - mo (float): Mass of oxidizer.
        - mf (float): Mass of fuel.
        - Po (float): Oxidizer pressure.
        - Pc (float): Chamber pressure.
        - Vu (float): Ullage volume.
        - r (float): Radius.
        - I (float): Impulse.

        Returns:
        tuple: Differentiated values for oxidizer mass, fuel mass, oxidizer pressure, chamber pressure, ullage volume, radius, and thrust.
        """
    
        dmo_dt, dPo_dt, dVu_dt = self.blowdown.blowdownModel(Po, Pc, Vu)
        if self.initialInputs['ColdFlow']:
            dr_dt, dmf_dt, dPc_dt, T, OF = 0, 0, 0, 0, 0
            T_chmb, M_chmb, gamma = self.combustion.T_chmb, self.combustion.M_chmb, self.combustion.gamma
        else:
            dr_dt, dmf_dt, dPc_dt, T, OF, T_chmb, M_chmb, gamma = self.combustion.combustionModel(dmo_dt, Pc, r)
        return dmo_dt, dmf_dt, dPo_dt, dPc_dt, dVu_dt, dr_dt, T, OF, T_chmb, M_chmb, gamma

    def ode_function(self, t, y):
        """
        ODE function for the Bernoulli model.

        Parameters:
        - t (float): Time.
        - y (list): Current state values.

        Returns:
        list: List of differentiated values.
        """
        mo, mf, Po, Pc, Vu, r, I = y
        return list(self.model(t, mo, mf, Po, Pc, Vu, r, I)[:-4])

    def ode_function_zk(self, t, y):
        """
        ODE function for the ZK model.

        Parameters:
        - t (float): Time.
        - y (list): Current state values.

        Returns:
        list: List of differentiated values.
        """
        To, ng, nl, mf, Pc, r, I = y
        return list(self.model_zk(t, To, ng, nl, mf, Pc, r, I)[:-5])

    def ODE45(self):
        """
        Solve the ODEs using scipy's `solve_ivp` and return results in a dataframe.

        Parameters:
        None.

        Returns:
        pd.DataFrame: Results containing information about thrust, impulse, pressures, and more.
        """
        # define data columns and initialize the dataframe
        data_columns = [
            "time", "thrust", "thrustN", "impulse", "Pc", "Pox", "mox", "mf",
            "dmox", "dmf", "OF", "r", "dm_total_dt", "cstar", "T_chmb", "M_chmb", "gamma"
        ]
        self.df = pd.DataFrame(columns=data_columns)

        # determine the model function based on ZK flag
        ode_func = self.ode_function_zk if self.ZK else self.ode_function

        # list of events to monitor during ODE integration
        events = [
            self.termination_event_radius,
            self.termination_event_oxidizer_mass,
            self.termination_event_pressure_difference,
        ]

        # solve ODE
        sol = solve_ivp(
            ode_func, self.tspan, self.y0,
            t_eval=np.linspace(self.tspan[0], self.tspan[1], self.iterations),
            events=events
        )

        # print termination event messages
        name = 'ZK:' if self.ZK else 'Bernoulli:'
        if sol.t_events[0].size > 0:
            print(f"{name} Integration terminated because port radius exceeded half of the fuel outer diameter.")
        elif sol.t_events[1].size > 0:
            print(f"{name} Integration terminated because oxidizer mass reached zero.")
        elif sol.t_events[2].size > 0:
            print(f"{name} Integration terminated because chamber pressure exceeded oxidizer tank pressure.")

        return self.getResults(sol)

    def getResults(self, sol):
        for i, ti in enumerate(sol.t):
            if self.ZK:
                dTo, dng_dt, dnl_dt, dmf_dt, dPc_dt, dr_dt, T, Po, OF, T_chmb, M_chmb, gamma = self.model_zk(sol.t[i],*sol.y[:, i])
                dmo_dt = -dnl_dt * self.blowdown.MW_N2O
                dmf_dt = -dmf_dt
                impulse = sol.y[6, i]
                Pc = sol.y[4, i]
                mox = sol.y[2, i] * self.blowdown.MW_N2O
                mf = sol.y[3, i]
                r = sol.y[5, i]
            else:
                dmo_dt, dmf_dt, dPo_dt, dPc_dt, dVu_dt, dr_dt, T, OF, T_chmb, M_chmb, gamma = self.model(sol.t[i],*sol.y[:, i])
                dmo_dt = -dmo_dt
                dmf_dt = -dmf_dt
                impulse = sol.y[6, i]
                Po = sol.y[2, i]
                Pc = sol.y[3, i]
                mox = sol.y[0, i]
                mf = sol.y[1, i]
                r = sol.y[5, i]

            # process data to English units
            new_data = {
                "time": ti,
                "thrust": ToEnglish(T, 'N'),
                "thrustN": T,
                "impulse": ToEnglish(impulse, 'N'),
                "Pc": ToEnglish(Pc, 'Pa'),
                "Pox": ToEnglish(Po, 'Pa'),
                "mox": ToEnglish(mox, 'kg'),
                "mf": ToEnglish(mf, 'kg'),
                "dmox": ToEnglish(abs(dmo_dt), 'kg'),
                "dmf": ToEnglish(abs(dmf_dt), 'kg'),
                "OF": ToEnglish(OF, 'unitless'),
                "r": ToEnglish(r, 'm'),
                "dm_total_dt": ToEnglish(abs(dmo_dt + dmf_dt), 'kg'),
                "cstar": ToEnglish(T, 'N') / ToEnglish(abs(dmo_dt + dmf_dt), 'kg'),
                "T_chmb": T_chmb,
                "M_chmb": M_chmb,
                "gamma": gamma,
            }
            
            # append the data to dataframe
            self.df.loc[len(self.df)] = new_data
        return self.df


