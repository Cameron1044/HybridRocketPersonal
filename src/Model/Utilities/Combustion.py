import numpy as np
import math as m
import pandas as pd
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from scipy.interpolate import Rbf, griddata
from scipy.spatial import Delaunay
from .utilities import ToMetric, ToEnglish, plot_graph

class CombustionModel():
    """
    ##### ##### ##### ##### ##### ##### ##### ##### ##### ##### #####
    ## This class outlines the entire Combustion Chamber Model     ##
    #   Governing Equations                                         #
    #       - Ideal Gas Law and Equation of State                   #
    ##       - Conversation of Mass                                ##
    ##### ##### ##### ##### ##### ##### ##### ##### ##### ##### #####
    """
    def __init__(self, inputs):
        """
        # Purpose:  Initiates the class
        # Inputs:   inputs - The input dictionary from main.py
        #
        # Outputs:  self - Input Constants necessary to perform the calculations for Combustion Model
        """
        self.inputs = inputs
        self.a = inputs["a"]                    # Burn Rate Coefficient
        self.n = inputs["n"]                    # Burn Rate Coefficient
        if self.inputs["GeometryType"] == "Helical":
            self.L = inputs["HelixLength"]               # Length of the Fuel Grain
        else:
            self.L = inputs["L_fuel"]               # Length of the Fuel Grain
        self.Ru = inputs["Ru"]                  # Universal Gas Constant
        self.rho_f = inputs["rho_fuel"]         # Density of the Fuel Grain
        self.Pe = inputs["P_amb"]               # Ambient Pressure, fine until Summerfield Condition
        self.A_t = np.pi*(inputs["d_t"]/2)**2   # Area of the Nozzle Throat
        self.alpha = inputs["alpha"]            # Nozzle divergent half-cone angle
        self.M_chmb = inputs["M_chmb"]
        self.T_chmb = inputs["T_chmb"]
        self.gamma = inputs["gamma"]

        self.geodf = pd.read_csv('src/Model/CSV/burnback_table.csv')
        self.PeoPc = self.Pe/950                        # Calculate constant, choked value of Pe/Pc for representative chamber pressure
        self.nozzlelambda = ((1+np.cos(self.alpha)) / 2 ) # Calculate nozzle correction factor

    def OFratio(self, dm_ox, dm_f):
        """
        # Purpose:  Calculate the Oxidizer to Fuel Mass Flow Ratio
        # Inputs:   dm_ox - Current Oxidizer Mass Flow Rate
        #           dm_f  - Current Fuel Mass Flow Rate
        #
        # Outputs:  OF    - Current Oxidizer to Fuel Mass Flow Ratio
        """
        self.OF = dm_ox / dm_f

    def fuelGeometry(self, r):
        # find index of closest r value in df
        self.geodf = self.geodf.sort_values('r')
        idx_below = self.geodf['r'].searchsorted(r, side='right') - 1
        idx_above = idx_below + 1

        # return value if r is in table and doesn't need interpolation
        if idx_below >= 0 and self.geodf.iloc[idx_below]['r'] == r:
            return self.geodf.iloc[idx_below]['area'], self.geodf.iloc[idx_below]['perimeter']

        # handle for if r is outside of the table. Grab the first or last value instead
        if idx_below < 0:
            idx_below = 0
            return self.geodf.iloc[idx_below]['area'], self.geodf.iloc[idx_below]['perimeter']
        if idx_above >= len(self.geodf):
            idx_above = len(self.geodf) - 1
            return self.geodf.iloc[idx_above]['area'], self.geodf.iloc[idx_above]['perimeter']

        # grab values for linear interpolation
        r_below = self.geodf.iloc[idx_below]['r']
        r_above = self.geodf.iloc[idx_above]['r']
        area_below = self.geodf.iloc[idx_below]['area']
        area_above = self.geodf.iloc[idx_above]['area']
        perimeter_below = self.geodf.iloc[idx_below]['perimeter']
        perimeter_above = self.geodf.iloc[idx_above]['perimeter']

        # linear interpolation
        t = (r - r_below) / (r_above - r_below)
        area = area_below + t * (area_above - area_below)
        perimeter = perimeter_below + t * (perimeter_above - perimeter_below)
        return area, perimeter

    def fuelGrainState(self, r, Pc):
        """
        # Purpose:  Calculate the current state of the fuel grain in the combustion chamber
        # Inputs:   r        - The current Radius of the fuel grain port
        #           Pc       - The current Combustion Chamber Pressure
        #
        # Outputs:  A_p      - Area of the fuel grain port
        #           A_b      - Surface Area of the burnable fuel grain 
        #           V_chmb   - Volume of the Combustion Chamber Products
        #           rho_chmb - Density of the Combustion Chamber Products
        #           R_prime  - Molecular Gas Constant of the Combustion Chamber Products
        """

        # Take necessary constant valus from init dunder function
        Ru = self.Ru
        L = self.L
        M_chmb = self.M_chmb
        T_chmb = self.T_chmb

        # Calculate the current state of the combustion chamber and fuel grain
        R_prime = Ru / M_chmb

        if self.inputs["GeometryType"] == "Cylindrical":
            A_p = np.pi * r**2
            A_b = 2 * np.pi * r * L * 2
        elif self.inputs["GeometryType"] == "Custom":
            area, perimeter = self.fuelGeometry(r)
            A_p = area
            A_b = perimeter * L
        elif self.inputs["GeometryType"] == "Helical":
            A_p = np.pi * r**2
            A_b = np.pi * r*2 * L

        V_chmb = A_p * L
        rho_chmb = Pc / (R_prime * T_chmb)
        return A_p, A_b, V_chmb, rho_chmb, R_prime

    def fuelRegression(self, dmo_dt, A_p):
        """
        # Purpose:  Calculate the rate at which the fuel grain burns
        # Inputs:   dmo_dt - Current oxidizer mass flow rate
        #           A_p    - Current area of the fuel grain port
        #
        # Outputs:  dr_dt  - Change in fuel grain port Radius with Time
        """

        # Take necessary constant values from init dunder function
        a = self.a
        n = self.n

        # Define Derivative for Fuel Grain port radius from Marxman Fuel Regression
        dr_dt = a * (-dmo_dt / A_p)**n
        if self.inputs["GeometryType"] == "Helical":
            dr_dt = dr_dt * self.inputs["HelixRegressionAmplification"]
        return dr_dt

    def mfDeriv(self, dr_dt, A_b):
        """
        # Purpose:  Calculate the mass flow of the fuel
        # Inputs:   dr_dt  - Current rate at which the fuel grain port Radius changes
        #           A_b    - Current surface area of burnable fuel grain
        #
        # Outputs:  dmf_dt - Change in fuel mass flow rate with Time
        """

        # Take necessary constant values from init dunder function
        rho_f = self.rho_f
        
        # Define Derivative for Fuel Mass Flow from Marxman Fuel Regression
        dmf_dt = -A_b * rho_f * dr_dt
        return dmf_dt

    def PChmbDeriv(self, A_b, V_chmb, rho_chmb, R_prime, dr_dt, dmo_dt, Pc):
        """
        # Purpose:  Calculate the chamber pressure
        # Inputs:   A_b      - Current surface area of burnable fuel grain
        #           V_chmb   - Volume of the Combustion Chamber Products
        #           rho_chmb - Density of the Combustion Chamber Products
        #           R_prime  - Molecular Gas Constant of the Combustion Chamber Products
        #           dr_dt    - Change in fuel grain port Radius with Time
        #           dmo_dt   - Current oxidizer mass flow rate
        #           Pc       - Current Combustion Chamber Pressure
        #
        # Outputs:  dPc_dt   - Change in Combustion Chamber Pressure with Time
        """

        # Take necessary constant values from init dunder function
        rho_f = self.rho_f
        A_t = self.A_t
        gam = self.gamma
        T_chmb = self.T_chmb

        # Calculating Fuel Mass relations from gas generation and fuel regression
        FuelMassRelation = A_b * dr_dt * (rho_f - rho_chmb)
        # Calculating mass flow through a Nozzle
        NozzleMass = Pc * A_t * np.sqrt(gam / (R_prime * T_chmb)) * (2 / (gam + 1))**((gam + 1)/(2 * (gam - 1))); 
        # Define Derivative for Oxidizer Mass Flow from Bernoulli's Equation
        dPc_dt = (R_prime * T_chmb/V_chmb) * (FuelMassRelation - dmo_dt - NozzleMass);   
        return dPc_dt

    def combustionModel(self, dmo_dt, Pc, r):
        """
        # Final Wrapper that holds all of the parts of the Combustion model
        # Inputs:  dmo_dt - Current Oxidizer Mass Flow Rate
        #           Pc    - Current Combustion Chamber Pressure
        #           r     - Current Fuel Grain Port Radius
        # Outputs: dr_dt  - Change in Fuel Grain Port Radius with Time
        #          dmf_dt - Change in Fuel Mass Flow  with Time
        #          dPc_dt - Change in Combustion Chamber Pressure with Time
        #          F      - The Force produced
        """
        
        A_p, A_b, V_chmb, rho_chmb, R_prime = self.fuelGrainState(r, Pc)
        dr_dt = self.fuelRegression(dmo_dt, A_p)
        dmf_dt = self.mfDeriv(dr_dt, A_b)
        self.OFratio(dmo_dt, dmf_dt)
        dPc_dt = self.PChmbDeriv(A_b, V_chmb, rho_chmb, R_prime, dr_dt, dmo_dt, Pc)        
        PeoPc = self.Pe / ToMetric(750, 'psi') # Constant, choked value of Pe/Pc for representative chamber pressure
        
        F = self.nozzlelambda * self.A_t * Pc * np.sqrt( ((2*self.gamma**2)/(self.gamma-1)) * (2/(self.gamma+1))**((self.gamma+1)/(self.gamma-1)) * (1-(PeoPc)**((self.gamma-1)/self.gamma)) )
        return dr_dt, dmf_dt, dPc_dt, F, self.OF, self.T_chmb, self.M_chmb, self.gamma 