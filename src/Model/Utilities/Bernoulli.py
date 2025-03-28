import numpy as np
import math as m
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

from .ChemicalProperties import ChemicalProperties

class BernoulliModel(ChemicalProperties):
    """
    ##### ##### ##### ##### ##### ##### ##### ##### #####
    ## This class outlines the entire Blowdown Model   ##
    #   Governing Equations                             #
    #       - Ideal Gas Law and Equation of State       #
    #       - Conversation of Mass                      #
    #       - Bernoulli's Compressible Flow             #
    ##### ##### ##### ##### ##### ##### ##### ##### #####
    """
    def __init__(self, inputs):
        """
        # Purpose:  Initiates the class
        # Inputs:   inputs - The input dictionary from main.py
        #
        # Outputs:  self - Input Constants necessary to perform the calculations for Blowdown Model
        """

        self.A_inj = inputs["A_inj"]                                # Area of the Injector
        self.C_d = inputs["C_d"]                                    # Coefficient of Discharge of Injector
        self.Po_i = inputs["P_tank"]                                # Pressure of the Oxidizer Tank
        self.T_tank = inputs["T_tank"]                              # Temperature of the Oxidizer Tank
        self.m_N2O = inputs["m_N2O"]                                # Mass of Liquid Nitrous Oxide
        self.V_tank = inputs["V_tank"]                              # Volume of the Oxidizer Tank

        super().__init__(T_tank=self.T_tank, V_tank=self.V_tank, P_tank=self.Po_i, m_N2O=self.m_N2O)

        self.V_u_i = inputs["V_tank"] * self.ullageFraction         # Volume of the Ullage
        self.rho_ox = self.densityN2OLiquid(self.T_tank)            # Density of the Liquid Oxidizer

    def oxidizerMassDeriv(self, Po, P_c):
        """
        # Purpose:  Calculate the oxidizer mass flow
        # Inputs:   Po     - Current Oxidizer Tank Pressure
        #           P_c    - Current Combustion Chamber Pressure
        #
        # Outputs:  dmo_dt - Change in Massflow rate of liquid oxidizer with Time
        """

        # Take necessary constant values from init dunder function
        A_inj = self.A_inj
        C_d = self.C_d
        rho_ox = self.rho_ox

        # Define Derivative for Oxidizer Mass Flow from Bernoulli's Equation
        dmo_dt = -A_inj*C_d*np.sqrt(2*rho_ox*(Po - P_c))
        return dmo_dt

    def volDeriv(self, dmo_dt):
        """
        # Purpose:  Calculate the volume of the ullage in the oxidizer tank
        # Inputs:   dmo_dt - Change in Massflow rate of liquid oxidizer with Time
        #
        # Outputs:  dVu_dt - Change in ullage volume rate with Time
        """

        # Take necessary constant values from init dunder function
        rho_ox = self.rho_ox

        # Define Derivative of Ullage Volume from Oxidizer Mass Flow Rate
        dVu_dt = -dmo_dt/rho_ox

        return dVu_dt

    def pressureDeriv(self, Vu, dVu_dt):
        """
        # Purpose:  Calculate the change in the Oxidizer Tank Pressure
        # Inputs:   Vu     - Current volume of the ullage
        #           dVu_dt - Change in ullage Volume rate with Time
        #
        # Outputs:  dPo_dt - Change in oxidizer tank Pressure with Time
        """

        # Take necessary constant values from init dunder function
        Po_i = self.Po_i
        V_u_i = self.V_u_i

        # Define Derivative of Tank Pressure from Ideal Gas & constant Temperature assumption
        dPo_dt = -((Po_i * V_u_i)/Vu**2) * dVu_dt

        return dPo_dt
    
    def blowdownModel(self, Po, Pc, Vu):
        """
        # Final Wrapper that holds all of the parts of the blowdown model
        # Inputs:   Po    - Current Oxidizer Tank Pressure
        #           Pc    - Current Combustion Chamber Pressure
        #           Vu    - Current volume of the ullage
        #
        # Outputs: dmo_dt - Change in Mass Flow Rate of Oxidizer with Time
        #          dPo_dt - Change in Oxidizer Tank Pressure with Time
        #          dVu_dt - Change in Ullage Gas Volume with Time
        """

        dmo_dt = self.oxidizerMassDeriv(Po, Pc)
        dVu_dt = self.volDeriv(dmo_dt)
        dPo_dt = self.pressureDeriv(Vu, dVu_dt)

        return dmo_dt, dPo_dt, dVu_dt