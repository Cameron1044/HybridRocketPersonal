import numpy as np
import math as m
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

from .ChemicalProperties import ChemicalProperties

class ZKmodel(ChemicalProperties):
    def __init__(self, inputs):
        self.A_inj = inputs["A_inj"]    # Area of the Injector [m^2]
        self.C_d = inputs["C_d"]        # Coefficient of Discharge of Injector [unitless]
        self.P_tank = inputs["P_tank"]  # Pressure of the Oxidizer Tank [Pa]
        self.V_tank = inputs["V_tank"]  # Volume of the Oxidizer Tank [m^3]
        self.T_tank = inputs["T_tank"]  # Initial temperature [K]
        self.m_T = inputs["m_T"]        # Total mass of tank [kg]
        self.m_N2O = inputs["m_N2O"]    # Mass of loaded N2O [kg]
        super().__init__(T_tank=self.T_tank, V_tank=self.V_tank, P_tank=self.P_tank, m_N2O=self.m_N2O)
    
    def ZKModel(self, To, n_go, n_lo, P_chmb):
        Vhat_l = self.molarVolumeN2O(To)                # molar volume of liquid N2O [m**3/kmol]
        CVhat_Ar = self.specificHeatArgon(To)           # heat capacity of Ar at constant pressure [J/(kmol*K)]
        CVhat_g = self.specificHeatN2OGas(To)           # heat capacity of N2O gas at constant pressure [J/(kmol*K)]
        CVhat_l = self.specificHeatN2OLiquid(To)        # heat capacity of N2O liquid at constant pressure [J/(kmol*K)]
        delta_Hv = self.heatVaporizationN2O(To)         # heat of vaporization of N2O [J/kmol]
        P_sat = self.vaporPressureN2O(To)               # vapor pressure of N20 [Pa]
        dP_sat = self.vaporPressureN2ODerivative(To)    # derivative of vapor pressure of N20 [Pa/K]
        Cp_T = self.specificHeatAluminum(To)            # heat capacity of aluminum [J/(kg*K)]

        nT = n_go + n_lo
        #print percent of gas and liquid
        # print(n_go/nT*100, n_lo/nT*100)
        
        # Oxidizer Tank Pressure
        Po = self.tankPressure(n_lo, n_go, self.n_Ar, To, self.V_tank)

        # Derivative Equations
        a = self.m_T*Cp_T + self.n_Ar*CVhat_Ar + n_go*CVhat_g + n_lo*CVhat_l
        b = Po*Vhat_l
        e = -delta_Hv + self.R*To
        f = -self.C_d*self.A_inj*np.sqrt(2/self.MW_N2O)*np.sqrt((Po-P_chmb)/Vhat_l)
        j = -Vhat_l*P_sat
        k = (self.V_tank - n_lo*Vhat_l)*dP_sat
        m = self.R*To
        q = self.R*n_go
        
        # Derivative Functions
        Z = (-f*(-j*a + (q-k)*b)) / (a*(m+j) + (q-k)*(e-b))
        W = (-Z*(m*a + (q-k)*e)) / (-j*a + (q-k)*b)
        dT = (b*W+e*Z)/a
        dn_g = Z
        dn_l = W

        return dT, dn_g, dn_l, Po
