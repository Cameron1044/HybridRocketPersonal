# Import necessary libraries
import numpy as np
import pandas as pd

class ChemicalProperties():
    def __init__(self, T_tank=0, V_tank=0, P_tank=0, m_N2O=0):
        self.T_tank = T_tank # initial temperature [K]
        self.V_tank = V_tank # volume of tank [m**3]
        self.P_tank = P_tank # initial pressure [Pa]
        self.m_N2O = m_N2O # mass of loaded N2O [kg]

        self.MW_N2O = 44.013 # molecular weight of N2O
        self.Tc = 309.57 # critical temperature of N2O [K]
        self.R = 8314.3 # universal gas constant [J/(kmol*K)]
    
        self.C1 = 0.2079e5 # heat capacity of He AND argon at constant pressure [J/(kmol*K)] coefficients
        self.C2 = 0 # valid for Temp range [100 K - 1500 K]
        self.C3 = 0
        self.C4 = 0
        self.C5 = 0

        self.G1 = 96.512  # vapor pressure of N2O [Pa] coefficients
        self.G2 = -4045  # valid for Temp range [182.3 K - 309.57 K]
        self.G3 = -12.277
        self.G4 = 2.886e-5
        self.G5 = 2

        self.Q1 = None  # molar specific volume of liquid N2O [m**3/kmol] coefficients
        self.Q2 = None
        self.Q3 = None
        self.Q4 = None
        self.findQ(self.P_tank)

        self.J1 = 2.3215e7 # heat of vaporization of N2O [J/kmol] coefficients
        self.J2 = 0.384 # valid for Temp range [182.3 K - 309.57 K]
        self.J3 = 0
        self.J4 = 0

        self.D1 = 0.2934e5 # heat capacity of N2O gas at constant pressure [J/(kmol*K)] coefficients
        self.D2 = 0.3236e5 # valid for Temp range [100 K - 1500 K]
        self.D3 = 1.1238e3
        self.D4 = 0.2177e5
        self.D5 = 479.4

        self.E1 = 6.7556e4 # heat capacity of N2O liquid at constant pressure [J/(kmol*K)] coefficients
        self.E2 = 5.4373e1 # valid for Temp range [182.3 K - 200 K]
        self.E3 = 0
        self.E4 = 0
        self.E5 = 0

        self.n_go, self.n_lo, self.n_Ar = self.initialMoles()
        self.ullageFraction = self.ullageFractionFunc()
    
    def findQ(self, pressure):
        pressure = pressure/6895  # convert pressure from Pa to PSI
        #round pressure to nearest whole number
        pressure = round(pressure)
        try:
            q_values_df = pd.read_csv('src/Model/NistDataProc/data/Q_values.csv')
        except FileNotFoundError:
            raise FileNotFoundError("The Q values file could not be found.")
        
        if pressure in q_values_df['Pressure'].values:
            params = q_values_df[q_values_df['Pressure'] == pressure].iloc[0]
        else:
            closest_pressure = q_values_df.iloc[(q_values_df['Pressure'] - pressure).abs().argsort()[:1]]
            print(f"Taking the closest pressure value of {closest_pressure['Pressure'].values[0]}.")
            params = closest_pressure.iloc[0]
        self.Q1 = params['Q1']
        self.Q2 = params['Q2']
        self.Q3 = params['Q3']
        self.Q4 = params['Q4']

    def molarVolumeN2O(self, T):
        Vhat_li = self.Q2**(1 + (1 - T / self.Q3)**self.Q4) / self.Q1  # molar volume of liquid N2O [m**3/kmol]
        return Vhat_li
    
    def densityN2OLiquid(self, T):
        rho_li = self.MW_N2O / self.molarVolumeN2O(T)
        return rho_li
    
    # y = 44.013/(Q2**(1 + (1 - x/Q3)**Q4) / Q1)
    
    def vaporPressureN2O(self, T):
        P_sat = np.exp(self.G1 + self.G2 / T + self.G3 * np.log(T) + self.G4 * T**self.G5) # vapor pressure of N20 [Pa]
        return P_sat
    
    def vaporPressureN2ODerivative(self, T):
        dP_sat = (-self.G2/(T**2) + self.G3/T + self.G4*self.G5*T**(self.G5-1)) * np.exp(self.G1 + self.G2/T + self.G3*np.log(T) + self.G4*T**self.G5)
        return dP_sat
    
    def specificHeatN2OGas(self, T):
        CVhat_g = self.D1 + self.D2*((self.D3/T)/np.sinh(self.D3/T))**2 + self.D4*((self.D5/T)/np.cosh(self.D5/T))**2 - self.R
        return CVhat_g
    
    def specificHeatN2OLiquid(self, T):
        CVhat_l = self.E1 + self.E2*T + self.E3*T**2 + self.E4*T**3 + self.E5*T**4
        return CVhat_l
    
    def specificHeatArgon(self, T):
        CVhat_Ar = self.C1 + self.C2*T + self.C3*T**2 + self.C4*T**3 + self.C5*T**4 - self.R
        CVhat_Ar = CVhat_Ar*2
        return CVhat_Ar
    
    def heatVaporizationN2O(self, T):
        Tr = T/self.Tc
        delta_Hv = self.J1*(1 - Tr) ** (self.J2 + self.J3*Tr + self.J4*Tr**2)
        return delta_Hv
    
    def specificHeatAluminum(self, T):
        Cp_T = (4.8 + 0.00322*T)*155.239
        return Cp_T
    
    def tankPressure(self, n_lo, n_go, n_Ar, T_tank, V_tank):
        Vhat_l = self.molarVolumeN2O(T_tank)
        P = (n_Ar + n_go)*self.R*T_tank / (V_tank - n_lo*Vhat_l)
        return P
    
    def ullageFractionFunc(self):
        Vhat_li = self.molarVolumeN2O(self.T_tank)
        V_li = self.n_lo*Vhat_li
        ullageFraction = 1 - (V_li / self.V_tank)
        return ullageFraction
    
    def initialMoles(self):
        n_to = self.m_N2O / self.MW_N2O  # initial total N2O in tank [kmol]
        Vhat_li = self.molarVolumeN2O(self.T_tank)
        P_sato = self.vaporPressureN2O(self.T_tank)
        # P_sato printed from units of Pa to PSI
        # print("P_sato: ", P_sato*0.000145038)
        # print(self.specificHeatN2OLiquid(self.T_tank))
        n_go = P_sato * (self.V_tank - Vhat_li * n_to) / (-P_sato * Vhat_li + self.R * self.T_tank)  # initial N2O gas [kmol]
        n_lo = (n_to * self.R * self.T_tank - P_sato * self.V_tank) / (-P_sato * Vhat_li + self.R * self.T_tank)  # initial N2O liquid [kmol]
        # print(n_go/n_to*100, n_lo/n_to*100, n_to/(n_go + n_lo))
        n_Ar = self.P_tank * (self.V_tank - n_lo*Vhat_li)/(self.R*self.T_tank) - n_go
        return n_go, n_lo, n_Ar