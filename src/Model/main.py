import numpy as np
import math as m
import matplotlib.pyplot as plt
import pandas as pd
import pprint

from Utilities.ChemicalProperties import ChemicalProperties
from Utilities.Model import Model
from Utilities.utilities import ToMetric, ToEnglish, plot_graph

# Aluminum Model Helical
initialInputs_Helical = {
    ## Purpose:  Dictionary of Initial Inputs, organized by section
    "ColdFlow": False,
    "T_tank": ToMetric(287, 'K'),
    "m_T": ToMetric(2.1, 'kg'),
    "m_N2O": ToMetric(4, 'lbm'),
    #### Oxidizer Tank
    "V_tank": ToMetric(3.09, 'L'),
    "P_tank": ToMetric(1350, 'psi'),
    #### Injector
    "A_inj": ToMetric(0.044, 'in^2'),
    "C_d": ToMetric(0.2, 'unitless'),
    #### Fuel Properties
    "rho_fuel": ToMetric(2089.83281, 'kg/m^3'),
    # "M_chmb": ToMetric(30.169, 'g/mol'),
    # "gamma": ToMetric(1.2516, 'unitless'),
    # "T_chmb": ToMetric(4000, 'K'),
    "M_chmb": ToMetric(23.719, 'g/mol'),
    "gamma": ToMetric(1.2641, 'unitless'),
    "T_chmb": ToMetric(3075, 'K'),
    #### Fuel Regression Properties
    "n": ToMetric(1.681, 'unitless'),
    "a": ToMetric(9.33E-8, 'unitless'),
    #### Fuel Grains
    "L_fuel": ToMetric(9, 'in'),
    "OD_fuel": ToMetric(3.375, 'in'),
    #### Nozzle
    "d_t": ToMetric(0.7, 'in'),
    "alpha": np.deg2rad(15), # Nozzle Diverging Half-Cone Angle
    #### Ambient Conditions
    "P_amb": ToMetric(12.3, 'psi'),
    "P_int": ToMetric(12.3, 'psi'),
    # "P_amb": ToMetric(827371, 'Pa'),
    ##### CONSTANTS #####
    "Ru": ToMetric(8.3143, 'J/(mol*K)'),
    #### ----- Geometry Type ----- ####
    "GeometryType": "Helical",
    #### FOR CYLINDRICAL
    "ID_fuel": ToMetric(2.75, 'in'),
    #### FOR HELIX
    "HelixLength": ToMetric(11.793, 'in'),
    "HelixPortDiameter": ToMetric(1.5, 'in'),
    "HelixVolume": ToMetric(59.677, 'in^3'),
    "HelixRegressionAmplification": 3.0,
}

# Aluminum Model Custom
initialInputs_Custom = {
    ## Purpose:  Dictionary of Initial Inputs, organized by section
    "ColdFlow": False,
    "T_tank": ToMetric(287, 'K'),
    "m_T": ToMetric(2.1, 'kg'),
    "m_N2O": ToMetric(3, 'lbm'),
    #### Oxidizer Tank
    "V_tank": ToMetric(3, 'L'),
    "P_tank": ToMetric(3300, 'psi'),
    #### Injector
    # "A_inj": ToMetric(1.8e-5, 'm^2'),
    "A_inj": ToMetric(0.0441, 'in^2'),
    "C_d": ToMetric(0.1, 'unitless'),
    #### Fuel Properties
    "rho_fuel": ToMetric(2089.83281, 'kg/m^3'),
    # "M_chmb": ToMetric(30.169, 'g/mol'),
    # "gamma": ToMetric(1.2516, 'unitless'),
    # "T_chmb": ToMetric(4000, 'K'),
    "M_chmb": ToMetric(23.719, 'g/mol'),
    "gamma": ToMetric(1.2641, 'unitless'),
    "T_chmb": ToMetric(3075, 'K'),
    #### Fuel Regression Properties
    "n": ToMetric(1.681, 'unitless'),
    "a": ToMetric(9.33E-8, 'unitless'),
    #### Fuel Grains
    "L_fuel": ToMetric(9, 'in'),
    "OD_fuel": ToMetric(3.375, 'in'),
    #### Nozzle
    "d_t": ToMetric(0.5, 'in'),
    "alpha": np.deg2rad(15), # Nozzle Diverging Half-Cone Angle
    #### Ambient Conditions
    "P_amb": ToMetric(12.3, 'psi'),
    "P_int": ToMetric(12.3, 'psi'),
    # "P_amb": ToMetric(827371, 'Pa'),
    ##### CONSTANTS #####
    "Ru": ToMetric(8.3143, 'J/(mol*K)'),
    #### ----- Geometry Type ----- ####
    "GeometryType": "Custom",
    #### FOR CYLINDRICAL
    "ID_fuel": ToMetric(2.75, 'in'),
    #### FOR HELIX
    "HelixLength": ToMetric(11.793, 'in'),
    "HelixPortDiameter": ToMetric(1.5, 'in'),
    "HelixVolume": ToMetric(20.838, 'in^3'),
    "HelixRegressionAmplification": 2.0,
}

# Define Model Inputs
initialInputs = initialInputs_Helical
pprint.pprint(initialInputs)

modelZK = Model(initialInputs, ZK=True)
modelBernoulli = Model(initialInputs, ZK=False)
dfZK = modelZK.ODE45()
dfBe = modelBernoulli.ODE45()

totalImpulseZK = dfZK['impulse'].iloc[-1]
totalImpulseBernoulli = dfBe['impulse'].iloc[-1]

totalFinalMassZK = dfZK['mox'].iloc[-1] + dfZK['mf'].iloc[-1]
totalFinalMassBernoulli = dfBe['mox'].iloc[-1] + dfBe['mf'].iloc[-1]

totalMassConsumedZK = ToEnglish(modelZK.mt_i, 'kg') - totalFinalMassZK
totalMassConsumedBernoulli = ToEnglish(modelBernoulli.mt_i, 'kg') - totalFinalMassBernoulli

IspZK = totalImpulseZK/totalMassConsumedZK
IspBernoulli = totalImpulseBernoulli/totalMassConsumedBernoulli

avgThrustZK = totalImpulseZK/dfZK['time'].iloc[-1]
avgThrustBernoulli = totalImpulseBernoulli/dfBe['time'].iloc[-1]

peakThrustZK = dfZK['thrust'].max()
peakThrustBernoulli = dfBe['thrust'].max()

burnTimeZK = dfZK['time'].iloc[-1]
burnTimeBernoulli = dfBe['time'].iloc[-1]

finalRZK = dfZK['r'].iloc[-1]
finalRBe = dfBe['r'].iloc[-1]
initialRZK = dfZK['r'].iloc[0]
initialRBe = dfBe['r'].iloc[0]

### Console Output ###
print("\n")
print("-------------------- Initial Port Radius -------------------")
print(f'ZK:        {np.round(initialRZK, 3)} [in]')
print(f'Bernoulli: {np.round(initialRBe, 3)} [in]')
print("-------------------- Final Port Radius -------------------")
print(f'ZK:        {np.round(finalRZK, 3)} [in]')
print(f'Bernoulli: {np.round(finalRBe, 3)} [in]')
print("-------------------- Ullage Fraction -------------------")
print(f'ZK:        {np.round(modelZK.blowdown.ullageFraction, 3)} [unitless]')
print(f'Bernoulli: {np.round(modelBernoulli.blowdown.ullageFraction, 3)} [unitless]')
print("-------------------- Total Mass -------------------")
print(f'ZK:        {np.round(ToEnglish(modelZK.mt_i, "kg"), 3)} [lbm]')
print(f'Bernoulli: {np.round(ToEnglish(modelBernoulli.mt_i, "kg"), 3)} [lbm]')
print("-------------------- Mass Burned -------------------")
print(f'ZK:        {np.round(totalMassConsumedZK, 3)} [lbm]')
print(f'Bernoulli: {np.round(totalMassConsumedBernoulli, 3)} [lbm]')
print("-------------------- Average Thrust -------------------")
print(f'ZK:        {np.round(avgThrustZK, 3)} [lbf]')
print(f'Bernoulli: {np.round(avgThrustBernoulli, 3)} [lbf]')
print("-------------------- Peak Thrust -------------------")
print(f'ZK:        {np.round(peakThrustZK, 3)} [lbf]')
print(f'Bernoulli: {np.round(peakThrustBernoulli, 3)} [lbf]')
print("-------------------- Burn Time -------------------")
print(f'ZK:        {np.round(burnTimeZK, 3)} [s]')
print(f'Bernoulli: {np.round(burnTimeBernoulli, 3)} [s]')
print("-------------------- Total Impulse -------------------")
print(f'ZK:        {np.round(totalImpulseZK, 3)} [lbf-s]')
print(f'Bernoulli: {np.round(totalImpulseBernoulli, 3)} [lbf-s]')
print("-------------------- Specific Impulse -------------------")
print(f'ZK:        {np.round(IspZK, 3)} [s]')
print(f'Bernoulli: {np.round(IspBernoulli, 3)} [s]')
print("\n")

# Oxidizer and Fuel Masses over Time
plot_graph('Rocket Masses Over Time',
            'Time (s)',
            'Mass (lbm)',
            {'y': dfZK['mox'], 'x': dfZK['time'], 'label': 'Oxidizer Mass ZK'},
            {'y': dfBe['mox'], 'x': dfBe['time'], 'label': 'Oxidizer Mass Bernoulli'},
            {'y': dfZK['mf'], 'x': dfZK['time'], 'label': 'Fuel Mass ZK', 'linestyle': '--', 'color': 'r'},
            {'y': dfBe['mf'], 'x': dfBe['time'], 'label': 'Fuel Mass Bernoulli', 'linestyle': ':', 'color': 'g'})

# Rocket Mass Flow Rate over Time
plot_graph('Rocket Mass Flow Rate over Time', 
           'Time (s)', 
           'Mass Flow Rate (lbm/s)',
           {'y': dfZK['dmf'], 'x': dfZK['time'], 'label': 'Fuel Mass Flow ZK'},
           {'y': dfBe['dmf'], 'x': dfBe['time'], 'label': 'Fuel Mass Flow Bernoulli'},
           {'y': dfZK['dmox'], 'x': dfZK['time'], 'label': 'Oxidizer Mass Flow ZK', 'linestyle': '--', 'color': 'r'},
           {'y': dfBe['dmox'], 'x': dfBe['time'], 'label': 'Oxidizer Mass Flow Bernoulli', 'linestyle': ':', 'color': 'g'})

# Impulse vs. Time
plot_graph('Impulse vs. Time', 
           'Time (s)', 
           'Total Impulse Produced (lbf-s)',
           {'y': dfZK['impulse'], 'x': dfZK['time'], 'label': 'ZK'},
           {'y': dfBe['impulse'], 'x': dfBe['time'], 'label': 'Bernoulli'})

# Mixture Ratio vs. Time
plot_graph('Mixture Ratio vs. Time', 
           'Time (s)', 
           'O/F Ratio',
           {'y': dfZK['OF'], 'x': dfZK['time'], 'label': 'ZK'},
           {'y': dfBe['OF'], 'x': dfBe['time'], 'label': 'Bernoulli'})

# Port Radius vs Time
if initialInputs["GeometryType"] == "Cylindrical":
    noFuelLine = ToEnglish(initialInputs['OD_fuel']/2, 'm')
elif initialInputs["GeometryType"] == "Custom":
    noFuelLine = ToEnglish(modelBernoulli.r_final, 'm')
elif initialInputs["GeometryType"] == "Helical":
    noFuelLine = ToEnglish(initialInputs['OD_fuel']/2, 'm')
plot_graph('Port Radius vs Time', 
           'Time (s)', 
           'Port Radius (in)',
           {'y': dfZK['r'], 'x': dfZK['time'], 'label': 'Fuel Grain Port Radius ZK'},
           {'y': dfBe['r'], 'x': dfBe['time'], 'label': 'Fuel Grain Port Radius Bernoulli'},
           {'y': noFuelLine, 'label': 'Fuel Grain Outer Diameter', 'linestyle': ':', 'color': 'r'})

# Rocket Pressures Over Time
plot_graph('Pressures vs. Time', 
           'Time (s)',
           'Pressure (psi)',
           {'y': dfZK['Pc'], 'x': dfZK['time'], 'label': 'Chamber Pressure ZK Model'},
           {'y': dfBe['Pc'], 'x': dfBe['time'], 'label': 'Chamber Pressure First Principles Model'},
           {'y': dfZK['Pox'], 'x': dfZK['time'], 'label': 'Oxidizer Tank Pressure ZK Model', 'linestyle': '--', 'color': 'r'},
           {'y': dfBe['Pox'], 'x': dfBe['time'], 'label': 'Oxidizer Tank Pressure First Principles Model', 'linestyle': '--', 'color': 'g'})

# Thrust vs. Time
plot_graph('Thrust vs. Time', 
           'Time (s)', 
           'Thrust (lbf)',
           {'y': dfZK['thrust'], 'x': dfZK['time'], 'label': 'ZK Model'},
           {'y': dfBe['thrust'], 'x': dfBe['time'], 'label': 'First Principles Model'})
plt.show()

column_mapping = {
    "time": "time (s)",
    "thrust": "thrust (lbf)",
    "thrustN": "thrust (N)",
    "impulse": "impulse (lbf-s)",
    "Pc": "Chamber pressure (psi)",
    "Pox": "Tank pressure (psi)",
    "mox": "Liquid Oxidizer mass (lbm)",
    "mf": "Fuel mass (lbm)",
    "dmox": "Oxidizer mass flow rate (lbm/s)",
    "dmf": "Fuel mass flow rate (lbm/s)",
    "OF": "Oixidizer to Fuel Ratio (unitless)",
    "r": "Port Radius (in)",
    "dm_total_dt": "Total mass flow rate (lbm/s)",
    "cstar": "cstar (in/s)",
    "T_chmb": "Chamber temperature (K)",
    "M_chmb": "Chamber molecular weight (g/mol)",
    "gamma": "gamma (unitless)",
}

dfZK = dfZK.apply(lambda col: col.map(lambda x: format(x, '.3f')))
dfZK.rename(columns=column_mapping, inplace=True)
# dfZK.to_csv("src/Model/CSV/ModelData.csv", index=False)