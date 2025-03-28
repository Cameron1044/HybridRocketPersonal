import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def ToMetric(value, conversion, n=1):
    ## To take in a value and the current unit it is in to change it into metric
    ## INPUTS: value - the value we want to convert
    ##         unit  - the unit that will be changed to metric
    ##         n     - Fuel Regression constant used to calculate a
    ##
    ## OUTPUTS: The Value will have it's units changed into metric / SI units
    ##          If the unit is already in metric, nothing will happen
    conversionDict = {
        ## For Units that are English
        # Length --> m
        "in": 1/39.37,
        "ft": 1/3.281,
        # Surfrace Area --> m2
        "in^2": 1/1550,
        "ft^2": 1/10.764,
        # Volume --> m3
        "in^3": 1/61020,
        "ft^3": 1/35.315,
        # Pressure --> Pa
        "psi": 6895,
        # Mass --> kg
        "lbm": 1/2.205,
        # Force --> N
        "lbf": 4.448,
        # Density --> kg/m^3
        "lbm/in^3": 27680,
        "lbm/ft^3": 16.01846337396,
        # Molecular Weight --> lb/lbmo
        "lb/lbmol": 1/1000,
        # Temperature --> K
        "R": 5/9 ,

        ## For Units that are Metric
        # Length --> m
        "m": 1,
        "cm": 1/100,
        "mm": 1/1000,
        # Surfrace Area --> m2
        "m^2": 1,
        "cm^2": 1/10000,
        "mm^2": (1e-6),
        # Volume --> m3
        "m^3": 1,
        "cm^3": (1e-6),
        "mm^3": (1e-9),
        "L": 1/1000,
        # Pressure --> Pa
        "Pa": 1,
        "kPa": 1000,
        # Mass --> kg
        "kg": 1,
        "g": 1/1000,
        # Force --> N
        "N": 1,
        "kN": 1000,
        # Density --> kg/m^3
        "kg/m^3": 1,
        # Molecular Weight --> kg/mol
        "g/mol": 1/1000,
        "kg/mol": 1,
        # Temperature --> K
        "K": 1,
        # Gas Constant --> J/(mol*K)
        "J/(mol*K)": 1,

        ## Unitless
        "unitless": 1,

        ## For a Burn coefficient
        "a": (0.0254**(1 + 2*(n)))*(0.453592**(-n))
    }
    return value * conversionDict[conversion]

def ToEnglish(value, conversion):
    ## To take in a value and the current unit it is in to change it into English
    ## INPUTS: value - the value we want to convert
    ##         unit  - the unit that will be changed to metric
    ##         n     - Fuel Regression constant used to calculate a
    ##
    ## OUTPUTS: The Value will have it's units changed into English / Imperial units
    ##          If the unit is already in English, nothing will happen
    conversionDict = {
        ## For Units that are English
        # Length --> in
        "in": 1,
        "ft": 1/12,
        # Surfrace Area --> in2
        "in^2": 1,
        "ft^2": 1/144,
        # Volume --> in3
        "in^3": 1,
        "ft^3": 1/1728,
        # Pressure --> Psi
        "psi": 1,
        # Mass --> lbm
        "lbm": 1,
        # Force --> lbf
        "lbf": 1,
        # Density --> lbm/in^3
        "lbm/in^3": 1,
        "lbm/ft^3": 1/ 1728,
        # Molecular Weight --> lb/lbmol
        "lb/lbmol": 1,
        # Temperature --> R
        "R": 1,

        ## For Units that are Metric
        # Length --> in
        "m": 39.37,
        "cm": 1/2.54,
        "mm": 1/25.4,
        # Surfrace Area --> in2
        "m^2": 1550,
        "cm^2": 1/ 6.452,
        "mm^2": 1 / 645.2,
        # Volume --> in3
        "m^3": 61020,
        "cm^3": 1/16.387,
        "mm^3": 1/16390,
        "L": 61.024,
        # Pressure --> Psi
        "Pa": 1 / 6895,
        "kPa": 1 / 6.895,
        # Mass --> lbm
        "kg": 2.205,
        "g": 1/453.6,
        # Force --> lbf
        "N": 1/4.448,
        "kN": 1/224.8,
        # Density --> lbm/in^3
        "kg/m^3": 1 / 27680,
        # Molecular Weight --> lb/lbmole
        "g/mol": 1000,
        "kg/mol":1,
        # Temperature --> R
        "K": 1.8,

        ## Unitless
        "unitless": 1,
    }
    return value * conversionDict[conversion]

def plot_graph(title, xlabel, ylabel, *lines):
    """
    Generalized plotting function.
    lines: A list of dictionaries containing y data and labels for each line.
    """
    plt.figure()
    
    for line in lines:
        y = line['y']
        x = line.get('x', None) # Optional x data
        label = line.get('label', None)
        linestyle = line.get('linestyle', '-')
        color = line.get('color', None)
        if label:
            if x is None:
                plt.axhline(y=y, linewidth=2, label=label, linestyle=linestyle, color=color)
            else:
                plt.plot(x, y, linewidth=2, label=label, linestyle=linestyle, color=color)
        else:
            if x is None:
                plt.axhline(y=y, linewidth=2, label=label, linestyle=linestyle, color=color)
            else:
                plt.plot(x, y, linewidth=2, linestyle=linestyle, color=color)
            
    plt.grid(which='both', linestyle='--', linewidth=0.5)
    plt.legend(loc="best")
    plt.title(title, weight='bold', fontsize = 18)
    plt.xlabel(xlabel, weight='bold', fontsize = 14)
    plt.ylabel(ylabel, weight='bold', fontsize = 14)