import matplotlib as mpt
from cycler import cycler

def set_rcParams():
    mpt.rcParams['text.usetex'] = True
    mpt.rcParams['font.family'] = "serif"
    mpt.rcParams['font.serif'] = ['Computer Modern']
    mpt.rcParams['lines.linewidth'] = 1.
    mpt.rcParams['font.size'] = 20
    mpt.rcParams['axes.prop_cycle'] = cycler(color=['brown', 'b', 'y', 'darkgray'])
    mpt.rcParams['legend.labelcolor'] = 'black'
    mpt.rcParams['legend.fontsize'] = 18
    mpt.rcParams['xtick.labelsize'] = 18
    mpt.rcParams['xtick.labelcolor'] = 'black'
    mpt.rcParams['xtick.color'] = 'black'
    mpt.rcParams['ytick.labelsize'] = 18
    mpt.rcParams['ytick.labelcolor'] = 'black'
    mpt.rcParams['ytick.color'] = 'black'
    mpt.rcParams['axes.prop_cycle'] = cycler(color=['r', 'g', 'b', 'y'])
    
    mpt.rcParams['lines.linewidth'] = 1
