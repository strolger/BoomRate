#!/usr/bin/env python
import string
from pylab import *

from scipy.integrate import quad
#from scipy.integrate import inf
from math import pi,sqrt

from matplotlib.font_manager import fontManager, FontProperties

def col(data, colindex):
    return [row[colindex] for row in data]


def cosmotime(z,Qm=0.27,Ql=0.73,ho=71):
    noi=200
    res=1.49E-4

    if (z < 500):
        (int,err)=quad(lambda tt: ((1.+tt)**(-1.))*((((1.+tt)**2.)*(1.+Qm*tt))-(tt*(2.+tt)*Ql))**(-0.5),0,z,limit=noi,epsabs=res)
    else:
        int=0
    lookback=(1./ho)*1E6*3.086E13/(3.156E7*1E9)*int
    return lookback


def cosmoz(ltt,WM=0.27,WV=0.73,ho=71.):
    h = ho/100.
    WR = 4.165E-5/(h*h)# includes 3 massless neutrino species, T0 = 2.72528
    WK = 1.0-WM-WR-WV
    age = 0
    ad = 1
    a=0.9995
    Tyr=977.8 # coefficient for converting 1/H into Gyr
    tltt=ho*ltt/Tyr
    adot=0
    while (age < tltt): 
        #for (a = 0.9995; age < ltt; a = a-0.001) {
        adot=sqrt(WK+(WM/a)+(WR/(a*a))+(WV*a*a))
        age = age + 0.001/adot
        ad = ad-0.001
        if (a < 0.001):
            age = tltt+100
            adot = 0
            ad = 1.0E-9
        a=a-0.001
    z = 1.0/(ad+(age-tltt)*adot) - 1.0
    return (z)


if __name__ == '__main__':

    z=float(sys.argv[1])
    val=cosmotime(z)
    #val=cosmoz(z)
    print (val)
    
