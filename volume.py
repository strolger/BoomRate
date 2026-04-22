#!/usr/bin/env python
from scipy import *
from pylab import *
import os,sys
#from strolger_util import cosmocalc
import cosmocalc


def run(z, qm=0.27, ql=0.73, ho=71):
    (dl,mu,peak)=cosmocalc.run(z, qm=qm, ql=ql, ho=ho)
    dm=dl/(1+z)
    qk=1.-qm-ql
    dh=3.0e5/ho

    if (qk > 0):
        vc=(4*pi*dh**3/(2*qk))*(dm/dh*sqrt(1+qk*(dm/dh)**2)-1/(sqrt(abs(qk)))*math.asinh(sqrt(abs(qk))*dm/dh))
    elif (qk < 0):
        vc=(4*pi*dh**3/(2*qk))*(dm/dh*sqrt(1+qk*(dm/dh)**2)-1/(sqrt(abs(qk)))*math.asin(sqrt(abs(qk))*dm/dh))
    else: #(qk == 0):
        vc=4*pi/3*dm**3
    return (vc)

if __name__ == '__main__':
    z=sys.argv[1]
    print(run(float(z))/1000.**3.)
