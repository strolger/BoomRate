#!/usr/bin/env python
import string
from pylab import *

from scipy.integrate import quad
from math import pi
from scipy.interpolate import *
from numpy import r_

from matplotlib.font_manager import fontManager, FontProperties
from strolger_util import cosmotools

import warnings
warnings.simplefilter("ignore",RuntimeWarning)

par_model = [0.0134, 2.55, 3.3, 6.1] #strolger2020
par_err = [0.0009, 0.09, 0.2, 0.2]

def csfh_crazy(z, *p):
    A,B,C,D = p
    sfr1=A*((1+z)**C)/(1+((1+z)/B)**D)
    pfuz= [0.003, 3.0, 2.6, 7.71604675]
    A2, B2, C2, D2 = pfuz
    sfr2=A2*((1+z-3)**C2)/(1+((1+z-3)/B2)**D2)
    sfr2=nan_to_num(sfr2)
    sfr = sfr1+sfr2
    return(sfr)
    
def csfh_crazy_time(time, *p):
    from strolger_util import cosmotools as ct
    A,B,C,D = p
    lbt = 13.6 - time
    z = array([ct.cosmoz(x) for x in lbt])
    sfr1=A*((1+z)**C)/(1+((1+z)/B)**D)
    pfuz= [0.003, 3.0, 2.6, 7.71604675]
    A2, B2, C2, D2 = pfuz
    sfr2=A2*((1+z-3)**C2)/(1+((1+z-3)/B2)**D2)
    sfr2=nan_to_num(sfr2)
    sfr = sfr1+sfr2
    return(sfr)


def csfh(z,*p):
    A,B,C,D = p
    sfr=A*((1+z)**C)/(1+((1+z)/B)**D)
    return(sfr)

def dfdp_m(p,z): #for confidence band calculation
    A, B, C, D = p
    term1 =((1+z)**C)/(1+((1+z)/B)**D)
    term2 = A*D*((1+z)**C)*(((1+z)/B)**D)/(B*(1+((1+z)/B)**D)**2)
    term3 = (A*((1+z)**C)*log(1+z))/(1+((1+z)/B)**D)
    term4 = (A*((1+z)**C)*(((1+z)/B)**D)*log((1+z)/B))/((1+((1+z)/B)**D)**2.)
    return ([term1, term2,term3,term4])


def confidence_band(x, y, err, dfdp, confprob, func, popt, pcov): ## be sure to set absolute_sigma=True in curve_fit!!
   from scipy.stats import t
   import numpy
   # Given the confidence probability confprob = 100(1-alpha)
   # we derive for alpha: alpha = 1 - confprob

   dof = len(x)-len(popt)
   chi2 = sum(((y - func(x,*popt))**2.)/err**2.)
   rchi2= chi2/dof

   alpha = 1.0 - confprob
   prb = 1.0 - alpha/2
   tval = t.ppf(prb, dof)


   C = pcov
   n = len(popt)              # Number of parameters from covariance matrix
   p = popt
   N = len(x)
   covscale = rchi2
   df2 = numpy.zeros(N)
   for j in range(n):
      for k in range(n):
         df2 += dfdp[j]*dfdp[k]*C[j,k]
   df = numpy.sqrt(rchi2*df2)
   yy = func(x, *p)
   delta = tval * df
   upperband = yy + delta
   lowerband = yy - delta
   return (yy, upperband, lowerband)


def csfh_time(time, *p):
    from strolger_util import cosmotools as ct
    A,B,C,D = p
    lbt = 13.6 - time
    z = array([ct.cosmoz(x) for x in lbt])
    sfr=A*((1+z)**C)/(1+((1+z)/B)**D)
    return(sfr)
  
def ccsnr(z):
    A = 0.015
    B = 1.5
    C = 5.0
    D = 6.1
    sfr=A*((1+z)**C)/(1+((1+z)/B)**D)
    return(sfr)
    
def sco_model(x): ## a good fit to observed Ia rates
    if x < 1.0:
        y=2.5e-5*(1+x)**(1.5)
    else:
        y=9.7e-5*(1+x)**(-0.5)
    return(y)

vsco_model = vectorize(sco_model)


def powerdtd(time, *p, normed=True, cutoff=True):
    from scipy.integrate import simpson
    aa, bb= p
    warnings.simplefilter("ignore",RuntimeWarning)
    ret = time**aa
    if cutoff: ret = piecewise(time, [time <=bb, time > bb], [0.0, lambda x: x**aa]) ## for putting in a cutoff.
    ret[~isfinite(ret)]=0.0
    if normed:
        dd = simpson(ret, x=time)
        if dd!=0.0:
            dd = 1.0/dd
        else:
            dd = 1.0
    else:
        dd = 1.0
    return(dd*ret)

def expdtd(time, *p, normed=True):
    import scipy
    aa = p[0]
    yy = abs(aa)*scipy.stats.expon.pdf(time, aa)
    if normed:
        scale = scipy.integrate.simpson(yy, x=time)
        if scale !=0.0:
            scale = 1.0/scale
        else:
            scale = 1.0
    else:
        scale = 1.0 
    return(yy)



def dtdfunc(time,*p, norm=True):
    '''
    This is the better one
    '''
    from scipy.integrate import simpson,trapezoid
    from scipy.special import erf
    import warnings
    warnings.simplefilter("error",RuntimeWarning)
    #    aa=3.2## xi
    #    bb=0.2## omega
    #    cc=2.2## alpha
    aa, bb, cc= p
    tt = array([cc*(x-aa)/bb for x in time])
    val1 = sqrt(pi/2.)*erf(tt/sqrt(2.))+sqrt(pi/2.)
    val2 = (1./(bb*pi))*exp(-((time-aa)**2)/(2*bb**2))
    ret = val1*val2
    try:
        dd = trapezoid(ret,x=time)
    except:
        import pdb; pdb.set_trace()
    if norm and dd !=0.0:
        try:
            dd = 1.0/dd
        except RuntimeWarning:
            dd=0.0
    else:
        dd = 1.0
    return(dd*ret)

def dtdfunc_v2(time,*p, norm=True):
    '''
    This one generally works, but has issues at certain values\
    due to quad
    '''

    from scipy.integrate import simpson
    #    aa=3.2## xi
    #    bb=0.2## omega
    #    cc=2.2## alpha
    aa, bb, cc= p
    val1 =  [quad(lambda x: exp(-((x**2)/2)), -inf, cc*(tt-aa)/bb)[0] for tt in time]
    val1 = array(val1)
    val2=(1./(bb*pi))*exp(-((time-aa)**2)/(2*bb**2))
    ret = val1*val2
    dd = simpson(ret,x=time)
    if norm and dd !=0.0:
        dd = 1.0/dd
    else:
        dd = 1.0
    return(dd*ret)

def dtdfunc_v1(aa,bb,cc):
    
#    aa=3.2## xi
#    bb=0.2## omega
#    cc=2.2## alpha

    step=0.01
    noi=200
    
    i=-4.0
    time=[]
    val=[]
    while i < 15:
        par=cc*(i-aa)/bb
        (int,err)=quad(lambda x: exp(-((x**2)/2)), -inf, par)
        val1=(1/(bb*pi))*exp(-((i-aa)**2)/(2*bb**2))
        time.append(i)
        val.append(val1*int)#*step)
        i=i+step
    return time,val

def dtd(t,time,dfunc):
    for n in range(len(time)-1):
        if (abs(time[n]-t)<0.01):
            dtdt=dfunc[n]
        elif ((t>time[n]) & (t<=time[n+1])):
            slope=(dfunc[n+1]-dfunc[n])/(time[n+1]-time[n])
            b=dfunc[n+1]-slope*time[n+1]
            dtdt=slope*t+b
    return dtdt
            


if __name__ == '__main__':
    print('Hi')
