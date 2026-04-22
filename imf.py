#!/usr/bin/env python
import os,sys,pdb,scipy,glob
from pylab import *
#from strolger_util import util as u
import util as u
from scipy.integrate import quad
from scipy.optimize import curve_fit as cf
rcParams['font.size']=20.0
rcParams['figure.figsize']=9,8
rcParams['font.family'] = 'serif'


def venditti(m, *p):
    a, b, mc = p
    ksi = m**-a*exp(-(mc/m)**b)
    return(ksi)

def venditti1(m, *p):
    a, b, mc = p
    ksi = m**-a*exp(-(mc/m)**b)
    return(m*ksi)

def salpeter(m, a=2.35):
    return(m**(-a))

def salpeter1(m, a=2.35):
    return(m*m**(-a))

def kroupa(m, *p):
    a1=0.3; a2=1.3; a3=2.3;
    c1=25.; c2=2.

    mb,c3=p

    if type(m) is ndarray:
        ksi = zeros(len(m))
        idx=where(m <= 0.08)
        ksi[idx] = c1*m[idx]**(-a1)
        idx=where((m>0.08)&(m<=mb))
        ksi[idx] = c2*m[idx]**(-a2)
        idx=where(m >mb)
        ksi[idx] = c3*m[idx]**(-a3)
    else:
        if m <= 0.08:
            ksi = c1*m**(-a1)
        elif ((m>0.08)&(m<=mb)):
            ksi = c2*m**(-a2)
        else:
            ksi = c3*m**(-a3)
    return(ksi)

def kroupa1(m, *p):
    a1=0.3; a2=1.3; a3=2.3;
    c1=25.; c2=2.

    mb,c3=p

    if type(m) is ndarray:
        ksi = zeros(len(m))
        idx=where(m <= 0.08)
        ksi[idx] = c1*m[idx]**(-a1)
        idx=where((m>0.08)&(m<=mb))
        ksi[idx] = c2*m[idx]**(-a2)
        idx=where(m >mb)
        ksi[idx] = c3*m[idx]**(-a3)
    else:
        if m <= 0.08:
            ksi = c1*m**(-a1)
        elif ((m>0.08)&(m<=mb)):
            ksi = c2*m**(-a2)
        else:
            ksi = c3*m**(-a3)
    return(m*ksi)

def weisz(m, *p):
    a1=0.3; a2=1.3; a3=2.45;
    c1=25.; c2=2.

    mb,c3=p

    if type(m) is ndarray:
        ksi = zeros(len(m))
        idx=where(m <= 0.08)
        ksi[idx] = c1*m[idx]**(-a1)
        idx=where((m>0.08)&(m<=mb))
        ksi[idx] = c2*m[idx]**(-a2)
        idx=where(m >mb)
        ksi[idx] = c3*m[idx]**(-a3)
    else:
        if m <= 0.08:
            ksi = c1*m**(-a1)
        elif ((m>0.08)&(m<=mb)):
            ksi = c2*m**(-a2)
        else:
            ksi = c3*m**(-a3)
    return(ksi)

def weisz1(m, *p):
    a1=0.3; a2=1.3; a3=2.45;
    c1=25.; c2=2.

    mb,c3=p

    if type(m) is ndarray:
        ksi = zeros(len(m))
        idx=where(m <= 0.08)
        ksi[idx] = c1*m[idx]**(-a1)
        idx=where((m>0.08)&(m<=mb))
        ksi[idx] = c2*m[idx]**(-a2)
        idx=where(m >mb)
        ksi[idx] = c3*m[idx]**(-a3)
    else:
        if m <= 0.08:
            ksi = c1*m**(-a1)
        elif ((m>0.08)&(m<=mb)):
            ksi = c2*m**(-a2)
        else:
            ksi = c3*m**(-a3)
    return(m*ksi)


def chary(m, *p):
    a1=0.3; a2=1.3;# a3=1.65;
    c1=25.; c2=2.

    mb,c3,zz=p
    a3 = -0.072*zz+2.3

    if type(m) is ndarray:
        ksi = zeros(len(m))
        idx=where(m <= 0.08)
        ksi[idx] = c1*m[idx]**(-a1)
        idx=where((m>0.08)&(m<=mb))
        ksi[idx] = c2*m[idx]**(-a2)
        idx=where(m >mb)
        ksi[idx] = c3*m[idx]**(-a3)
    else:
        if m <= 0.08:
            ksi = c1*m**(-a1)
        elif ((m>0.08)&(m<=mb)):
            ksi = c2*m**(-a2)
        else:
            ksi = c3*m**(-a3)
    return(ksi)

def chary1(m, *p):
    a1=0.3; a2=1.3;# a3=1.65;
    c1=25.; c2=2.

    mb,c3,zz=p
    a3 = -0.072*zz+2.3

    if type(m) is ndarray:
        ksi = zeros(len(m))
        idx=where(m <= 0.08)
        ksi[idx] = c1*m[idx]**(-a1)
        idx=where((m>0.08)&(m<=mb))
        ksi[idx] = c2*m[idx]**(-a2)
        idx=where(m >mb)
        ksi[idx] = c3*m[idx]**(-a3)
    else:
        if m <= 0.08:
            ksi = c1*m**(-a1)
        elif ((m>0.08)&(m<=mb)):
            ksi = c2*m**(-a2)
        else:
            ksi = c3*m**(-a3)
    return(m*ksi)



def fline(x,*p):
    m,b=p
    return(m*x+b)


if __name__=='__main__':
    
    dm = 0.01
    m = arange(dm,350,dm)
    zz = arange(0.5,4.0,0.5)
    zz = array([0.5, 1.,2.,4.])
    
    plot1=True
    massfn = True
    
    if plot1:
        ax=subplot(111)
        p0=[0.5,1.]
        p1=[0.5,1.5,9.0]
        if massfn:
            ## ax.plot(log10(m),log10(10*weisz(m,*p0)*m*log(10.)),'r-', lw=3, label='Weisz (2015)')
            ax.plot(log10(m),log10(10*salpeter(m)*m*log(10.)),'k-', label='Salpeter (1955)')
            ax.plot(log10(m),log10(10*kroupa(m,*p0)*m*log(10.)),'b-', label='Kroupa (2003)')
            ax.plot(log10(m),log10(10*chary(m,*p1)*m*log(10.)), 'g--', lw=3, label='Chary (2008)')
            ax.annotate(r'z$\sim$%2.1f'%9, xy=(log10(m[-1])+0.1,log10(10*chary(m[-1],*p1)*m[-1]*log(10.))),
                        xycoords='data')#, fontsize=10.)
            for i,z in enumerate(zz):
                mb = 0.5*(1.+z)**2.
                c3=(1.+z)**2.
                p0=[mb,c3]
                if i ==0:
                    ax.plot(log10(m),log10(10*kroupa(m,*p0)*m*log(10.)),'r--', label='Davé (2008)')
                else:
                    ax.plot(log10(m),log10(10*kroupa(m,*p0)*m*log(10.)),'r--')
                ax.annotate('z=%2.1f'%z, xy=(log10(m[-1])+0.1,log10(10*kroupa(m[-1],*p0)*m[-1]*log(10.))),
                            xycoords='data')#, fontsize=10.)
                
            p0=[2.35, 1, 10]
            ax.plot(log10(m), log10(1e3*venditti(m,*p0)*m*log(10.)), '-', color='purple', lw=3, label='Venditti+(2023)')
                
        else:       
            ## ax.plot(log10(m),log10(10*weisz(m,*p0)*dm*log(10.)),'r-', lw=3, label='Weisz (2015)')
            ax.plot(log10(m),log10(10*salpeter(m)*dm*log(10.)),'k-', label='Salpeter (1955)')
            ax.plot(log10(m),log10(10*kroupa(m,*p0)*dm*log(10.)),'b-', label='Kroupa (2003)')
            ax.plot(log10(m),log10(10*chary(m,*p1)*dm*log(10.)), 'g--', lw=3, label='Chary (2008)')
            ax.annotate(r'z$\sim$%2.1f'%9, xy=(log10(m[-1])+0.1,log10(10*chary(m[-1],*p1)*dm*log(10.))),
                       xycoords='data')#, fontsize=10.)
            ## for i,z in enumerate(zz):
            ##     mb = 0.5*(1.+z)**2.
            ##     c3=(1.+z)**2.
            ##     p0=[mb,c3]
            ##     if i ==0:
            ##         ax.plot(log10(m),log10(kroupa(m,*p0)*dm*log(10.)),'r--', label='Davé (2008)')
            ##     else:
            ##         ax.plot(log10(m),log10(kroupa(m,*p0)*dm*log(10.)),'r--')
            ##     ax.annotate('z=%2.1f'%z, xy=(log10(m[-1])+0.1,log10(kroupa(m[-1],*p0)*dm*log(10.))),
            ##                 xycoords='data')#, fontsize=10.)


        ## SNe Ia
        ax.axvline(log10(3), color='orange')
        ax.axvline(log10(8), color='orange')
        ax.axvspan(log10(3), log10(8), color='orange', alpha=0.1, zorder=0)
        ax.annotate('SNe Ia', xy=(log10(3)+0.1,2.0), rotation=90,
            xycoords='data')#, fontsize=14.)

        ax.axvline(log10(8), color='g')
        ax.axvline(log10(50), color='g')
        ax.axvspan(log10(8), log10(50), color='g', alpha=0.1, zorder=0)
        ax.annotate('CC SNe', xy=(log10(8)+0.1,2.0), rotation=90,
            xycoords='data')#, fontsize=14.)

        ax.axvline(log10(50), color='b')
        ax.axvline(log10(300), color='b')
        ax.axvspan(log10(50), log10(300), color='b', alpha=0.1, zorder=0)
        ax.annotate('SLSNe+PISNe', xy=(log10(50)+0.1,1.0), rotation=90,
            xycoords='data',)# fontsize=14.)

        ## ax.set_xlabel(r'$\log\, M\,(M_{\odot})$')#, fontsize=14)
        ## ax.set_ylabel(r'$\log[M\times\xi(M)]$')#, fontsize=14)
        ax.set_title('Initial Mass Function')
        ax.set_xlabel(r'$\log_{10}(M/M_{\odot})$', fontsize=24)
        if massfn:
            ax.set_ylabel(r'$\log_{10}$ (total mass per mass interval)', fontsize=22)
        else:
            ax.set_ylabel(r'$\log_{10}(N)$', fontsize=24)
        ax.set_ylim(-2,3)
        #ax.grid()
        ax.set_xlim(-2.1,3.6)
        #ax.set_xlim(-2.1,2.1)
        lg = ax.legend(loc=3)
        lg.draw_frame(False)
        savefig('figure_imf.png')
    
    num1 = quad(salpeter,8,50)[0]
    den1 = quad(salpeter1,0.1,125)[0]
    print ('Salpeter k=%1.4f' %(num1/den1))

    p0=[0.5,1.]
    num = quad(kroupa,8,50,args=tuple(p0))[0]
    den = quad(kroupa1,0.1,350,args=tuple(p0))[0]
    print ('Kroupa k=%1.4f' %(num/den))

    p0=[0.5,1.]
    num = quad(weisz,8,50,args=tuple(p0))[0]
    den = quad(weisz1,0.1,350,args=tuple(p0))[0]
    print ('Weisz k=%1.4f' %(num/den))

    val=[]
    print ('Dave k(z):')
    for z in zz:
        mb = 0.5*(1.+z)**2.
        c3=(1.+z)**2.
        p0=[mb,c3]
        num = quad(kroupa,8,50,args=tuple(p0))[0]
        den = quad(kroupa1,0.1,350,args=tuple(p0))[0]
        val.append(num/den)
        print ('%2.1f %1.4f' %(z,num/den))
