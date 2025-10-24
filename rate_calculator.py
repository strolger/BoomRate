#!/usr/bin/env python

'''
This script calculates rates from an observed number distribution
with redshift
'''


import os,sys,pdb,scipy,glob, time, pickle
from pylab import *
from scipy.optimize import curve_fit
from scipy.integrate import quad
from strolger_util import util as u
from strolger_util import rates_z as rz
from strolger_util import imf
import volume, control_time
import warnings#,exceptions
warnings.simplefilter("error",RuntimeWarning)
warnings.filterwarnings("ignore")
import multiprocessing
import json, logging
import pandas as pd
from scipy import stats




vol_frac_a={ # Li et al. 2011
    'iip': 0.524,
    'iin': 0.064,
    'iil': 0.073,
    'ib' : 0.069,
    'ic' : 0.176,
    'ia' : 1.0,
    'slsn':0.00022, ## for SLSNe, from Prajs et al. 2019; Not very happy with this scaling (SLSN inefficient?, or can i do imf scaling?)
    #'fast': 1.0, # for A. Rest, 2017
    }
vol_frac_b={ # Richardson et al. 2014
    'iip': 0.409,
    'iin': 0.116,
    'iil': 0.094,
    'ib' : 0.099,
    'ic' : 0.199,
    }

vol_frac=vol_frac_a

absmags_li_2011 = {
    'iip': [-15.66, 1.23, 0.16],
    'iin': [-16.86, 1.61, 0.59],
    'iil': [-17.44, 0.64, 0.22],
    'ib' : [-17.01, 0.41, 0.17],
    'ic' : [-16.04, 1.28, 0.31],
    'ibc': [-16.04, 1.28, 0.31],
    }

absmags_richardson_2014 = {
    'iip': [-16.80, 0.97, 0.37],
    'iin': [-18.62, 1.48, 0.32],
    'iil': [-17.98, 0.90, 0.34],
    'ib' : [-17.54, 0.94, 0.33],
    'ic' : [-16.67, 1.04, 0.40],
    'ibc': [-16.67, 1.04, 0.40],
    'ia' : [-19.26, 0.51, 0.20],
    #'fast' : [-17.5, 1.0, 0.1], ## not sure where this is from
    'slsn': [-21.7, 0.4,0.0], ## from Quimby+2013, by way of Gal-Yam 2018   
    ## 'slsn': [-30, 2.5,0.0], ## from Whalen et al. 2013
    }
#
absmags=absmags_richardson_2014
verbose = True

#absmag_new = {}
#for key in absmags.keys(): absmag_new[key]=[absmags[key][0]-absmags[key][2],absmags[key][1],absmags[key][2]]
#absmags=absmag_new


def snrates(z,*p):
    A,B,C,D=p
    k = 0.0091*(.70)**2. ## for low-mass core-collapse supernovaae
    return(1e4*k*A*((1+z)**C)/(1+((1+z)/B)**D))

def cc_snrates(z,type):
    #k = 0.0091*(.70)**2. ## for low-mass core-collapse supernovaae
    k = 0.007*(.70)**2. ## for low-mass core-collapse supernovaae
    #return(1e4*k*vol_frac[type]*rz.sfr_2020(z))
    return(1e4*k*rz.sfr_2020(z)) ## I'm removing volumetric fraction to use iip rates as a proxy

def snrates_Ia(z):
    ##data = loadtxt('SNRmodelTable.dat')
    data = loadtxt('LGSfitTable.dat')
    try:
        junk, yy = u.recast(z,0.,data[:,0], data[:,1])
    except:
        pdb.set_trace()
    yy = [x if x > 0.0 else 0.0 for x in yy]
    return(yy)

def make_cadence_table(types,redshift,tess_sens=19):
    rise_times = {
        'iip':20,
        }
    rise = rise_times['iip']*(1.+redshift)
    N = int(365./rise)
    
    cadence_table = 'cadences.txt'
    if os.path.isfile(cadence_table): os.remove(cadence_table)
    f = open(cadence_table,'w')
    f.write("#Cadence(days)  Area(sqarcmin)  Sens(iA) Prev_baseline\n")
    data = []
    for i in range(N):
        f.write("%d %4.2f %3.1f %d\n" %(rise, tess_area, tess_sens, rise))
        data.append([int(rise),float(tess_area),float(tess_sens), int(rise)])
    data = array(data)
    f.close()
    return(data)
    

def get_unique_visits(survey):
    temp=[]
    for item in survey:
        temp.append('_'.join(item.astype('str')))
    out=[]
    temp2 = sorted(set(temp), key = lambda x: float(x.split("_")[1]), reverse=True)
    for i,item in enumerate(temp2):
        out.append(list(map(float,item.split('_')))+ [temp.count(item)])
    return(array(out))

def fline(x,*p):
    m,b = p
    return m*x+b

def fline2(x,*p):
    m,b = p
    return (m*x+b)/(1.0+x)


def run(redshift2, redshift1, rate_guess, number_guess,
        types=['ia'],Nproc=1,extinction=True,obs_extin=True,survey=None, cadence_file=None, passband=None,
        verbose=verbose, maglim=22., parallel=True, integ=True, passskiprow=1, passwavemult=0.1,
        dstep=0.5, dmstep=0.1, dastep=0.1,
        ratefile=None, eventtable=None):
    rate=0.0
    N={}
    denom={}
    redshift = (redshift2+redshift1)/2.
    print('z=%2.1f rg=%2.2f ng=%2.1f' %(redshift, rate_guess, number_guess))
    rate_guess = rate_guess*1.0e-4
    
    ## types=['iip','iin','iil','ib','ic']
    ## types = ['ia']
    ## types=['iil']
    ## types = ['slsn']


    ### Survey Constants
    #tess_area = (60.)**2 ## a square degree, to put the final number in per sq. degree
    #tess_sens = 17.0 ## mag
    #survey = make_cadence_table(types,redshift, tess_sens=maglim)
    if cadence_file and not survey:
        survey = loadtxt(cadence_file)
    else:
        if len(shape(survey))==1:
            survey = append(survey, 1.)
            survey = array([list(survey)])
        else:
            survey = get_unique_visits(survey)
        Dvol = volume.run(redshift2)-volume.run(redshift1)
        
    tc_tot=0
    for i,item in enumerate(survey):
        baseline=item[0]
        area = item[1]
        area_frac = area * (1./60.)**2*(pi/180)**2*(4.0*pi)**(-1)
        #sens= item[2]
        sens =maglim
        prev = item[3]
        for type in types:
            if not integ:
                tc = control_time.run(redshift, baseline, sens, dstep=10, Nproc=Nproc, parallel=parallel,
                                      extinction=extinction, obs_extin=obs_extin,
                                      type=[type], prev=prev, passband=passband,
                                      passwavemult=passwavemult, passskiprow=passskiprow,
                                      verbose=verbose, plot=False)
            else:
                tc1 = control_time.run(redshift1, baseline, sens, Nproc=Nproc, parallel=parallel,
                                       extinction=extinction, obs_extin=obs_extin,
                                       type=[type], prev=prev,passband=passband,
                                       passwavemult=passwavemult, passskiprow=passskiprow,
                                       dstep=dstep, dmstep=dmstep, dastep=dastep,
                                       verbose=verbose)
                tc2 = control_time.run(redshift2, baseline, sens, Nproc=Nproc, parallel=parallel,
                                       extinction=extinction, obs_extin=obs_extin,
                                       type=[type], prev=prev,passband=passband,
                                       passwavemult=passwavemult, passskiprow=passskiprow,
                                       dstep=dstep, dmstep=dmstep, dastep=dastep,
                                       verbose=verbose)
                xx =array([redshift1, redshift2])
                yy = array([tc1,tc2])
                yy[isnan(yy)]=0.0 ## remove any nans
                p0=[1.0,0.0]
                pout = curve_fit(fline,xx,yy,p0=p0)[0]
                tc = quad(fline,xx[0],xx[1],args=tuple(pout))[0]/diff(xx)
                tc = tc[0]
            exp_num = rate_guess*(tc*Dvol*area_frac)

            print("Mean Control Time= %4.2f observed frame days" %(tc*365.25))
            try:
                multiplier = item[-1]
            except:
                multiplier = 1.0
            try:
                N[type]+=(exp_num*multiplier)
            except:
                N[type]=exp_num*multiplier
            if verbose: print('%d %s %2.1f' %(i,type, N[type]))
            tc_tot += tc*multiplier

        print("%d %2.2f %s" %(i, sum(list(N.values())), ','.join(list(map(str,item)))))
        if verbose: print("\n")
        
    Nexp = sum(list(N.values()))
    print('Redshift= %2.2f (%2.2f - %2.2f)' %(redshift,redshift1, redshift2))
    if Nexp > 1.0:
        Nexp_hi,Nexp_lo = poisson_error(Nexp)
    elif Nexp == 0:
        Nexp_hi = 0
        Nexp_lo = 0
    else:
        temp = 1.0/Nexp
        Nexp_hi,Nexp_lo = poisson_error(1.0)
        Nexp_hi=Nexp_hi/temp
        Nexp_lo=Nexp_lo/temp

    
    Nobs = number_guess
    Nobs_hi, Nobs_lo = poisson_error(Nobs)

    Robs = Nobs/(tc_tot*Dvol*area_frac)*1e4
    Rerr_hi = Nobs_hi/(tc_tot*Dvol*area_frac)*1e4-Robs
    Rerr_lo = Robs-Nobs_lo/(tc_tot*Dvol*area_frac)*1e4

    print('Rate from %2.1f expected events to %2.1f mag: R=%2.2f+%2.2f-%2.2f' %(number_guess, sens, Robs, Rerr_hi, Rerr_lo))
    print('Number expected to %2.1f mag from expected rate of R=%2.2f: N=%2.1f+%2.1f-%2.1f' %(sens, rate_guess*1e4, Nexp, Nexp_hi-Nexp, Nexp-Nexp_lo))
    print('-------\n')
    print('%.2f   %.2f   %.2f   %.2f   %.2f   %.2f   %.1f   %.1f   %.1f   %.1f'
          %(redshift, redshift1, redshift2, Robs, Rerr_hi, Rerr_lo, Nexp, Nexp_hi-Nexp, Nexp-Nexp_lo, number_guess))
    print('[%.2f,%.2f,%.2f,%.2f,%.2f,%.2f],'
          %(redshift, redshift1, redshift2, Robs, Rerr_hi, Rerr_lo))
    f=open(ratefile,'a')
    f.write('%.2f,%.2f,%.2f,%.2f,%.2f,%.2f,%.1f,%.1f,%.1f,%.1f\n'
            %(redshift, redshift1, redshift2, Robs, Rerr_hi, Rerr_lo,
              Nexp, Nexp_hi-Nexp, Nexp-Nexp_lo, Nobs)
            )
    f,close()
    return([Nexp, Nexp_hi, Nexp_lo, tc_tot])

def poisson_error(n):
    #from table in Gehrels (1986), where CL are determined from Newton's method solution.
    if type(n) is not ndarray:
        n = array([n])
    ul =array([
        [0,1.841],
        [1,3.300],
        [2,4.638],
        [5,8.382],
        [10,14.27],
        [20,25.55],
        [40,47.38],
        [100,111.0],
        ])
    ll =array([
        [0,0.0],
        [1,0.173],
        [2,0.708],
        [5,2.840],
        [10,6.891],
        [20,15.57],
        [40,33.70],
        [100,90.02],
        ])
    (junk,nul)=u.recast(n,0.0,ul[:,0],ul[:,1])
    (junk,nll)=u.recast(n,0.0,ll[:,0],ll[:,1])
    return(nul[0],nll[0])


def main(configfile=None):
    if not configfile:
        print("No configfile specified\n")
        return() 
    with open(configfile) as data_file:
        config = json.loads(data_file.read())
    clobber = json.loads(config['clobber'])
    verbose = json.loads(config['verbose'])

    #survey
    sntypes = config['sntypes']
    imf_evol = config['imf_evol']
    extinction = json.loads(config['extinction'])
    try:
        obs_extin = json.loads(config['obs_extin'])
        if obs_extin==True: obs_extin='nominal' ## for backward compatabilty
    except:
        obs_extin = config['obs_extin']
    cadence_file=config['cadence_file']
    itermag = json.loads(config['itermag'])
    passband = config['passband']
    passskiprow = config['passskiprow']
    passwavemult = config['passwavemult']
    
    #processing 
    outfile = config['outfile_rates']
    multiproc = json.loads(config['multiproc'])
    
    #sample table
    eventtable = config['eventtable']
    determinate = json.loads(config['determinate'])

    #fake binned SNe
    falseevents = json.loads(config['falseevents'])
    falsetable = config['falsetable']
    if not falsetable:
        falsetable = config['outfile_numbers']

    Nbins = config['nbins']
    if not Nbins: Nbins = 3

    if multiproc:
        Nproc=int(multiprocessing.cpu_count()-2)
    else:
        Nproc=1
    if falseevents and determinate: determinate=False    


    ## some for setting the fineness of SN parameter eval
    dstep=config['mag_step']
    dmstep=config['abs_mag_step']
    dastep=config['extinction_step']
    integ=json.loads(config['box_tc'])

    if not os.path.isfile(outfile) or clobber:
        if falseevents and os.path.isfile(falsetable):
            tmp = pickle.load(open(falsetable,'rb'))
            redshifts = tmp[:,-2]
            redshifts=append(redshifts,tmp[-1,-1])
            ng = tmp[:,1]
        elif not eventtable:
            print('No event table')
            pdb.set_trace()
        else:
            sn_table = pd.read_csv(eventtable, sep='\t')
            Nbins = Nbins
            splits = arange(0, 1+1./Nbins, 1./Nbins).tolist()
            if  max(sn_table['pIa']+sn_table['pII']+sn_table['pIbc']) > 99:
                sn_table['pIa']=sn_table['pIa'].apply(lambda x: x*0.01)
                sn_table['pII']=sn_table['pII'].apply(lambda x: x*0.01)
                sn_table['pIbc']=sn_table['pIbc'].apply(lambda x: x*0.01)

            if determinate:
                if 'ia' in sntypes:
                    sn_condition = sn_table['pIa']>0.5
                elif (('iil' in sntypes) or ('iip' in sntypes)
                      or ('ic' in sntypes) or ('ib' in sntypes) or ('iin' in sntypes)):
                    sn_condition1 = sn_table['pII']>0.5
                    sn_condition2 = sn_table['pIbc']>0.5
                    sn_condition = sn_condition1 | sn_condition2
                else:
                    print('Only set for SNe Ia and CCSNe\n')
                    pdb.set_trace()
                sne_select = sn_table.where(sn_condition)
                sne_select['Redshift']=sne_select['z_host']
                redshift_bins = stats.mstats.mquantiles(sne_select['Redshift'][~np.isnan(sne_select['Redshift'])], splits)

                ### need a more elegant counter than this soon.
                ng, redshifts = histogram(sne_select['Redshift'][~np.isnan(sne_select['Redshift'])], redshift_bins)
                ax = subplot(111)
                style = {'facecolor': 'none', 'edgecolor': 'C0', 'linewidth': 3}
                ax.hist(sne_select['Redshift'][~np.isnan(sne_select['Redshift'])], redshift_bins, label='%.1f total'%(sum(ng)),**style)
                ax.set_xlabel('Redshift')
                ax.set_xlim(0,6)
                if 'ia' in sntypes:
                    ax.set_ylabel('Number of SNeIa')
                else:
                    ax.set_ylabel('Number of CCSNe')
                ax.legend(loc=1)
                savefig('tmp1.png')
                clf()                
            else:
                ## dzza=0.001
                ## redshift_tmp=arange(0,10+dzza,dzza)
                ## rv = zeros(len(redshift_tmp))
                ## for index, row in sn_table.iterrows():
                ##     if row['z_type']=='spec':
                ##         zz = row['z']
                ##         dzz = dzza
                ##     else:
                ##         zz = row['z']
                ##         dzz = max(row['z']-row['z16'],row['z85']-row['z'])
                ##     if 'ia' in sntypes:
                ##         rv_tmp = stats.norm.pdf(redshift_tmp, loc=zz, scale=dzz)*row['pIa']
                ##     elif (('iil' in sntypes) or ('iip' in sntypes)
                ##           or ('ic' in sntypes) or ('ib' in sntypes) or ('iin' in sntypes)):
                ##         rv_tmp1 = stats.norm.pdf(redshift_tmp, loc=zz, scale=dzz)*row['pII']
                ##         rv_tmp2 = stats.norm.pdf(redshift_tmp, loc=zz, scale=dzz)*row['pIbc']
                ##         rv_tmp=rv_tmp1+rv_tmp2
                ##     else:
                ##         print('Only set for SNe Ia and CCSNe\n')
                ##         pdb.set_trace()
                ##     rv+=rv_tmp
                dzza=0.001
                redshift_tmp=arange(0,10+dzza,dzza)
                rv = zeros(len(redshift_tmp))
                for index, row in sn_table.iterrows():
                    if row['pIa']==row['pII']==row['pIbc']==-0.99: continue
                    if row['pIa']==row['pII']==row['pIbc']==-99: continue
                    zz = row['z_host']
                    dzz = row['z_host_err']
                    if ((dzz == -99.0) | (isnan(dzz))): dzz = dzza
                    if 'ia' in sntypes:
                        rv_tmp = stats.norm.pdf(redshift_tmp, loc=zz, scale=dzz)*row['pIa']
                    elif (('iil' in sntypes) or ('iip' in sntypes)
                          or ('ic' in sntypes) or ('ib' in sntypes) or ('iin' in sntypes)):
                        rv_tmp1 = stats.norm.pdf(redshift_tmp, loc=zz, scale=dzz)*row['pII']
                        rv_tmp2 = stats.norm.pdf(redshift_tmp, loc=zz, scale=dzz)*row['pIbc']
                        rv_tmp=rv_tmp1+rv_tmp2
                    else:
                        print('Only set for SNe Ia and CCSNe\n')
                        pdb.set_trace()
                    rv+=rv_tmp
                pv = cumsum(rv)*dzza#/sum(rv)

                ax = subplot(111)
                ax.plot(redshift_tmp,rv*dzza,'k-')
                ax.set_xlabel('Redshift')
                ax.set_xlim(0,6)
                if 'ia' in sntypes:
                    ax.set_ylabel('Number of SNeIa')
                else:
                    ax.set_ylabel('Number of CCSNe')
                savefig('tmp1.png')
                clf()
                ax = subplot(111)
                ax.plot(redshift_tmp,cumsum(rv)*dzza, 'k-', label='%.1f Total'%(cumsum(rv)[-1]*dzza))
                ax.set_xlabel('Redshift')
                ax.set_xlim(0,6)
                if 'ia' in sntypes:
                    ax.set_ylabel('Cumulative number of SNeIa')
                else:
                    ax.set_ylabel('Cumulative number of CCSNe')
                ax.legend(loc=1)
                savefig('tmp2.png')
                clf()
                
                bins=zeros(Nbins+1)
                for i,split in enumerate(splits):
                    if i==0: continue
                    ii =where(pv/pv[-1]<=splits[i])
                    bins[i]=redshift_tmp[ii][-1]
                ii = where(pv/pv[-1] < 0.01)
                bins[0]=redshift_tmp[ii][-1]
                ii = where((max(pv)-pv)/pv[-1] < 0.01)
                bins[-1]=redshift_tmp[ii][0]

                ng = zeros(Nbins)
                for i,bin in enumerate(bins):
                    if i == 0: continue
                    ii = where((redshift_tmp>bins[i-1])&(redshift_tmp < bins[i]))
                    ng[i-1] = sum(rv[ii])*0.001
                redshifts = bins
        med_z = (redshifts[1:]+redshifts[:-1])/2.
        
        if 'ia' in sntypes:
            rg = array(snrates_Ia(med_z))## for SNe Ia, from Strolger et al. 2020
        elif (('iil' in sntypes) or ('iip' in sntypes) or ('ic' in sntypes) or ('ib' in sntypes) or ('iin' in sntypes)):
            if imf_evol is None:
                rg = cc_snrates(array(med_z),sntypes[0])
            elif imf_evol=='dave':
                print('Evol. k(z) testing...')
                sfrg = rz.sfr_2020(med_z)
                k_z=[]
                for zz in med_z:
                    mb = 0.5*(1.+zz)**2.
                    c3 = (1.+zz)**2.
                    p0 = [mb,c3]
                    num = quad(imf.kroupa,8,50,args=tuple(p0))[0] ## covering mass range of CCSNe
                    den = quad(imf.kroupa1,0.1,350,args=tuple(p0))[0]
                    k_z.append(num/den)
                k_z = array(k_z)
                k_z = 1.e4*(.70)**2.*k_z
                rg = k_z * sfrg#*vol_frac[type]
            else:
                pdb.set_trace()
                
                
        elif 'slsn' in sntypes:
            if imf_evol is None:
                #rg = cc_snrates(array(med_z),sntypes[0])
                #rg = rg * 2.2e-4 
                sfrg = rz.sfr_2020(med_z)
                ## k=quad(imf.salpeter,50,350)[0]/quad(imf.salpeter1,50,350)[0]
                p0=[0.5,1.]
                k = quad(imf.kroupa,50,350,args=tuple(p0))[0]/quad(imf.kroupa1,0.1,350,args=tuple(p0))[0]
                rg =  1.e4*(.70)**2.*k*sfrg
                rg = rg *0.022
                
            elif imf_evol=='dave':
                print('Evol. k(z) testing...')
                sfrg = rz.sfr_2020(med_z)
                k_z=[]
                for zz in med_z:
                    mb = 0.5*(1.+zz)**2.
                    c3 = (1.+zz)**2.
                    p0 = [mb,c3]
                    num = quad(imf.kroupa,50,350,args=tuple(p0))[0] ## covering mass range of popIII SNe
                    den = quad(imf.kroupa1,0.1,350,args=tuple(p0))[0]
                    k_z.append(num/den)
                k_z = array(k_z)
                k_z = 1.e4*(.70)**2.*k_z ## the factor of 10 accounts for fewer really high mass stars in IMF
                rg = k_z * sfrg
                #rg = rg *10. ## a guess, really, at efficiency
                rg = rg *0.022 ## a guess, really, at efficiency
            ##if imf_evol=='chary': ## not yet ready on this
            else:
                pdb.set_trace()
                

        survey = loadtxt(cadence_file)
        if itermag:
            ## mags = arange(18.0,33.0,0.1)
            mags = arange(24, 31, 2)
        else:
            try:
                mags = array(list(set(survey[:,2])))
            except:
                mags = array([survey[2]])
        numbers = []
        ctr = 0

        out_rate_file = outfile.replace('.pkl','.txt')
        if os.path.isfile(out_rate_file): os.remove(out_rate_file)
        f=open(out_rate_file,'a')
        f.write('#z,z_low,z_hi,R,R_+err,R_-err,N_exp,N_+err,N_-err,Nobs\n')
        f.close()
        for j,mag in enumerate(mags):
            for i,redshift in enumerate(redshifts):
                if i==0: continue
                ctr+=1
                print('\n\n Iteration %d of %d\n\n' %(ctr,len(mags)*(len(redshifts)-1)))
                print('Rate guess = %2.2f' %rg[i-1])
                num, nhi, nlo, tc = run(redshifts[i], redshifts[i-1], rg[i-1], ng[i-1],
                                        types=sntypes, Nproc=Nproc, maglim=mag, parallel=multiproc,passband=passband,
                                        verbose=verbose, extinction=extinction, obs_extin=obs_extin,
                                        survey = survey,
                                        cadence_file = None,
                                        integ=integ,
                                        dstep=dstep, dmstep=dmstep, dastep=dastep,
                                        passwavemult=passwavemult,
                                        passskiprow=passskiprow,
                                        eventtable=eventtable,
                                        ratefile=out_rate_file,
                                        )
                numbers.append([mag, num, nhi, nlo, (redshifts[i]+redshifts[i-1])/2., redshifts[i-1], redshifts[i]])
        numbers=array(numbers)
        if os.path.isfile(outfile) and clobber: os.remove(outfile)
        pickle.dump(numbers,open(outfile,'wb'))
    else:
        outdata = pickle.load(open(outfile,'rb'))

if __name__=='__main__':
    
    if sys.argv[1] and sys.argv[1].endswith('.json'):
        configfile='./'+sys.argv[1]
    else:
        configfile = './config.json'
    print("Proceeding with rate_estimator using %s" %(configfile))
    logging.info("Proceeding with rate_estimator using %s" %(configfile))
    main(configfile=configfile)
            
    



    
