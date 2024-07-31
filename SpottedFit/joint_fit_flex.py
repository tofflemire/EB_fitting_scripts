import numpy as np
import emcee
#import matplotlib
#import matplotlib.pyplot as plt
#from matplotlib.backends.backend_pdf import PdfPages
#import corner
from scipy.optimize import curve_fit
import astropy.units as u
import copy
#import corner as corner
from scipy.signal import spectral
import pickle as pkl
import astropy.io.fits as fits
import batman
#import celerite
#from celerite import terms
#from mixterm import MixtureOfSHOsTerm as celeritekernel
from scipy.stats import kde
from astropy import constants as const
import sys
import argparse
import time
import os
import scipy.interpolate as interpolate

t_start = time.time()
t_max = 5.5 * 60.0 * 60.0
#t_max = 2 * 60.0
n_threads = 140
step_check = 1000

parser=argparse.ArgumentParser()
parser.add_argument("infile", type=str, help="name of the input file")
parser.add_argument("-n", "--new", action="store_true", help="Ignore any saved run and start fresh")

args=parser.parse_args()
infile=args.infile

#Constants
ms = const.M_sun.cgs.value #1.989*10**33 #g
G = const.G.cgs.value      #6.6743*10**-8 #cm3/g/s2
au = const.au.cgs.value    #1.496*10**13 #cm
rs = const.R_sun.cgs.value #6.955*10**10 #cm

rv_file, lc_file_1,kde_file_1,filt_1,q1_val_1,q2_val_1,qerr_1, lc_file_2,kde_file_2,filt_2,q1_val_2,q2_val_2,qerr_2, lc_file_3,kde_file_3,filt_3,q1_val_3,q2_val_3,qerr_3, max_steps = np.loadtxt(infile,unpack=True,dtype='U100,U100,U100,U100,f,f,f,U100,U100,U100,f,f,f,U100,U100,U100,f,f,f,i',delimiter=',')

#1 - TESS
#2 - r
#3 - I

#This is the Kipping 2013 sampling thing
g1_val_1 = (q1_val_1+q2_val_1)**2
g2_val_1 = 0.5*q1_val_1/(q1_val_1+q2_val_1)

g1_val_2 = (q1_val_2+q2_val_2)**2
g2_val_2 = 0.5*q1_val_2/(q1_val_2+q2_val_2)

g1_val_3 = (q1_val_3+q2_val_3)**2
g2_val_3 = 0.5*q1_val_3/(q1_val_3+q2_val_3)

def binary_rvs(t,P,T0,K1pK2,q,gamma,e,omega):
    
    #q is the mass ratio (K1/K2)
    #K1, K2 in km/s (K1 + K2 is sum of semimajor amplitude, guess=~140)
    #gamma in km/s (COM vel, guess=~28)
    #omega in radians (0 for this system because eccentricity is 0... longitude of periastron)
    #T0,P,t in the same units of time, days is fine
    #t has to be the same array doubled, i.e. t=np.append(t,t)
    # -This is dumb but it's an easy way to fit it to observed RVs
    #The output rv array is an array of even length where 
    #rv1 = rvs[:rvs.size/2]
    #rv1-rv2 = rvs[rvs.size/2:]

    #This function can be fit to data with scipy.optimize.curve_fit, but 
    #you need to have the primary and secondary properly labeled and in the 
    #output format, i.e. rvs = np.append(rv1,rv1-rv2).
    #Also, the guesses have to be somewhat close.
    #Would recommend using emcee or some other MCMC approach.
    

    rv1 = np.zeros(t.size//2)
    rv2 = np.zeros(t.size//2)
    theta = np.zeros(t.size//2)
    f = np.zeros(t.size//2)

    for i in range(t.size//2):
        M = ((2.0*np.pi/P) * (t[i]-T0)) % (2.0*np.pi) #Mean Anomaly

        E = ((2.0*np.pi/P) * (t[i]-T0)) % (2.0*np.pi) #Eccentric Anomaly
        
        M_out =  E - e*np.sin(E)

        step = 0
        while ((np.abs(M_out - M) > 0.000000001) & (step < 1000000)):
            E = E - ((E-e*np.sin(E)-M)/(1.0-e*np.cos(E)))
            M_out = E - e*np.sin(E)
            step = step + 1

        if e > 0:
            x = ((1.0-e**2)/(e*(1.0-e*np.cos(E))) - (1.0/e))
            if x > 1: x = 1.0
            if x < -1: x = -1.0
            theta[i] = np.arccos(x)
        else:
            theta[i] = ((t[i]-T0)/P % 1) * 2.0*np.pi
            if theta[i] > np.pi: theta[i] = 2.0*np.pi - theta[i]

        if M > np.pi:
            theta = 2.0*np.pi-theta

        K1 = K1pK2/(1.0+1.0/q)
        K2 = K1/q

        rv1[i] = K1*(np.cos(theta[i]+omega) + e*np.cos(omega)) + gamma
        rv2[i] = -K2*(np.cos(theta[i]+omega) + e*np.cos(omega)) + gamma

    rvs = np.append(rv1,rv1-rv2)
    #rvs = np.append(rv1,rv2)

    return rvs


def binary_lc(t,t0,P,rsrp,aorpprs,inc,sbr,ecc,omega,g1,g2,f_spot,f_ecl,C):
    
    #Turn the Kipping limb darkening gs into qs
    if ((g1 == 0) & (g2 == 0)):
        q1 = 0 
        q2 = 0
    else:
        q1 = np.sqrt(g1)/(1.0+((0.5-g2)/g2))
        q2 = np.sqrt(g1) - q1

    #for the primary star
    q1_1 = q1
    q1_2 = q2

    #for the secondary star, assuming it is the same as the primary
    q2_1 = q1
    q2_2 = q2

    ##### Descrition of params ####
    #t0 - Time of Periastron passage (days, Barycentric Julian)
    #P - Orbital Period (days)
    #rsrp - Ratio of the radii, secondary radius divided by primary radius (unitless)
    #aorpprs - semi-major axis divided by the sum of the stellar radii (unitless)
    #inc - Orbit Inclination (degrees)
    #ecc - Orbit Eccentricity (unitless)
    #omega - Longitude of Periastron (radians)
    #q1_1 - First quadratic limb darkening parameter for primary (unitless)
    #q1_2 - Second quadratic limb darkening parameter for primary (unitless)
    #q2_1 - First quadratic limb darkening parameter for secondary (unitless)
    #q2_2 - Second quadratic limb darkening parameter for secondary (unitless)
    #sbr - central surface brightness ratio, SBs/SBp, called J_2/J_1 below (unitless)

    ##### A bunch of trig that's overly complicated #####
    #It has to do with the timing of the eclipses w.r.t periastron passage
    #Fine to ignore the details
    
    PHI_peri = (omega - np.pi/2.0)/(2.0*np.pi)
    
    #The inferior conjunction (little c) - The smaller star in front, i.e., primary eclipse
    #typically the deeper eclipse
    fc = (np.pi/2.0) - omega #True Anomaly of inferior conjunction
    Ec = 2.0 * np.arctan((np.sqrt((1-ecc)/(1+ecc)))*np.tan(fc/2.0)) #Eccentirc Anomaly of inferior conjunction
    Mc = Ec - ecc*np.sin(Ec) #Mean Anomaly of inferior conjunction
    PHI_c = (Mc+omega)/(np.pi*2.0) - (1.0/4.0) #Phase of the inferior conjuction

    t0_infc = t0+P*((PHI_c-PHI_peri)) #Time of inferior conjunction
    
    #now the same for the superior conjuction (big C) - The bigger star in front, i.e., secondary eclipse
    fC = fc + np.pi % (2.0*np.pi)
    EC = 2.0 * np.arctan((np.sqrt((1-ecc)/(1+ecc)))*np.tan(fC/2.0))
    MC = EC - ecc*np.sin(EC)
    PHI_C = (MC+omega)/(np.pi*2.0) - (1.0/4.0)

    t0_supc = t0+P*((PHI_C-PHI_peri)) #Time of superior conjunction

    w_degree = (omega/np.pi) * 180.0 #turn longitude of periastron into degrees. 
    
    #### Put the relevant parameters into batman ####
    
    #Primary eclipse
    aors_p = aorpprs*(1.0+rsrp) #For the primary eclipse, we want aors to be a/Rp
    
    params_p = batman.TransitParams()       #object to store transit parameters
    params_p.t0 = t0_infc                        #time of inferior conjunction
    params_p.per = P                       #orbital period
    params_p.rp = rsrp                       #For primary eclipse, we want Rs/Rp
    params_p.a = aors_p                        #semi-major axis (in units of primary stellar radii)
    params_p.inc = inc                      #orbital inclination (in degrees)
    params_p.ecc = ecc                       #eccentricity
    params_p.w = w_degree                        #longitude of periastron (in degrees)
    params_p.limb_dark = "quadratic"        #limb darkening model
    params_p.u = [q1_1,q1_2]      #limb darkening coefficients [u1, u2, u3, u4]
    
    model_primary = batman.TransitModel(params_p, t)    #initializes model
    batmodel_primary = model_primary.light_curve(params_p)                    #calculates light curve

    rel_flux_of_secondary = sbr*rsrp**2

    batmodel_primary_dilute = (batmodel_primary + rel_flux_of_secondary) / (1.0 + rel_flux_of_secondary)

    #Code to alter the eclipse depth to simulate spots
    tdr = (1-f_ecl+f_ecl*C)/(1-f_spot+f_spot*C)

    batmodel_primary_dilute = 1-((1-batmodel_primary_dilute)*tdr)

    #Secondary eclipse
    aors_s = aorpprs*(1.0+1.0/rsrp) #For the secondary eclipse, we want aors to be a/Rs
    
    params_s = batman.TransitParams()       #object to store transit parameters
    params_s.t0 = t0_supc                        #time of superior conjunction
    params_s.per = P                       #orbital period
    params_s.rp = 1.0/rsrp                       #For seconadry eclipse, we want Rp/Rs
    params_s.a = aors_s                        #semi-major axis (in units of secondary stellar radii)
    params_s.inc = inc                      #orbital inclination (in degrees)
    params_s.ecc = ecc                       #eccentricity
    params_s.w = w_degree-180.0             #longitude of periastron (in degrees)
    params_s.limb_dark = "quadratic"        #limb darkening model
    params_s.u = [q2_1,q2_2]      #limb darkening coefficients [u1, u2, u3, u4]
    model_secondary = batman.TransitModel(params_s, t)    #initializes model
    batmodel_secondary = model_secondary.light_curve(params_s)                    #calculates light curve

    rel_flux_of_primary = 1.0/rel_flux_of_secondary

    batmodel_secondary_dilute = (batmodel_secondary + rel_flux_of_primary) / (1.0 + rel_flux_of_primary)
    
    #### combine the eclipses ####
    
    batmodel_binary = (batmodel_primary_dilute-1) + (batmodel_secondary_dilute-1) + 1
    
    return batmodel_binary


def phase_of_conjunction(ecc,omega):
    #omega in radians

    PHI_peri = (omega - np.pi/2.0)/(2.0*np.pi)
    
    #The inferior conjunction (little c) - The smaller star in front, i.e., primary eclipse
    #typically the deeper eclipse
    fc = (np.pi/2.0) - omega #True Anomaly of inferior conjunction
    Ec = 2.0 * np.arctan((np.sqrt((1-ecc)/(1+ecc)))*np.tan(fc/2.0)) #Eccentirc Anomaly of inferior conjunction
    Mc = Ec - ecc*np.sin(Ec) #Mean Anomaly of inferior conjunction
    PHI_c = (Mc+omega)/(np.pi*2.0) - (1.0/4.0) #Phase of the inferior conjuction
    
    #now the same for the superior conjuction (big C) - The bigger star in front, i.e., secondary eclipse
    fC = fc + np.pi % (2.0*np.pi)
    EC = 2.0 * np.arctan((np.sqrt((1-ecc)/(1+ecc)))*np.tan(fC/2.0))
    MC = EC - ecc*np.sin(EC)
    PHI_C = (MC+omega)/(np.pi*2.0) - (1.0/4.0)


    return PHI_peri,PHI_C,PHI_c


def log_likelihood(params, t_lc, f, ferr, lc_id, t_rv, rv, rverr, rv_id):

    lnl = 0.0

    #Precomputed from PHOENIX models
    C1 = 0.53 #TESS
    C2 = 0.29 #rp
    C3 = 0.63 #I

    #unpacking the fit variables
    t0,P,rsrp,aorpprs,cosi,sqesinw,sqecosw,g1_f1,g2_f1,sbr_f1,logf_lc_f1, g1_f2,g2_f2,sbr_f2,logf_lc_f2, g1_f3,g2_f3,sbr_f3,logf_lc_f3, f_spot,f_ecl, K1pK2,q,gamma,mu = params
    
    inc = 180.0*np.arccos(cosi)/np.pi

    ecc = sqesinw**2+sqecosw**2
    if sqesinw >= 0:
        omega = np.arccos(sqecosw/np.sqrt(ecc))
    else:
        omega = 2.0*np.pi-np.arccos(sqecosw/np.sqrt(ecc))

    #computing log likelihood for the light curve
    params_inlc_f1 = t0,P,rsrp,aorpprs,inc,sbr_f1,ecc,omega,g1_f1,g2_f1,f_spot,f_ecl,C1
    params_inlc_f2 = t0,P,rsrp,aorpprs,inc,sbr_f2,ecc,omega,g1_f2,g2_f2,f_spot,f_ecl,C2
    params_inlc_f3 = t0,P,rsrp,aorpprs,inc,sbr_f3,ecc,omega,g1_f3,g2_f3,f_spot,f_ecl,C3

    model_lc_f1 = binary_lc(t_lc[lc_id==1],*params_inlc_f1)
    model_lc_f2 = binary_lc(t_lc[lc_id==2],*params_inlc_f2)
    model_lc_f3 = binary_lc(t_lc[lc_id==3],*params_inlc_f3)


    sigma2_lc_f1 = ferr[lc_id==1]**2 + model_lc_f1**2*np.exp(2*logf_lc_f1)
    sigma2_lc_f2 = ferr[lc_id==2]**2 + model_lc_f2**2*np.exp(2*logf_lc_f2)
    sigma2_lc_f3 = ferr[lc_id==3]**2 + model_lc_f3**2*np.exp(2*logf_lc_f3)
    
    lnl += -0.5*np.sum((f[lc_id==1]-model_lc_f1)**2/sigma2_lc_f1 + np.log(sigma2_lc_f1))
    lnl += -0.5*np.sum((f[lc_id==2]-model_lc_f2)**2/sigma2_lc_f2 + np.log(sigma2_lc_f2))
    lnl += -0.5*np.sum((f[lc_id==3]-model_lc_f3)**2/sigma2_lc_f3 + np.log(sigma2_lc_f3))

    #computing log likelihood for the RV curve

    salt = (rv_id == 'salt')
    harps = (rv_id == 'eso36')

    params_inrv = P,t0,K1pK2,q,gamma,ecc,omega

    model_rv = binary_rvs(t_rv,*params_inrv)

    sigma_s1_salt = rverr[:t_rv.size//2][salt]**2 #+ np.exp(2*logf_rv1_s)
    sigma_s12_salt = rverr[t_rv.size//2:][salt]**2 #+ np.exp(2*logf_rv12_s)

    sigma_s1_harps = rverr[:t_rv.size//2][harps]**2 #+ np.exp(2*logf_rv1_h)
    sigma_s12_harps = rverr[t_rv.size//2:][harps]**2 #+ np.exp(2*logf_rv12_h)

    lnl += -0.5*np.sum((rv[:t_rv.size//2][salt]-model_rv[:t_rv.size//2][salt])**2/sigma_s1_salt + np.log(sigma_s1_salt))
    lnl += -0.5*np.sum((rv[t_rv.size//2:][salt]-model_rv[t_rv.size//2:][salt])**2/sigma_s12_salt + np.log(sigma_s12_salt))

    lnl += -0.5*np.sum((rv[:t_rv.size//2][harps]-mu-model_rv[:t_rv.size//2][harps])**2/sigma_s1_harps + np.log(sigma_s1_harps))
    lnl += -0.5*np.sum((rv[t_rv.size//2:][harps]-model_rv[t_rv.size//2:][harps])**2/sigma_s12_harps + np.log(sigma_s12_harps))

    return lnl


def log_of_normal(x,mu,sig):

	return np.log(1.0/(np.sqrt(2*np.pi)*sig))-0.5*(x-mu)**2/sig**2


def log_prior(params):
    t0,P,rsrp,aorpprs,cosi,sqesinw,sqecosw, g1_f1,g2_f1,sbr_f1,logf_lc_f1, g1_f2,g2_f2,sbr_f2,logf_lc_f2, g1_f3,g2_f3,sbr_f3,logf_lc_f3, f_spot,f_ecl, K1pK2,q,gamma,mu = params
    
    #parameters with Gaussian Priors - from lc analysis
    #Fit value is within this prior - were good
    P_mu = 10.71475
    P_sig = 0.0003
    P_prior = np.log(1.0/(np.sqrt(2*np.pi)*P_sig))-0.5*(P-P_mu)**2/P_sig**2 #log_of_normal(P,P_mu,P_sig)

    #parameters with Gaussian Priors - from lc analysis
    #Fit value is within this prior - were good
    f_spot_mu = 0.3
    f_spot_sig = 0.15
    f_spot_prior = np.log(1.0/(np.sqrt(2*np.pi)*f_spot_sig))-0.5*(f_spot-f_spot_mu)**2/f_spot_sig**2 

    #parameters with flat priors
    t0_min = 2458712.0445504654 - 0.5 - 2457000
    t0_max = 2458712.0445504654 + 0.5 - 2457000

    aorpprs_min = 10.0
    aorpprs_max = 60.0

    inc_min = 82.0
    inc_max = 90.0

    ecc_min = 0.1
    ecc_max = 0.5

    w_min = 0.0
    w_max = 2.0*np.pi

    logf_lc_min = -12.0
    logf_lc_max = 1.0

    K1pK2_min = 25.0
    K1pK2_max = 100.0

    q_min = 0.8
    q_max = 1.2

    gamma_min = 14.0
    gamma_max = 34.0

    #Parameters using the KDE joint prior
    rr_sbr_jp_f1 = kde_rr_sbr_f1((rsrp,sbr_f1))
    rr_sbr_jp_f2 = kde_rr_sbr_f2((rsrp,sbr_f2))
    rr_sbr_jp_f3 = kde_rr_sbr_f3((rsrp,sbr_f3))

    if aorpprs_min < aorpprs < aorpprs_max and 0 <= cosi <= 1 and -1 < sqecosw < 1 and -1 < sqesinw < 1 and logf_lc_min < logf_lc_f1 < logf_lc_max and logf_lc_min < logf_lc_f2 < logf_lc_max and logf_lc_min < logf_lc_f3 < logf_lc_max and K1pK2_min < K1pK2 < K1pK2_max and q_min < q < q_max and gamma_min < gamma < gamma_max and rr_sbr_jp_f1 > 0 and rr_sbr_jp_f2 > 0 and rr_sbr_jp_f3 > 0 and 0 < g1_f1 < 1 and 0 < g2_f1 < 1 and 0 < g1_f2 < 1 and 0 < g2_f2 < 1 and 0 < g1_f3 < 1 and 0 < g2_f3 < 1 and 0.6 < rsrp < 1.8 and 0.2 < sbr_f1 < 1.8 and 0.2 < sbr_f2 < 1.8 and 0.2 < sbr_f3 < 1.8 and 0.0 <= f_spot <= 0.8 and 0.0 <= f_ecl <= 1.0: 
    	return P_prior + np.log(rr_sbr_jp_f1) + np.log(rr_sbr_jp_f2) + np.log(rr_sbr_jp_f3) + f_spot_prior
    else:
    	return -np.inf


def log_probability(params, t_lc, f, ferr, lc_id, t_rv, rv, rverr, rv_id):
	lnl = log_likelihood(params, t_lc, f, ferr, lc_id, t_rv, rv, rverr, rv_id)

	lp = log_prior(params)

	if not np.isfinite(lp):
		return -np.inf
	return lp + lnl


def next_pow_two(n):
    """Returns the next power of two greater than or equal to `n`"""
    i = 1
    while i < n:
        i = i << 1
    return i

def function_1d(x):
    """Estimate the normalized autocorrelation function of a 1-D series
    Args:
        x: The series as a 1-D numpy array.
    Returns:
        array: The autocorrelation function of the time series.
    """
    x = np.atleast_1d(x)
    if len(x.shape) != 1:
        raise ValueError("invalid dimensions for 1D autocorrelation function")
    n = next_pow_two(len(x))

    # Compute the FFT and then (from that) the auto-correlation function
    f = np.fft.fft(x - np.mean(x), n=2 * n)
    acf = np.fft.ifft(f * np.conjugate(f))[: len(x)].real
    acf /= acf[0]
    return acf

def auto_window(taus, c):
    m = np.arange(len(taus)) < c * taus
    if np.any(m):
        return np.argmin(m)
    return len(taus) - 1

def integrated_time(x, c=5, tol=50, quiet=False):
    """Estimate the integrated autocorrelation time of a time series.
    This estimate uses the iterative procedure described on page 16 of
    `Sokal's notes <http://www.stat.unc.edu/faculty/cji/Sokal.pdf>`_ to
    determine a reasonable window size.
    Args:
        x: The time series. If multidimensional, set the time axis using the
            ``axis`` keyword argument and the function will be computed for
            every other axis.
        c (Optional[float]): The step size for the window search. (default:
            ``5``)
        tol (Optional[float]): The minimum number of autocorrelation times
            needed to trust the estimate. (default: ``50``)
        quiet (Optional[bool]): This argument controls the behavior when the
            chain is too short. If ``True``, give a warning instead of raising
            an :class:`AutocorrError`. (default: ``False``)
    Returns:
        float or array: An estimate of the integrated autocorrelation time of
            the time series ``x`` computed along the axis ``axis``.
    Raises
        AutocorrError: If the autocorrelation time can't be reliably estimated
            from the chain and ``quiet`` is ``False``. This normally means
            that the chain is too short.
    """
    x = np.atleast_1d(x)
    if len(x.shape) == 1:
        x = x[:, np.newaxis, np.newaxis]
    if len(x.shape) == 2:
        x = x[:, :, np.newaxis]
    if len(x.shape) != 3:
        raise ValueError("invalid dimensions")

    n_w, n_t, n_d = x.shape
    tau_est = np.empty(n_d)
    windows = np.empty(n_d, dtype=int)

    # Loop over parameters
    for d in range(n_d):
        f = np.zeros(n_t)
        for k in range(n_w):
            f += function_1d(x[k, :, d])
        f /= n_w
        taus = 2.0 * np.cumsum(f) - 1.0
        windows[d] = auto_window(taus, c)
        tau_est[d] = taus[windows[d]]

    # Check convergence
    flag = tol * tau_est > n_t

    # Warn or raise in the case of non-convergence
    #if np.any(flag):
    #    msg = (
    #        "The chain is shorter than {0} times the integrated "
    #        "autocorrelation time for {1} parameter(s). Use this estimate "
    #        "with caution and run a longer chain!\n"
    #    ).format(tol, np.sum(flag))
    #    msg += "N/{0} = {1:.0f};\ntau: {2}".format(tol, n_t / tol, tau_est)
    #    if not quiet:
    #        raise AutocorrError(tau_est, msg)
    #    logging.warning(msg)

    return tau_est


#Read in the data for the radius ratio -- surface brightness ratio joint prior
#FILTER 1
kde_rr_sbr_f1 = pkl.load(open(str(kde_file_1),'rb'))

#FILTER 2
kde_rr_sbr_f2 = pkl.load(open(str(kde_file_2),'rb'))

#FILTER 3
kde_rr_sbr_f3 = pkl.load(open(str(kde_file_3),'rb'))

window_out = '2.0_kde_all'

def read_lcs(lc_file,filt):

    if filt=='TESS':
    	#Read in the de-flared, detrended TESS data
    	t,f,ferr,q = np.loadtxt(str(lc_file),unpack=True)   
    	t = t - 2457000
    
    	#from an initial run
    	t0_init = 2458712.032951-2457000
    	P_init = 10.71476726
    	e_init = 0.294293
    	w_init = 2.603442
    	
    	PHI_peri_init,PHI_C_init,PHI_c_init = phase_of_conjunction(e_init,w_init)
    	t0_supc_init = t0_init+P_init*((PHI_C_init-PHI_peri_init))
    	t0_infc_init = t0_init+P_init*((PHI_c_init-PHI_peri_init))
    	
    	phase_supc_init = (t0_supc_init - t0_init)/P_init % 1
    	phase_infc_init = (t0_infc_init - t0_init)/P_init % 1
    	
    	phase_lc_init = (t - t0_init)/P_init % 1
    	
    	to_fit = (((phase_lc_init > phase_supc_init-0.05) & (phase_lc_init < phase_supc_init+0.05)) | 
    	          ((phase_lc_init > phase_infc_init-0.05) & (phase_lc_init < phase_infc_init+0.05)))
    else:
    	t,f,ferr = np.loadtxt(str(lc_file),unpack=True)   
    	t = t - 2457000
    
    	to_fit = np.ones(t.size,dtype=bool)

    return t,f,ferr,to_fit

#1 - TESS
#2 - r
#3 - I

t1,f1,ferr1,to_fit1 = read_lcs(lc_file_1,filt_1)
lc_id1 = np.zeros(t1.size)+1

t2,f2,ferr2,to_fit2 = read_lcs(lc_file_2,filt_2)
lc_id2 = np.zeros(t2.size)+2

t3,f3,ferr3,to_fit3 = read_lcs(lc_file_3,filt_3)
lc_id3 = np.zeros(t3.size)+3

t = np.concatenate([t1[to_fit1],t2[to_fit2],t3[to_fit3]])
f = np.concatenate([f1[to_fit1],f2[to_fit2],f3[to_fit3]])
ferr = np.concatenate([ferr1[to_fit1],ferr2[to_fit2],ferr3[to_fit3]])
lc_id = np.concatenate([lc_id1[to_fit1],lc_id2[to_fit2],lc_id3[to_fit3]])

#Read in the SALT RVs
Name,Tele,bjd,\
    vsini1,vsini1_errp,vsini1_errm,vsini1_95,vmac1,vmac1_errp,vmac1_errm,vmac1_95,brv1,BRV1errp,BRV1errm,\
    vsini2,vsini2_errp,vsini2_errm,vsini2_95,vmac2,vmac2_errp,vmac2_errm,vmac2_95,brv2,BRV2errp,BRV2errm,\
    f_ratio,f_ratio_err,separated,note = np.loadtxt(str(rv_file),unpack=True,delimiter=',',
                                                    dtype='U100,U100,float64, f,f,f,f,f,f,f,f,f,f,f, f,f,f,f,f,f,f,f,f,f,f, f,f,U5,U100')

brv1err = np.max([BRV1errm,BRV1errp],axis=0)
brv2err = np.max([BRV2errm,BRV2errp],axis=0)

brv1 = brv1[np.argsort(bjd)]
brv1err = brv1err[np.argsort(bjd)]
brv2 = brv2[np.argsort(bjd)]
brv2err = brv2err[np.argsort(bjd)]
bjd = bjd[np.argsort(bjd)]

bjd = bjd-2457000

t_rv = np.append(bjd,bjd)
rv = np.append(brv1,brv1-brv2)
rverr = np.append(brv1err,np.sqrt(brv1err**2 + brv2err**2))

#Labels
labels = ['t0','P','rsrp','aorpprs','cosi','sqesinw','sqecosw',  'g1_f1','g2_f1','sbr_f1','logf_lc_f1',  'g1_f2','g2_f2','sbr_f2','logf_lc_f2',  'g1_f3','g2_f3','sbr_f3','logf_lc_f3', 'f_spot','f_ecl',  'K1+K2','q','gamma','mu']

#MCMC Inputs 
mcmc_start = np.array([1712.021 - 26*10.714762,
                       10.71476726,
                       0.99387,
                       40.75077/2.0,
                       np.cos(np.pi*87.266822/180.0),
                       np.sqrt(0.294293)*np.sin(2.597),
                       np.sqrt(0.294293)*np.cos(2.597),

                       g1_val_1,
                       g2_val_1,
                       1.0,
                       -9.0,

                       g1_val_2,
                       g2_val_2,
                       1.0,
                       -9.0,

                       g1_val_3,
                       g2_val_3,
                       1.0,
                       -9.0,

                       0.3,
                       0.3,

                       71.503966,
                       1.002862,
                       23.784741,
                       0.29])

h5_name = 'TOI450_jointrvd_'+window_out+'.h5'

if (args.new == True) | (os.path.exists('./TOI450_jointrvd_'+window_out+'_mcmc_chains.p') == False):
	continue_run  = False
else:
	continue_run = True

if continue_run == False:
	print('new')

	#start_param_lc = mcmc_start[0],mcmc_start[1],mcmc_start[2],mcmc_start[3],mcmc_start[4],mcmc_start[9],mcmc_start[5],mcmc_start[6],mcmc_start[7],mcmc_start[8]
	
	pos = mcmc_start + np.array([1e-4,1e-6,1e-4,1e-4,1e-4,1e-4,1e-4, 1e-4,1e-4,1e-4,1e-4, 1e-4,1e-4,1e-4,1e-4, 1e-4,1e-4,1e-4,1e-4, 1e-4,1e-4, 1e-4,1e-4,1e-4,1e-4])*np.random.randn(mcmc_start.size*5, mcmc_start.size)
	
	nwalkers, ndim = pos.shape
	
	index = 0
	autocorr = np.zeros(max_steps)
	N = np.zeros(max_steps)
	old_tau = np.inf
	converged = False
	max_steps_ip = copy.deepcopy(max_steps)

if continue_run == True:
	print('continue')
	samples_pr,corner_samples,N,autocorr,index,old_tau,burnin = pkl.load(open('TOI450_jointrvd_'+window_out+'_mcmc_chains.p','rb'))
	
	pos = samples_pr[:,-1,:]
	nwalkers, ndim = pos.shape
	max_steps_ip = max_steps - samples_pr.shape[1]
	
	autocorr = np.concatenate([autocorr,np.zeros(max_steps-samples_pr.shape[1])])
	N = np.concatenate([N,np.zeros(max_steps-samples_pr.shape[1])])
	converged = False

sampler = emcee.EnsembleSampler(nwalkers, ndim, log_probability, 
                        		args=(t,f,ferr,lc_id, t_rv,rv,rverr,Tele),
                        		threads=n_threads)

# Now we'll sample for up to max_n steps
for sample in sampler.sample(pos, iterations=max_steps_ip, progress=False):
	# Only check convergence every 100 steps
	if converged == False:
			
		if sampler.iteration % step_check:
		    continue
		else:
			#Measure convergence 

			if continue_run == True:
				samples = sampler.chain
				samples = np.append(samples_pr,samples,axis=1)
			else:
				samples = sampler.chain
			tau = integrated_time(samples)

			#tau = sampler.get_autocorr_time(tol=0)

			autocorr[index] = np.mean(tau)
			N[index] = samples.shape[1]
			#N[index] = sampler.iteration
		
			#Check convergence
			converged = np.all(tau * 50 < samples.shape[1])
			#converged = np.all(tau * 50 < sampler.iteration)
			converged &= np.all(np.abs(old_tau - tau) / tau < 0.05)

			old_tau = tau
			index += 1
			
			if converged:
				#converge_step = sampler.chain.shape[1]
				tau_converge = copy.deepcopy(tau) #sampler.get_autocorr_time()
				burnin = int(5 * np.max(tau_converge))
				break

		if sampler.iteration % step_check:
			continue
		else:
			t_iter = time.time()
			print('Step Count:',samples.shape[1], 'Time count:',t_iter - t_start)
			if t_iter - t_start > t_max:
				print('Out of Time -- Exceeded Max:',t_max)
				break

if converged == False:
	print('')
	print('WARNING: Unable to measure convergence')
	print('')
	if t_iter - t_start > t_max:
		burnin = int(samples.shape[1]/4.0)
	else:
		burnin = int(max_steps/4.0)
	note = 'unconverged'

N = N[:index]
autocorr = autocorr[:index]

if continue_run == True:
	samples = sampler.chain
	samples = np.append(samples_pr,samples,axis=1)
else:
	samples = sampler.chain
#samples = sampler.chain

corner_samples = sampler.get_chain(discard=burnin, flat=True)

pkl.dump([samples,corner_samples,N,autocorr,index,old_tau,burnin],open('TOI450_jointrvd_'+window_out+'_mcmc_chains.p','wb'))




