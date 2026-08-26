import os
os.environ.setdefault("SPS_HOME", "/home/zesha/python-fsps/src/fsps/libfsps")  # honour an existing SPS_HOME if set
#DELETE ABOVE TWO LINES FOR USE ON HPC

import fsps
import numpy as np
import matplotlib.pyplot as plt
from astropy import units as u
import lbg_forecast.sfh as sfh
import lbg_forecast.zhistory as zh
import lbg_forecast.population_model as pop
import lbg_forecast.lyalpha as ly
import lbg_forecast.igm as igm

from astropy.cosmology import WMAP1 as cosmo
#from astropy.coordinates import Distance
from astropy.constants import L_sun
from astropy.constants import c

from sedpy import observate

def initialise_sps_model(neb_em, sfh_type=3, zcont=1, imf_type=2, igm=True):
    
    sps_model = fsps.StellarPopulation(compute_vega_mags=False, zcontinuous=zcont, imf_type=imf_type, sfh=sfh_type, dust_type=4)

    sps_model.params['dust1_index'] = -1.0
    sps_model.params['add_neb_emission'] = neb_em
    sps_model.params['add_neb_continuum'] = neb_em
    sps_model.params['nebemlineinspec'] = neb_em
    
    sps_model.params['add_igm_absorption'] = igm
    sps_model.params['imf_type'] = imf_type

    return sps_model 

def update_model(sps_model, sps_parameters, z_history, agebins, binscale=sfh.zred_to_agebins):
    
    sps_model.params['zred'] = sps_parameters[0]
    sps_model.params['logzsol'] = sps_parameters[1]
    sps_model.params['dust1'] = sps_parameters[2]
    sps_model.params['dust2'] = sps_parameters[3]
    sps_model.params['dust_index'] = sps_parameters[4]
    sps_model.params['igm_factor'] = sps_parameters[5]
    sps_model.params['gas_logu'] = sps_parameters[6]
    sps_model.params['fagn'] = sps_parameters[8]
    sps_model.params['agn_tau'] = sps_parameters[9]

    log_sfr_ratios = sps_parameters[10:-1]

    total_mass_formed = sps_parameters[-1]
    
    shifted_age_bins = binscale(sps_model.params['zred'], agebins)
    time, star_formation_history, tage = sfh.continuity_sfh(shifted_age_bins, log_sfr_ratios, total_mass_formed)[0]
    sps_model.params['tage'] = tage

    if(z_history):
        Z_MIST = 0.0142 #solar metallicity for MIST
        sps_model.params['logzsol'] = 0.0
        sps_model.params['gas_logz'] = 0.0
        metallicity_history = zh.sfr_to_zh(star_formation_history, time,
                                            (10**sps_model.params['logzsol'])*Z_MIST, sps_parameters[-1])
        sps_model.set_tabular_sfh(time, star_formation_history, metallicity_history)
    else:
        sps_model.params['gas_logz'] = sps_parameters[7]
        sps_model.set_tabular_sfh(time, star_formation_history) 


def simulate_photometry(sps_parameters, filters, imf, nebem=True, zhistory=True, agebins=None, enable_mpi=False, lya_uncertainity=False, mpi_rank=0, save_spec=False, run_count=0, path="./"):

    ngalaxies = sps_parameters.shape[0]
    cosmology=cosmo

    if(nebem == False and zhistory == True):
        raise Exception("nebular emission cannot be turned off with zhistory enabled at present")
    
    if agebins is None:
        agebins = sfh.default_agebins()

    print("Starting Run")
    sps_model = initialise_sps_model(neb_em=nebem, sfh_type=3, zcont=1, imf_type=imf)
    indx = ly.find_wave_range(sps_model.wavelengths, 1215.67, 100)#for saving lya peak
    print("libraries: ", sps_model.libraries)

    i = 0
    photometry = []
    spectra = []
    while(i < ngalaxies):

        source = sps_parameters[i, :]
        update_model(sps_model, source, z_history=False, agebins=agebins)

        #generate photometry for source
        if(save_spec):
            phot, spec = get_magnitudes(sps_model, filters=filters, cosmology=cosmology, lya_uncertainity=lya_uncertainity, return_spec=True, path=path)
            photometry.append(phot)
            spectra.append(spec[indx])
        else:
            photometry.append(get_magnitudes(sps_model, filters=filters, cosmology=cosmology, lya_uncertainity=lya_uncertainity, path=path))

        i+=1
        if(i%1000 == 0 and mpi_rank==0):
            print(i)

    photometry = np.vstack(np.asarray(photometry))
    if(save_spec):
        spectra = np.vstack(np.asarray(spectra))

    print("Complete")

    if(enable_mpi):
        np.save(path+"simulation_data/simulated_photometry_"+str(mpi_rank+run_count)+".npy", photometry)
        np.save(path+"simulation_data/sps_parameters_"+str(mpi_rank+run_count)+".npy", sps_parameters)
        if(save_spec):
            np.save(path+"simulation_data/spectra_"+str(mpi_rank+run_count)+".npy", spectra)

    else:
        np.save(path+"simulation_data/simulated_photometry.npy", photometry)
        np.save(path+"simulation_data/sps_parameters.npy", sps_parameters)
        if(save_spec):
            np.save(path+"simulation_data/spectra.npy", spectra)

    if(mpi_rank==0):
        np.save(path+"simulation_data/wavelengths_"+str(mpi_rank)+".npy", sps_model.wavelengths[indx])

    return photometry


def fsps_get_sed(sps_model):
    
    tage = sps_model.params['tage']
    angstroms, spectrum = sps_model.get_spectrum(tage=tage, peraa=True)

    return angstroms, spectrum

def fsps_get_magnitudes(sps_model, filters):

    if(filters=='lsst'):
        bands = fsps.find_filter('lsst')
    if(filters=='suprimecam'):
        bands = fsps.find_filter('suprimecam')[1:2]+fsps.find_filter('suprimecam')[3:]
    if(filters=='all'):
        bands = fsps.find_filter('lsst') + fsps.find_filter('suprimecam')[1:2]+fsps.find_filter('suprimecam')[3:]

    mags = sps_model.get_mags(tage=sps_model.params['tage'], bands=bands)
    return mags# - 2.5*logmass


def get_magnitudes(sps_model, filters, cosmology, modify_igm=False, lya_uncertainity=False, return_spec=False, path="./"):
    
    lsun = L_sun.cgs.value
    redshift = sps_model.params['zred']
    f_igm = sps_model.params['igm_factor']

    lambdas, spectrum = sps_model.get_spectrum(tage=sps_model.params['tage'], peraa=True)

    if(modify_igm):
        att = igm.apply_igm_attenuation(lambdas, redshift, f_igm)
        spectrum = spectrum*att

    if(lya_uncertainity):
        lya_width = np.random.uniform(30, 100)
        lya_bias = 0
        a_param = np.random.uniform(0, 1)
        spectrum = ly.modify_peak(lambdas, spectrum, 60, lya_width, lya_bias, a_param, diagnostics=False)

    luminosity_distance = cosmology.luminosity_distance(redshift).cgs.value

    redshifted_spectrum = redshift_fsps_spectrum(sps_model, (lambdas*(1+redshift), spectrum)) #lsun aa-1

    redshifted_spectrum_cgs = redshifted_spectrum*(lsun/(4.0*np.pi*(luminosity_distance**2))) #erg s-1 cm-2 aa-1
    redshifted_spectrum_sedpy = redshifted_spectrum_cgs #erg s-1 cm-2 aa-1
    if(filters == 'lsst'):
        bands = get_lsst_filters(path)
    elif(filters == 'suprimecam'):
        bands = get_suprimecam_filters()
    else:
        raise Exception("invalid filters")

    magnitudes = observate.getSED(lambdas, redshifted_spectrum_sedpy, filterlist=bands, linear_flux=False)
    
    if(return_spec):
        return magnitudes, (lambdas, redshifted_spectrum_sedpy)
    else:
        return magnitudes

def redshift_fsps_spectrum(sps_model, spec):
    """Takes output of fsps_get_sed() and redshifts spectrum.
    Used in get_magnitudes() too. Wavelength grid is kept the same.
    Spectrum must be in Lsun/AA. Wavelength in AA.
    
    """


    lambdas, spectrum = spec
    redshift = sps_model.params['zred']
    redshifted_spectrum = np.interp(lambdas, lambdas*(1+redshift), spectrum)/(1+redshift)

    return redshifted_spectrum 




def _build_filter(kname, wave, trans):
    """Build a sedpy Filter straight from transmission arrays.

    sedpy's ``data=`` keyword sets ``filename`` to None and then type-checks it,
    so on older sedpy versions that path always raises TypeError even though the
    filter itself was built correctly. Fall back to assembling the object
    directly when that happens.
    """
    try:
        return observate.Filter(kname, data=(wave, trans))
    except TypeError:
        filt = observate.Filter.__new__(observate.Filter)
        filt.name = kname
        filt.nick = kname
        filt.min_trans = 1e-5
        filt.filename = None
        filt._process_filter_data(np.asarray(wave), np.asarray(trans))
        filt.get_properties()
        return filt


def get_lsst_filters(path):
            
    ufltr = fsps.filters.Filter(144, 'lsst_u', 'lsst')
    gfltr = fsps.filters.Filter(145, 'lsst_g', 'lsst')
    rfltr = fsps.filters.Filter(146, 'lsst_r', 'lsst')
    ifltr = fsps.filters.Filter(147, 'lsst_i', 'lsst')
    zfltr = fsps.filters.Filter(148, 'lsst_z', 'lsst')
    yfltr = fsps.filters.Filter(149, 'lsst_y', 'lsst')

    filters = []
    for band in ['u', 'g', 'r', 'i', 'z', 'y']:
        filter_data = np.genfromtxt(path+'lbg_forecast/lsst_filters/total_'+band+'.dat', skip_header=7, delimiter=' ')
        filter_data[:, 0] = filter_data[:, 0]*10 #covert to angstroms
        filters.append(_build_filter("lsst_"+band, filter_data[:, 0], filter_data[:, 1]))
    
    return filters

#for homebrew get_mags
def get_lsst_filters_fsps():

    filters = []
    for band in ['u', 'g', 'r', 'i', 'z']:
        filter_data = np.genfromtxt('lbg_forecast/lsst_filters_fsps/total_'+band+'.dat', skip_header=1, delimiter=' ')
        filters.append([filter_data[:, 0], filter_data[:, 1]])
    
    return filters

#for homebrew get_mags
def get_suprimecam_filters():

    gfltr = observate.Filter("hsc_g")
    rfltr = observate.Filter("hsc_r")
    ifltr = observate.Filter("hsc_i")
    zfltr = observate.Filter("hsc_z")
    yfltr = observate.Filter("hsc_y")

    filters = np.array([gfltr, rfltr, ifltr, zfltr, yfltr])

    return filters

def plot_lsst_filters(factor):

    ufltr = fsps.filters.Filter(144, 'lsst_u', 'lsst')
    gfltr = fsps.filters.Filter(145, 'lsst_g', 'lsst')
    rfltr = fsps.filters.Filter(146, 'lsst_r', 'lsst')
    ifltr = fsps.filters.Filter(147, 'lsst_i', 'lsst')
    zfltr = fsps.filters.Filter(148, 'lsst_z', 'lsst')
    yfltr = fsps.filters.Filter(149, 'lsst_y', 'lsst')

    plt.plot(ufltr.transmission[0], ufltr.transmission[1]*factor)
    plt.plot(gfltr.transmission[0], gfltr.transmission[1]*factor)
    plt.plot(rfltr.transmission[0], rfltr.transmission[1]*factor)
    plt.plot(ifltr.transmission[0], ifltr.transmission[1]*factor)
    plt.plot(zfltr.transmission[0], zfltr.transmission[1]*factor)
    plt.plot(yfltr.transmission[0], yfltr.transmission[1]*factor)

    return [ufltr, gfltr, rfltr, ifltr, zfltr, yfltr]

def plot_suprimecam_filters(factor):

    gfltr = fsps.filters.Filter(134, 'suprimecam_g', 'suprimecam')
    rfltr = fsps.filters.Filter(136, 'suprimecam_r', 'suprimecam')
    ifltr = fsps.filters.Filter(137, 'suprimecam_i', 'suprimecam')
    zfltr = fsps.filters.Filter(138, 'suprimecam_z', 'suprimecam')

    plt.plot(gfltr.transmission[0], gfltr.transmission[1]*factor)
    plt.plot(rfltr.transmission[0], rfltr.transmission[1]*factor)
    plt.plot(ifltr.transmission[0], ifltr.transmission[1]*factor)
    plt.plot(zfltr.transmission[0], zfltr.transmission[1]*factor)

def _fsps_lumdist(z):
    c_light = 2.9979E18
    dhub = c_light/1E13/72*1E6
    size = 10000
    i = 1
    zz = []
    while(i < size+1):
        zz.append((i/size)*z)
        i+=1
    zz = np.array(zz)
    hub = np.sqrt(0.27*(np.ones_like(zz)+zz)**3+0.73)
    luminosity_distance = np.trapz(1/hub, zz) * (1+z) * dhub
    return luminosity_distance

def fsps_cloned_get_magnitudes(sps_model, filters, modify_igm):
    lambdas, spectrum = sps_model.get_spectrum(tage=sps_model.params['tage'], peraa=False)

    if(modify_igm):
        att = igm.apply_igm_attenuation(lambdas, sps_model.params['zred'], 1.0)
        spectrum = spectrum*att

    redshift = sps_model.params['zred']
    redshifted_spectrum = np.interp(lambdas, lambdas*(1+redshift), spectrum)

    lsun = 3.839E33
    pc2cm = 3.08568E18
    mypi = 3.14159265
    c_light = 2.9979E18


    a = (np.linspace(1, 500, 500)-1)/499.*(1-1/1001.)+1/1001
    redshift_grid = 1/a-1

    luminosity_distances = []
    for red in redshift_grid:
        luminosity_distances.append(_fsps_lumdist(red))

    luminosity_distances = np.flip(np.array(luminosity_distances))

    luminosity_distance = _fsps_lumdist(redshift)
    #luminosity_distance = cosmo.luminosity_distance(redshift).cgs.value

    dm = 5.0*np.log10(luminosity_distance/10)
    const = dm - 2.5*np.log10(1+redshift)
    mag2cgs = np.log10(lsun/(4.0*mypi*pc2cm*pc2cm*100))

    if(filters == 'lsst'):
        bands = get_lsst_filters()
    if(filters == 'fsps'):
        bands = get_lsst_filters_fsps()
    if(filters == 'suprimecam'):
        bands = get_suprimecam_filters()

    magnitudes = []
    for band in bands:

        if(filters == 'fsps'):
            i = 1
            imax = 50000
            trans_lambdas = np.zeros(imax)
            trans = np.zeros(imax)
            j=0
            
            while(i < imax+1):
                trans_lambdas[i-1] = i
                trans[i-1] = 0.0
                if(j < len(band[0])):
                    if(i == band[0][j]):
                        trans[i-1] = band[1][j]
                        j+=1
                i+=1

            trans_v = trans#/(c_light/trans_lambdas)
            trans_v = np.maximum(trans_v, np.zeros_like(trans_v))
            trans_v = np.interp(lambdas, trans_lambdas, trans_v)

            #trans = trans[1:]
            #trans_lambdas = trans_lambdas[1:]
        else:
            trans = band[1]
            trans_lambdas = band[0]

            trans_v = trans#/(c_light/trans_lambdas)
            trans_v = np.maximum(trans_v, np.zeros_like(trans_v))
            trans_v = np.interp(lambdas, trans_lambdas, trans_v)

        trans_v = trans_v/np.trapz(trans_v/lambdas, lambdas)
        trans_v = np.maximum(trans_v, np.zeros_like(trans_v))

        mags = np.trapz(redshifted_spectrum*trans_v/lambdas, lambdas)
        mags = -2.5*np.log10(mags) - 48.60 - 2.5*mag2cgs + const
        magnitudes.append(mags)
    
    return np.array(magnitudes)
