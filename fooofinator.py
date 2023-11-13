import scipy
import numpy as np
from fooof import FOOOF
from fooof.sim.gen import gen_aperiodic, gen_periodic
from scipy.ndimage import gaussian_filter1d

from lagged_coherence import lagged_hilbert_coherence


def fooofinator(data, fs, freqs, alpha=0.1, lags=np.arange(1.0,2.0,.1), n_jobs=-1):
    f, psd = scipy.signal.welch(data, fs=fs, window='hann',
                                nperseg=fs, noverlap=int(fs / 2), nfft=fs * 2, detrend='constant',
                                return_onesided=True, scaling='density', axis=- 1, average='mean')
    f_idx = (f >= freqs[0]) & (f <= freqs[-1])
    f = f[f_idx]
    psd = np.mean(psd[:, f_idx], axis=0)

    # Fit the aperiodic component
    lc_hilbert = lagged_hilbert_coherence(data, f, lags, fs, n_shuffles=100, n_jobs=n_jobs)

    sigma = 1.0
    lc_smooth = gaussian_filter1d(np.squeeze(np.nanmean(np.nanmean(lc_hilbert, axis=-1), axis=0)), sigma)
    lc_smooth = lc_smooth - np.min(lc_smooth)
    lc_smooth = lc_smooth / np.max(lc_smooth)

    def gen_parameterized_spectrum(freqs, params):
        # Aperiodic component
        offset = params[0]
        slope = params[1]
        aper = offset + slope * np.log10(1. / (freqs))

        spec = aper + params[2] * lc_smooth
        return spec, aper

    # Fit log PSD
    fit_target = np.log10(psd)

    def err_func(params):
        spec,aper = gen_parameterized_spectrum(f, params)
        resid = fit_target - aper

        # Check for NaNs or overlapping Gaussians
        # if np.any(np.isnan(spec)) or check_gaussian_overlap(params):
        if np.any(np.isnan(spec)):
            return 1000000

        err = np.sqrt(np.sum(np.power(spec - fit_target, 2)))
        cost = np.sum(np.abs(resid[resid < 0]))

        return err+alpha+cost

    init_params = [fit_target[0],
                   fit_target[-1] - fit_target[0],
                   1]
    bounds = [(None, None),
              (None, None),
              (0, None)]
    method = 'SLSQP'
    xopt = scipy.optimize.minimize(err_func, init_params, method=method, bounds=bounds, options={'disp': False})
    params = xopt.x

    fm = FOOOF(aperiodic_mode='fixed', verbose=False)
    fm.fit(freqs=f, power_spectrum=psd)
    fm.aperiodic_params_ = [params[0], params[1]]
    fm._ap_fit = gen_aperiodic(fm.freqs, fm.aperiodic_params_)

    # Flatten the power spectrum using fit aperiodic fit
    fm._spectrum_flat = fm.power_spectrum - fm._ap_fit

    # Find peaks, and fit them with gaussians
    fm.gaussian_params_ = fm._fit_peaks(np.copy(fm._spectrum_flat))

    # Calculate the peak fit
    #   Note: if no peaks are found, this creates a flat (all zero) peak fit
    fm._peak_fit = gen_periodic(fm.freqs, np.ndarray.flatten(fm.gaussian_params_))

    # Create peak-removed (but not flattened) power spectrum
    fm._spectrum_peak_rm = fm.power_spectrum - fm._peak_fit

    # Create full power_spectrum model fit
    fm.fooofed_spectrum_ = fm._peak_fit + fm._ap_fit

    # Convert gaussian definitions to peak parameters
    fm.peak_params_ = fm._create_peak_params(fm.gaussian_params_)

    # Calculate R^2 and error of the model fit
    fm._calc_r_squared()
    fm._calc_error()

    return fm, lc_smooth, psd