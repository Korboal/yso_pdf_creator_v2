import scipy.optimize
import numpy as np
import pylab as plt
from astropy.timeseries import LombScargle as astropyLS
import config_file as cf
from typing import Union, Tuple, Callable, Any, List
from scipy.stats import skew, binned_statistic
from scipy.signal import lombscargle as scipy_lombscargle
import pandas
from matplotlib.pyplot import cm
from gatspy.periodic.lomb_scargle_multiband import LombScargleMultiband
from gatspy.periodic.lomb_scargle import LombScargle
import matplotlib
# from matplotlib.ticker import AutoMinorLocator, AutoLocator, MultipleLocator, FormatStrFormatter
from matplotlib.widgets import Slider, Button
from copy import deepcopy
from matplotlib.colors import Normalize

matplotlib.use('macosx')

ls_freq_start = 0.001  # 1000 days
ls_freq_stop = 0.1  # 10 days
ls_freq_stop = 2  # 0.5 day
ls_freq_stop = 0.5  # 2 days

# ls_freq_start = 0.5  # 2 days   for very short term variability
# ls_freq_stop = 100  # 14.4 min  for very short term variability

ls_freq_time_step = 0.00005
# ls_freq_time_step = 0.0001 # for very short term variability
# ls_freq_time_step = 0.00001

ls_freqs_input = np.arange(ls_freq_start, ls_freq_stop + ls_freq_time_step, ls_freq_time_step)

max_measurements_per_night = 10000000
min_time_between_measurements = 1 / 24  # 1 hour
remove_ztf_measurements_if_too_close = True

remove_1_day_peak_manually_ztf = False

alpha_periodogram_vlines = 0.5  # 0 to 1


class LightCurve:
    """
    Light Curve class, that should be inherited by individual classes. Contains light curve properties and can do
    fittings/plots.
    """
    def __init__(self, band_name: str, light_curve_name: str, save_image: bool, save_pdfs: bool, show_image: bool):
        self.number_of_points_on_fitting_function = 50000
        self.band_name, self.light_curve_name = band_name, light_curve_name

        self.fit_result, self.fitting_successful, self.period_fit, self.ls_freqs, self.ls_power, self.ls_guess_freq = None, None, None, None, None, None
        self.fit_result_constant_period, self.fitting_successful_constant_period, self.period_fit_constant_period = None, None, None
        self.data_t, self.data_y, self.data_error, self.data_length = None, None, None, None
        self.data_rej_t, self.data_rej_y, self.data_rej_error, self.data_rej_length = None, None, None, None

        self.max_diff, self.percentile_q95_5, self.st_dev, self.avg_error = None, None, None, None
        self.mean = None
        self.skewness = None
        self.fit_function_error = None
        self.total_obs_time = None
        self.nrmse_fit = None

        self.save_image, self.save_pdfs, self.show_image = save_image, save_pdfs, show_image

    def calculate_light_curve_properties(self):
        if self.data_length > 2:
            y_values, error_values = np.asarray(self.data_y), np.asarray(self.data_error)
            self.max_diff = np.max(y_values) - np.min(y_values)
            self.percentile_q95_5 = np.percentile(y_values, 95) - np.percentile(y_values, 5)
            self.st_dev = np.std(y_values)
            self.avg_error = np.mean(error_values)
            self.skewness = skew(y_values)
            self.mean = np.mean(y_values)
            self.total_obs_time = np.max(self.data_t) - np.min(self.data_t)
        else:
            self.max_diff, self.percentile_q95_5, self.st_dev, self.avg_error = cf.dflt_no_vle, cf.dflt_no_vle, cf.dflt_no_vle, cf.dflt_no_vle
            self.mean, self.skewness, self.total_obs_time = cf.dflt_no_vle, cf.dflt_no_vle, cf.dflt_no_vle

    def calculate_lightcurve_fit_error(self):
        """
        Calculate the error of fitted function
        """
        if self.fitting_successful:
            expected = self.fit_result["fitfunc"](self.data_t)
            self.fit_function_error = np.sum(np.square(self.data_y - expected))
            amp_fit = np.abs(self.fit_result["amp"])  # the amplitude of the fit
            self.nrmse_fit = np.sqrt(
                self.fit_function_error / self.data_length) / amp_fit  # normalised root mean square error from fit
        else:
            self.fit_function_error = cf.dflt_no_vle
            self.nrmse_fit = cf.dflt_no_vle

    def get_values(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
        """
        Returns the light curve data points as arrays with a length of the array
        :return: Time array, mag array, error array, length of time array
        """
        return self.data_t, self.data_y, self.data_error, self.data_length

    def fit_light_curve(self, print_variables=False, manual_period_guess=0, fit_mean=True, center_data=True,
                        normalization='standard', method='cython'):
        """
        Fits the light curve, if it has enough points. Non-linear fit based on LS peak guess, assuming global parameter constant period is not used.
        :param print_variables: Whether to print variables of fittings
        :param manual_period_guess: Overrides the LS peak guess if not 0
        :param fit_mean: Whether to use parameter fit mean in astropy LS guess
        :param center_data: Whether to use parameter center mean in astropy LS guess
        :param normalization: Normalisation for the LS. Usually either 'standard' or 'psd'
        :param method: Method to use for LS computation. Default 'cython', can be also 'slow', 'auto' etc
        """
        if self.data_length > cf.minimum_data_points_to_fit_light_curve:  # Try to fit if at least some points
            res, ls_freqs, ls_power, ls_guess_freq = non_linear_fit_sin(self.data_t, self.data_y, self.data_error,
                                                                        manual_period_guess, type(self),
                                                                        fit_mean=fit_mean, center_data=center_data,
                                                                        normalization=normalization, method=method)

            if cf.do_fit_with_constant_period:
                res = fit_sin_with_const_period(self.data_t, self.data_y, self.data_error, manual_period_guess)

            if print_variables:
                if not res['fit_success']:
                    print(f"Could not fit period for {self.light_curve_name} {self.band_name}")
                print(f"Light curve {self.light_curve_name} {self.band_name}")
                print(f"Guess period: {1 / ls_guess_freq}")
                print(f"Amplitude: {res['amp']}")
                print(f"Phase: {res['phase']}")
                print(f"Offset: {res['offset']}")
                print(f"Period: {res['period']}")
                print(f"Frequency: {res['freq']}")

        else:  # If not enough points, then no fit
            res = {"amp": 0, "omega": 0, "phase": 0, "offset": 0, "freq": 0, "period": 0, "fitfunc": 0, "maxcov": 0,
                   "rawres": (0, 0, 0), "fit_success": False, "ls_fap": cf.dflt_no_vle}
            ls_freqs, ls_power, ls_guess_freq, = [], [], cf.dflt_no_vle

        self.fit_result, self.fitting_successful, self.period_fit, self.ls_freqs, \
        self.ls_power, self.ls_guess_freq = res, res['fit_success'], res['period'], ls_freqs, ls_power, ls_guess_freq

        self.calculate_lightcurve_fit_error()

    def fit_light_curve_using_two_periods(self, period1=None, period2=None):
        """
        Tries to fit light curve using two periods. Guesses are done based on two LS peaks in original LS periodogram, if no periods are passed
        :param period1: First period to use for guessing
        :param period2: Second period to use for guessing
        """
        time, y_values, error, points_amount = self.data_t, self.data_y, self.data_error, self.data_length

        if points_amount > cf.minimum_data_points_to_fit_light_curve:  # Try to fit if at least some points
            time = np.asarray(time)
            y_values = np.asarray(y_values)

            freq_start = 0.001
            freq_stop = 0.501
            time_step = 0.00005

            freqs = np.arange(freq_start, freq_stop + time_step, time_step)  # Guess based on Lomb Scargle
            power = astropyLS(time, y_values).power(freqs)

            if period1 is None or period2 is None:
                guess_frequencies, _ = find_n_peaks_periodogram(freqs, power, 2)

            if period1 is None:
                guess_freq = guess_frequencies[0]
            else:
                guess_freq = 1 / period1

            if period2 is None:
                guess_freq_2 = guess_frequencies[1]
            else:
                guess_freq_2 = 1 / period2

            print(f"Double period guesses: {1 / guess_freq} and {1 / guess_freq_2} days")

            guess_amp = np.std(y_values) * 2. ** 0.5  # Guess for the scipy
            guess_offset = np.mean(y_values)
            guess = np.array(
                [guess_amp, 2. * np.pi * guess_freq, 0., guess_offset, guess_amp, 2. * np.pi * guess_freq_2, 0.])

            def sin_func(t, A, w, p, c, B, w2, p2):  # Function plot
                return (A * np.sin(w * t + p) + c) + B * np.sin(w2 * t + p2)

            try:  # Try to optimize
                fit_values, fit_values_cov = scipy.optimize.curve_fit(sin_func, time, y_values, method='lm',
                                                                      p0=guess)  # , sigma=error)
                A, w, p, c, B, w2, p2 = fit_values
                f = w / (2. * np.pi)
                period_fit = abs(1.0 / f)

                f2 = w2 / (2. * np.pi)
                period_fit2 = abs(1.0 / f2)

                function_fit = lambda t: (A * np.sin(w * t + p) + c) + B * np.sin(w2 * t + p2)
                fit_success = True
            except:  # If optimization took too long, abort and go here
                A, w, p, c, f, fit_values_cov, fit_values, function_fit, period_fit, period_fit2 = 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
                fit_success = False
                p2 = 0

            fit_result = {"amp": A, "omega": w, "phase": p, "offset": c, "freq": f, "period": period_fit,
                          "period2": period_fit2, "fitfunc": function_fit, "phase2": p2,
                          "maxcov": np.max(fit_values_cov), "rawres": (guess, fit_values, fit_values_cov),
                          "fit_success": fit_success}

            time_rej, y_values_rej, error_rej, points_amount_rej = self.data_rej_t, self.data_rej_y, self.data_rej_error, self.data_rej_length

            draw_points_err_bar(time, y_values, error, label="Data", alpha=0.5)
            draw_points_err_bar(time_rej, y_values_rej, error_rej, fmt='x', label="Rejected data", color="green")

            print(f"Fit curves: {round(fit_result['period'], 5)} d; {round(fit_result['period2'], 5)} d")

            plot_function(time, fit_result["fitfunc"],
                          f"Fit curves: {round(fit_result['period'], 2)} d; {round(fit_result['period2'], 2)} d",
                          alpha=0.75, draw_function=fit_success)

            prepare_plot(f"Fit of {self.band_name} {self.light_curve_name} double period", "Time [days]",
                         f"{self.band_name} {self.light_curve_name} band [mag]", True, False)

            if fit_result["period"] > fit_result["period2"]:
                func1 = {"period": fit_result["period"], "phase": fit_result["phase"]}
                func2 = {"period": fit_result["period2"], "phase": fit_result["phase2"]}
                function_fit_temp = lambda t: A * np.sin(w * t + p)
            else:
                func2 = {"period": fit_result["period"], "phase": fit_result["phase"]}
                func1 = {"period": fit_result["period2"], "phase": fit_result["phase2"]}
                function_fit_temp = lambda t: B * np.sin(w2 * t + p2)

            if fit_success:
                self.draw_folded_light_curve_with_other_function(func1, "first longer")

                temp_copy = deepcopy(self)
                temp_copy.data_y = temp_copy.data_y - function_fit_temp(temp_copy.data_t)
                temp_copy.draw_folded_light_curve_with_other_function(func2, "second shorter")

            return fit_result

    def fit_light_curve_with_const_period(self, period_to_fit: float, print_variables=False):
        """
        Fits the light curve using constant period
        :param period_to_fit: Period to fit
        :param print_variables: Whether to print the variables from fitted light curve
        """
        time, y_values, error, points_amount = self.data_t, self.data_y, self.data_error, self.data_length

        period_to_fit = abs(period_to_fit)
        if period_to_fit == 0:
            raise ValueError("Asking to fit period that is equal to 0")

        if points_amount > cf.minimum_data_points_to_fit_light_curve:  # Try to fit if at least some points
            res = fit_sin_with_const_period(time, y_values, error, period_to_fit)
            if print_variables:
                if not res['fit_success']:
                    print(f"Could not fit period for {self.light_curve_name} {self.band_name}")
                print(f"Light curve {self.light_curve_name} {self.band_name}")
                print(f"Amplitude: {res['amp']}")
                print(f"Phase: {res['phase']}")
                print(f"Offset: {res['offset']}")
                print(f"Period: {res['period']}")
                print(f"Frequency: {res['freq']}")
        else:  # If not enough points, then no fit
            res = {"amp": 0, "omega": 0, "phase": 0, "offset": 0, "freq": 0, "period": 0, "fitfunc": 0, "maxcov": 0,
                   "rawres": (0, 0, 0), "fit_success": False}

        self.fit_result_constant_period, self.fitting_successful_constant_period, \
        self.period_fit_constant_period = res, res['fit_success'], res['period']

    def draw_raw_light_curve(self, image_output_png=None, image_output_pdf=None):
        """
        Draws the original light curve without any fits/folding/etc
        :param image_output_png: Output .png file location string
        :param image_output_pdf: Output .pdf file location string
        """
        if self.save_image or self.show_image:
            draw_points_err_bar(self.data_t, self.data_y, self.data_error, label="Data")
            draw_points_err_bar(self.data_rej_t, self.data_rej_y, self.data_rej_error, fmt='x', label="Rejected data")
            draw_empty_graph([self.data_length, self.data_rej_length])

            prepare_plot(f"Original data of {self.light_curve_name} {self.band_name}", "Time [days]",
                         f"{self.light_curve_name} {self.band_name} band [mag]", self.show_image, self.save_image,
                         image_output_png=image_output_png, image_output_pdf=image_output_pdf, save_pdf=self.save_pdfs)

    def draw_fitted_light_curve(self, image_output_png=None, image_output_pdf=None, do_second_fit=False,
                                second_fit_period=0, second_fit_name="Other"):
        """
        Draws the light curve with the fit (if it exists, otherwise just original light curve). Can draw second fit using passed period
        :param image_output_png: Output .png file location string
        :param image_output_pdf: Output .pdf file location string
        :param do_second_fit: True/False, whether to draw second fit on top
        :param second_fit_period: Period for the second fit (will optimize all parameters with constant period for this fit)
        :param second_fit_name: Name of the second fit for the label
        """
        if self.save_image or self.show_image:
            from tools import check_if_not_nan_or_zero

            if self.data_length > 0:
                if self.fit_result is None:
                    self.fit_light_curve(False)

                draw_points_err_bar(self.data_t, self.data_y, self.data_error, label="Data")
                draw_points_err_bar(self.data_rej_t, self.data_rej_y, self.data_rej_error, fmt='x',
                                    label="Rejected data", color="green")
                plot_function(self.data_t, self.fit_result["fitfunc"], f"Fit curve: {round(self.period_fit, 2)} d",
                              alpha=0.5, draw_function=self.fitting_successful, color="red")

                if do_second_fit:
                    # Draw other fot on top
                    if check_if_not_nan_or_zero(second_fit_period):
                        time_function = np.linspace(np.min(self.data_t), np.max(self.data_t),
                                                    self.number_of_points_on_fitting_function)
                        try:  # Try to fit curve as well with known period
                            second_fit_res = fit_sin_with_const_period(self.data_t, self.data_y, self.data_error,
                                                                       second_fit_period)
                            y_gaia = second_fit_res["fitfunc"](time_function)
                        except:  # if fit doesnt work, use either fitted variables or just guess to approximate
                            w = 2.0 * np.pi / second_fit_period
                            if self.fitting_successful:  # If fit worked before, use same values
                                A, w, p, c = self.fit_result['rawres'][1]
                            else:  # If fitting didn't work, just make a rough guess
                                A = np.std(self.data_y) * 2. ** 0.5
                                c = np.mean(self.data_y)
                                p = 0
                            fitfunc_second_fit = lambda t: A * np.sin(w * t + p) + c
                            y_gaia = fitfunc_second_fit(time_function)

                        plt.plot(time_function, y_gaia, label=f"{second_fit_name}: {round(second_fit_period, 2)} d",
                                 linewidth=2, color="b", alpha=0.5)
            else:
                draw_empty_graph([0])

            prepare_plot(f"Fit of {self.band_name} {self.light_curve_name}", "Time [days]",
                         f"{self.band_name} {self.light_curve_name} band [mag]", self.show_image, self.save_image,
                         image_output_png=image_output_png, image_output_pdf=image_output_pdf, save_pdf=self.save_pdfs)

    def draw_periodogram(self, image_output_png=None, image_output_pdf=None, second_fit_periods=None,
                         second_fit_names=None, find_peaks=False) -> Tuple[float, float, float, float]:
        """
        Draws periodogram in semilogx graph with period on x-axis. The peak is highlighted as a vertical line.
        :param image_output_png: Output .png file location string
        :param image_output_pdf: Output .pdf file location string
        :param second_fit_periods: Array of other periods to highlight as vertical lines
        :param second_fit_names: The names of other periods as vertical lines for label
        :param find_peaks: Whether to find the value of two most significant peaks and their locations
        :return: Period location of 1st peak, value of 1st peak, period location of 2nd peak, value of 2nd peak
        """
        if second_fit_periods is None:
            second_fit_periods = []
        if self.save_image or self.show_image:
            import tools

            if self.fit_result is None:
                self.fit_light_curve()

            if self.ls_guess_freq > -90:
                y_lim_max = np.max(self.ls_power)

                ls_periods = 1 / self.ls_freqs

                plt.semilogx(ls_periods, self.ls_power, label="Periodogram", color="green")

                if self.fitting_successful:
                    plt.vlines(self.period_fit, 0, y_lim_max,
                               label="Fitted period " + self.light_curve_name + " " + self.band_name + " " + str(
                                   round(self.period_fit, 2)), color="black", alpha=alpha_periodogram_vlines)
                else:
                    plt.vlines(1. / self.ls_guess_freq, 0, y_lim_max,
                               label="Guess period " + self.light_curve_name + " " + self.band_name + " " + str(
                                   round(1. / self.ls_guess_freq, 2)), color="black", alpha=alpha_periodogram_vlines)

                if second_fit_periods is not None:
                    n = min(np.size(second_fit_periods), np.size(second_fit_names))
                    color = iter(cm.rainbow(np.linspace(0, 1, n + 1)))
                    for i in range(n):
                        c = next(color)
                        if tools.check_if_not_nan_or_zero(second_fit_periods[i]):
                            plt.vlines(second_fit_periods[i], 0, y_lim_max,
                                       label=second_fit_names[i] + " period " + str(round(second_fit_periods[i], 2)),
                                       color=c, alpha=alpha_periodogram_vlines)

                prepare_plot(f"Lomb-Scargle Periodogram {self.light_curve_name} {self.band_name}", 'Period [days]',
                             'Lomb-Scargle Power', self.show_image, self.save_image,
                             image_output_png=image_output_png, image_output_pdf=image_output_pdf,
                             save_pdf=self.save_pdfs, xlim=(np.min(ls_periods) * 0.95, np.max(ls_periods) * 1.05),
                             ylim=(0, y_lim_max), invert_yaxis=False)
            else:
                draw_empty_graph([0], label="Not enough points to make periodogram")
                prepare_plot(f"Lomb-Scargle Periodogram {self.light_curve_name} {self.band_name}", 'Period [days]',
                             'Lomb-Scargle Power', self.show_image, self.save_image,
                             image_output_png=image_output_png, image_output_pdf=image_output_pdf,
                             save_pdf=self.save_pdfs, invert_yaxis=False)
        if find_peaks:
            if self.ls_guess_freq > -90:
                freq1, peak1_value, freq2, peak2_value = find_two_peaks_periodogram(self.ls_freqs, self.ls_power)
                return 1. / freq1, peak1_value, 1. / freq2, peak2_value
            else:
                return cf.dflt_no_vle, cf.dflt_no_vle, cf.dflt_no_vle, cf.dflt_no_vle

    def draw_frequency_gram(self, image_output_png=None, image_output_pdf=None, second_fit_periods=None,
                            second_fit_names=None):
        """
        Draws periodogram in graph with frequency on x-axis. The peak is highlighted as a vertical line.
        :param image_output_png: Output .png file location string
        :param image_output_pdf: Output .pdf file location string
        :param second_fit_periods: Array of other periods to highlight as vertical lines
        :param second_fit_names: The names of other periods as vertical lines for label
        """
        if second_fit_periods is None:
            second_fit_periods = []
        if self.save_image or self.show_image:
            import tools

            if self.fit_result is None:
                self.fit_light_curve(False)

            if self.ls_guess_freq > -90:
                y_lim_max = np.max(self.ls_power)

                # plt.semilogx(self.ls_freqs, self.ls_power, label="Frequencies power", color="green")
                plt.plot(self.ls_freqs, self.ls_power, label="Frequencies power", color="green")

                if self.fitting_successful:
                    plt.vlines(1. / self.period_fit, 0, y_lim_max,
                               label="Fitted period " + self.light_curve_name + " " + self.band_name + " " + str(
                                   round(self.period_fit, 2)) + " days", color="black", alpha=alpha_periodogram_vlines)
                else:
                    plt.vlines(self.ls_guess_freq, 0, y_lim_max,
                               label="Guess period " + self.light_curve_name + " " + self.band_name + " " + str(
                                   round(1. / self.ls_guess_freq, 2)) + " days", color="black",
                               alpha=alpha_periodogram_vlines)

                if second_fit_periods is not None:
                    n = min(np.size(second_fit_periods), np.size(second_fit_names))
                    color = iter(cm.rainbow(np.linspace(0, 1, n + 1)))
                    for i in range(n):
                        c = next(color)
                        if tools.check_if_not_nan_or_zero(second_fit_periods[i]):
                            plt.vlines(1. / second_fit_periods[i], 0, y_lim_max,
                                       label=second_fit_names[i] + " period " + str(
                                           round(second_fit_periods[i], 2)) + " days", color=c,
                                       alpha=alpha_periodogram_vlines)

                prepare_plot(f"Lomb-Scargle Periodogram {self.light_curve_name} {self.band_name}", 'Frequency [1/days]',
                             'Lomb-Scargle Power', self.show_image, self.save_image,
                             image_output_png=image_output_png, image_output_pdf=image_output_pdf,
                             save_pdf=self.save_pdfs, xlim=(np.min(self.ls_freqs) * 0.95, np.max(self.ls_freqs) * 1.05),
                             ylim=(0, y_lim_max), invert_yaxis=False)
            else:
                draw_empty_graph([0], label="Not enough points to make periodogram")
                prepare_plot(f"Lomb-Scargle Periodogram {self.light_curve_name} {self.band_name}", 'Frequency [1/days]',
                             'Lomb-Scargle Power', self.show_image, self.save_image,
                             image_output_png=image_output_png, image_output_pdf=image_output_pdf,
                             save_pdf=self.save_pdfs, invert_yaxis=False)

    def draw_folded_light_curve(self, image_output_png=None, image_output_pdf=None, draw_fit_function=True):
        """
        Draws the folded light curve using fitted period. If not fit is done, then tries to fit. If not successful,
        then draws original light curve without folding
        :param image_output_png: Output .png file location string
        :param image_output_pdf: Output .pdf file location string
        """
        if self.save_image or self.show_image:
            if self.fit_result is None:
                self.fit_light_curve(False)
            if self.fitting_successful:
                draw_points_err_bar(self.data_t, self.data_y, self.data_error, label="Data",
                                    folding_period=self.period_fit)

                if draw_fit_function:
                    plot_function(self.data_t, self.fit_result["fitfunc"], "Fit curve", folding_period=self.period_fit,
                                  color='r')

                prepare_plot(f"Folded {self.light_curve_name} {self.band_name}",
                             f"Phase with period {round(self.period_fit, 2)} days",
                             f"{self.light_curve_name} {self.band_name} band [mag]", self.show_image, self.save_image,
                             image_output_png=image_output_png, image_output_pdf=image_output_pdf,
                             save_pdf=self.save_pdfs)
            else:
                self.draw_raw_light_curve(image_output_png=image_output_png, image_output_pdf=image_output_pdf)

    def draw_folded_light_curve_with_other_function(self, function_to_fit_dict: dict, name_of_different_func: str,
                                                    image_output_png=None, image_output_pdf=None, fit_phase=False) -> Tuple[dict, float, float, float]:
        """
        Draws fitted light curve using passed function's period and phase. Fits amplitude and offset itself. Can fit
        phase if argument is passed for that
        :param function_to_fit_dict: The dictionary with period and phase. Should contain keys "period" and "phase" with period/phase to fit.
        :param name_of_different_func: The name of the different function for label on the plot
        :param image_output_png: Output .png file location string
        :param image_output_pdf: Output .pdf file location string
        :param fit_phase: If True, then does NOT use the phase from the dictionary, but fits it as well
        :return The dictionary with the fitted function, square error of fit, amplitude of fit, nrmse of fit
        """
        different_func_folded_fit = None
        if self.data_length >= 3:
            try:
                different_func_folded_fit = fit_sin_with_const_function(self.data_t, self.data_y,
                                                                        function_to_fit_dict["period"],
                                                                        function_to_fit_dict["phase"],
                                                                        fit_phase=fit_phase)
                different_func_fit_success = True
            except:
                different_func_fit_success = False
        else:
            different_func_fit_success = False

        if self.save_image or self.show_image:
            if different_func_fit_success:
                if self.data_length > 100:
                    alpha = 0.5
                else:
                    alpha = 1
                draw_points_err_bar(self.data_t, self.data_y, self.data_error, label="Data", alpha=alpha,
                                    folding_period=different_func_folded_fit["period"])
                plot_function(self.data_t, different_func_folded_fit["fitfunc"], "Fit curve",
                              folding_period=different_func_folded_fit["period"], color='r')

                prepare_plot(f"Folded {self.light_curve_name} {self.band_name} using {name_of_different_func} period",
                             f"Phase with period {round(different_func_folded_fit['period'], 2)} days",
                             f"{self.light_curve_name} {self.band_name} band [mag]", self.show_image, self.save_image,
                             image_output_png=image_output_png, image_output_pdf=image_output_pdf,
                             save_pdf=self.save_pdfs)
            else:
                self.draw_raw_light_curve(image_output_png=image_output_png, image_output_pdf=image_output_pdf)

        if different_func_folded_fit is None:
            return different_func_folded_fit, cf.dflt_no_vle, cf.dflt_no_vle, cf.dflt_no_vle
        else:
            expected_fit = different_func_folded_fit["fitfunc"](self.data_t)  # expected fit values
            error_fit = np.sum(np.square(self.data_y - expected_fit))  # square error from fit
            amp_fit = np.abs(different_func_folded_fit["amp"])  # the amplitude of the fit
            nrmse_fit = np.sqrt(error_fit / self.data_length) / amp_fit  # normalised root mean square error from fit
            return different_func_folded_fit, error_fit, amp_fit, nrmse_fit

    def remove_long_term_trend(self):
        """
        Removes the long term linear trend from the light curve
        """
        if self.data_length < 2:
            return

        from tools import linear_best_fit

        a, b = linear_best_fit(self.data_t, self.data_y)

        print(f'Linear trend {self.light_curve_name} {self.band_name} :y = {a} + {b}x')

        yfit = [b * xi for xi in self.data_t]

        self.data_y = self.data_y - yfit

    def fit_two_periods_new_version(self):
        """
        Fits two periods using following method: peak of LS of light curve followed by peak of LS of residuals
        """
        if self.data_length < cf.minimum_data_points_to_fit_light_curve:
            return
        time, data = self.data_t, self.data_y

        res, ls_freqs, ls_power, ls_guess_freq = non_linear_fit_sin(time, data, self.data_error, 0, type(self))

        if res["fit_success"]:
            data = data - res['fitfunc'](time)
            res_2, ls_freqs_2, ls_power_2, ls_guess_freq_2 = non_linear_fit_sin(time, data, self.data_error, 0,
                                                                                type(self))
            if res_2["fit_success"]:
                return self.fit_light_curve_using_two_periods(res["period"], res_2["period"])
        print("Could not fit double period")

    def draw_linear_trend_removal_example(self):
        """
        Draws original light curve with its linear trend. Draws the light curve with linear trend subtracted.
        """
        from tools import linear_best_fit

        a, b = linear_best_fit(self.data_t, self.data_y)

        fig, axs = plt.subplots(2)
        #gs = fig.add_gridspec(2, hspace=0)
        #axs = gs.subplots(sharex=True, sharey=True)
        fig.suptitle('Linear trend removal example')
        axs[0].errorbar(self.data_t, self.data_y, yerr=self.data_error, color='black', fmt='o', linewidth=2)

        func = lambda t: b * t + a
        time = np.linspace(np.min(self.data_t), np.max(self.data_t), 100)

        axs[0].plot(time, func(time), linewidth=2, color='black', label='Linear trend')

        self.remove_long_term_trend()

        axs[1].errorbar(self.data_t, self.data_y, yerr=self.data_error, color='black', fmt='o', linewidth=2)

        #plt.title(title)
        axs[0].legend(loc="best")
        plt.xlabel("Time [days]")
        #axs[0].ylabel(f"{self.light_curve_name} {self.band_name} band [mag]")
        #axs[1].ylabel(f"{self.light_curve_name} {self.band_name} band [mag]")
        fig.text(0.03, 0.5, f"{self.light_curve_name} {self.band_name} band [mag]", ha='center', va='center', rotation='vertical')

        axs[0].invert_yaxis()
        axs[1].invert_yaxis()

        #for ax in axs:
        #     ax.label_outer()

        plt.show()
        plt.close('all')

    def fitting_example_frequency_periodogram(self):
        """
        Draws an example of periodogram. Top: spectral window. Middle: LS of light curve. Bottom: LS of residulas.
        Normalises LS to be in units of magnitude.
        """
        fig, axs = plt.subplots(3)
        # gs = fig.add_gridspec(2, hspace=0)
        # axs = gs.subplots(sharex=True, sharey=True)
        #fig.suptitle('Linear trend removal example')

        spectral_win_y = np.ones(np.size(self.data_t))

        _, powers = calculate_periodogram_powers(self.data_t, spectral_win_y, ls_freqs_input, False, False, "psd")

        axs[0].plot(ls_freqs_input, 2 * np.sqrt(powers / self.data_length), linewidth=2, color='black', label='Spectral window')

        ls, powers_2 = calculate_periodogram_powers(self.data_t, self.data_y, ls_freqs_input, normalization="psd")
        """print(np.max(powers_2))
        print(ls.false_alarm_probability(np.max(powers_2))) #1.640405669083044e-107
        fap_level = ls.false_alarm_level([0.01], method='bootstrap')
        print(fap_level)
        fap_level = 2 * np.sqrt(fap_level / self.data_length)"""

        axs[1].plot(self.ls_freqs, 2 * np.sqrt(powers_2 / self.data_length), linewidth=2, color='black', label='Periodogram')
        #axs[1].plot(np.array([np.min(self.ls_freqs), np.max(self.ls_freqs)]), np.array([fap_level, fap_level]), linewidth=2, color='red', label='FAP at 0.01')

        data_y_residuals = self.data_y - self.fit_result['fitfunc'](self.data_t)

        _, powers_res = calculate_periodogram_powers(self.data_t, data_y_residuals, ls_freqs_input, normalization="psd")

        axs[2].plot(self.ls_freqs, 2 * np.sqrt(powers_res / self.data_length), linewidth=2, color='black', label='Residuals periodogram')

        # plt.title(title)
        axs[0].legend(loc='upper right')
        axs[1].legend(loc='upper right')
        axs[2].legend(loc='upper right')
        for ax in axs:
            ax.set_xlim(np.min(self.ls_freqs), np.max(self.ls_freqs))
            ax.set_ylim(0, None)
        plt.xlabel("Frequency [1/days]")
        # axs[0].ylabel(f"{self.light_curve_name} {self.band_name} band [mag]")
        # axs[1].ylabel(f"{self.light_curve_name} {self.band_name} band [mag]")
        fig.text(0.03, 0.5, f"Amplitude [mag]", ha='center', va='center',
                 rotation='vertical')

        plt.show()
        plt.close('all')

    def draw_spectral_window(self):
        """
        Draws spectral window. Normalises LS to be in units of magnitude.
        """
        spectral_win_y = np.ones(np.size(self.data_t))

        _, powers = calculate_periodogram_powers(self.data_t, spectral_win_y, ls_freqs_input, False, False, "psd")

        plt.plot(ls_freqs_input, 2 * np.sqrt(powers / self.data_length), linewidth=2, color='black',
                 label='Spectral window')

        prepare_plot(f'Spectral window {round(1 / np.max(self.ls_freqs), 1)} to {round(1 / np.min(self.ls_freqs), 0)} days', "Frequency [1/days]", "Amplitude [mag]", True, False,
                     invert_yaxis=False, xlim=(np.min(self.ls_freqs), np.max(self.ls_freqs)), ylim=(0, None))

    def choose_spectral_window(self, min_time: float, max_time: float):
        if max_time < min_time:
            min_time, max_time = max_time, min_time

        args_to_keep = np.logical_and.reduce((self.data_t >= min_time, self.data_t <= max_time))
        self.data_t = self.data_t[args_to_keep]
        self.data_y = self.data_y[args_to_keep]
        self.data_error = self.data_error[args_to_keep]
        self.data_length = np.size(self.data_t)

    def draw_folded_with_colored_time(self):
        x = (self.data_t % abs(self.period_fit)) / abs(self.period_fit)
        #plt.errorbar(x, y, yerr=error, fmt=fmt, label=label, linewidth=linewidth, color=color, alpha=alpha)

        cm = plt.cm.get_cmap('inferno')

        z = self.data_t
        sc = plt.scatter(x, self.data_y, c=z, cmap=cm, linewidth=1)
        plt.colorbar(sc, label='Time [days]')
        prepare_plot(f"Folded {self.light_curve_name} {self.band_name}", f"Phase with period {round(self.period_fit, 3)} d", "Band [mag]", True, False, invert_yaxis=True, show_legend=False)


class LightCurveGaia(LightCurve):
    def __init__(self, parsed_light_curve_votable, band: str, light_curve_name="Gaia", save_image=False,
                 save_pdfs=False, show_image=True):
        band = band.upper()
        if band not in ["G", "BP", "RP"]:
            raise ValueError("Unexpected Gaia band")
        super().__init__(band, light_curve_name, save_image, save_pdfs, show_image)

        clean_data, dirty_data = self.convert_gaia_votable_to_numpy_data(parsed_light_curve_votable, band)

        self.data_t, self.data_y, self.data_error, self.data_length = self.convert_gaia_data_to_three_arrays(clean_data)
        self.data_rej_t, self.data_rej_y, self.data_rej_error, self.data_rej_length = self.convert_gaia_data_to_three_arrays(
            dirty_data)

        self.calculate_light_curve_properties()
        self.nrmse_using_gaia_g = None

    @staticmethod
    def convert_gaia_data_to_three_arrays(data: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
        """
        Takes 2D array where each row has 3 elements and splits into 3 arrays. Checks if array is empty/1D and deals with it

        :param data: 2D/1D or empty array with data, where each row has 3 elements
        :return: Three 1D arrays with each one corresponding to 1 column and one float. If empty array is given, returns 3 empty lists. The float is length of the arrays
        """
        if data.ndim == 2:
            tt = data[:, 0]
            yy = data[:, 1]
            error = data[:, 2]
        elif np.size(data) == 0:
            tt = []
            yy = []
            error = []
        elif np.size(data) == 3:
            tt = data[0]
            yy = data[1]
            error = data[2]
        else:
            raise ValueError("Unexpected argument")
        return tt, yy, error, np.size(tt)

    @staticmethod
    def convert_gaia_votable_to_numpy_data(votable, band_to_get: str) -> Tuple[np.ndarray, np.ndarray]:
        """
        Takes a votable file and returns data (split into clean/rejected data) for all Gaia arrays

        :param votable: Parsed votable file location for Gaia light curve
        :param band_to_get: Band G, BP or RP to get from Gaia data
        :return: Returns clean and rejected data as arrays for all three bands as 6 arrays in order: G, BP, RP
        """
        if votable is None:
            return np.asarray([]), np.asarray([])

        # ids = votable.array["source_id"].astype(str)
        band = votable.array["band"]
        time = votable.array["time"].astype(float)
        mag = votable.array["mag"].astype(float)
        flux = votable.array["flux"].astype(float)
        flux_err = votable.array["flux_error"].astype(float)
        # flux_over_err = votable.array["flux_over_error"].astype(float)
        # error = 1 / votable.array["flux_over_error"].astype(float)
        # rej_by_phot = votable.array["rejected_by_photometry"]
        rej_by_var = votable.array["rejected_by_variability"]

        time = time - np.min(time)  # to start at time = 0

        clean_data, dirty_data = separate_gaia_light_curve(band, band_to_get.upper(), flux, flux_err, mag, rej_by_var,
                                                           time)

        return clean_data, dirty_data


class LightCurveZTF(LightCurve):
    def __init__(self, parsed_light_curve_votable, band: str, light_curve_name="ZTF", save_image=False, save_pdfs=False,
                 show_image=True):
        band = band.lower()
        if band not in ["g", "r", "i"]:
            raise ValueError("Unexpected ZTF band")
        super().__init__(band, light_curve_name, save_image, save_pdfs, show_image)

        clean_data = self.convert_ztf_votable_to_numpy_data(parsed_light_curve_votable, band)

        self.data_t, self.data_y, self.data_error, self.data_length = self.convert_ztf_data_to_three_arrays(clean_data)
        self.data_rej_t, self.data_rej_y, self.data_rej_error, self.data_rej_length = [], [], [], 0

        self.calculate_light_curve_properties()

        self.nrmse_using_gaia_g = None
        self.nrmse_using_other_ztf = None

        # if type(self) is LightCurveZTF:
        if len(cf.remove_ztf_peak_periodogram_days) > 0:
            for period_to_remove in cf.remove_ztf_peak_periodogram_days:
                for i in range(1):
                    self.fit_light_curve(manual_period_guess=period_to_remove)
                    if self.fitting_successful:
                        # print(self.period_fit)
                        if period_to_remove * 0.975 < self.period_fit < period_to_remove * 1.025:
                            self.data_y = self.data_y - self.fit_result["fitfunc"](
                                self.data_t)  # + self.fit_result["offset"]
                        else:
                            break

                # self.fit_light_curve_with_const_period(period_to_remove)
                # if self.fitting_successful_constant_period:
                #    self.data_y = self.data_y - self.fit_result_constant_period["fitfunc"](self.data_t) + self.fit_result_constant_period["offset"]

    @staticmethod
    def convert_ztf_data_to_three_arrays(data: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
        """
        Takes 2D array where each row has 3 elements and splits into 3 arrays. Checks if array is empty/1D and deals with it

        :param data: 2D/1D or empty array with data, where each row has 3 elements
        :return: Three 1D arrays with each one corresponding to 1 column and one float. If empty array is given, returns 3 empty lists. The float is length of the arrays
        """
        if np.size(data) == 0:
            tt = []
            yy = []
            error = []
        else:
            tt = np.asarray(data[0])
            yy = data[1]
            error = data[2]
        return tt, yy, error, np.size(tt)

    @staticmethod
    def convert_ztf_votable_to_numpy_data(votable, band_to_get) -> np.ndarray:
        """
        Takes a votable file and returns data for ZTF data

        :param votable: parsed votable file for ZTF light curve
        :param band_to_get: Band g, r or i to get from ZTF data
        :return: 3 2D arrays with time, magnitude and error for g, r and i ZTF bands
        """
        if votable is None:
            return np.asarray([])

        oids = votable.array["oid"].astype(str)
        # ra = votable.array["ra"].astype(float)
        # dec = votable.array["dec"].astype(float)

        band = votable.array["filtercode"]
        mjd_time = votable.array["mjd"].astype(float)
        mag = votable.array["mag"].astype(float)
        mag_err = votable.array["magerr"].astype(float)
        # catflags = votable.array["catflags"].astype(int)

        unique_oids, counts_oids = np.unique(oids, return_counts=True)

        if np.size(unique_oids) == 0:
            return np.array([])

        # oid_to_use = unique_oids[np.argmax(counts_oids)]
        # indices_to_use = np.where(oids == oid_to_use)[0]

        def get_ztf_specific_band(band_names: np.ndarray, mjd_times: np.ndarray, mags: np.ndarray, mag_errs: np.ndarray,
                                  band_name: str) -> np.ndarray:
            """
            Given all bands data, finds only the data for the required band_name and returns it. If empty, returns empty array

            :param band_names: Data with all band names
            :param mjd_times: Data with all time values
            :param mags: Data with all magnitudes
            :param mag_errs: Data with all errors
            :param band_name: Band name required to be extracted
            :return: The band_time, magnitude, error for required band_name extracted as 1 array. If no such data, returns empty array
            """
            band_index = np.where(band_names == band_name)[0]
            if np.size(band_index) != 0:
                band_time = mjd_times[band_index] - np.min(mjd_times)  # To make time start at 0
                band_mag = mags[band_index]
                band_magerr = mag_errs[band_index]
                # print(np.size(band_mag))
                if remove_ztf_measurements_if_too_close:
                    band_time, band_mag, band_magerr = remove_close_ztf_measurements(band_time, band_mag, band_magerr)
                band_time, band_mag, band_magerr = bin_nights_with_lots_of_measurements(band_time, band_mag,
                                                                                        band_magerr)

                # print(np.size(band_mag))

                return np.array([band_time, band_mag, band_magerr])
            else:
                return np.array([])

        # band_data = get_ztf_specific_band(band[indices_to_use], mjd_time[indices_to_use], mag[indices_to_use], mag_err[indices_to_use], "z" + band_to_get.lower())
        band_data = get_ztf_specific_band(band, mjd_time, mag, mag_err, "z" + band_to_get.lower())

        return band_data


class LightCurveNeowise(LightCurve):
    def __init__(self, parsed_light_curve_votable, source_id, band: str, light_curve_name="NEOWISE", save_image=False,
                 save_pdfs=False, show_image=True):
        band = band.lower()
        if band not in ["w1", "w2"]:
            raise ValueError("Unexpected NEOWISE band")
        super().__init__(band, light_curve_name, save_image, save_pdfs, show_image)

        self.data_t, self.data_y, self.data_error, self.data_length = self.convert_neowise_votable_to_numpy_data(
            parsed_light_curve_votable, source_id, band)

        self.data_rej_t, self.data_rej_y, self.data_rej_error, self.data_rej_length = [], [], [], 0

        self.calculate_light_curve_properties()

        self.nrmse_using_gaia_g = None
        self.nrmse_using_other_ztf = None

    @staticmethod
    def convert_neowise_votable_to_numpy_data(votable, source_id, band_to_get) -> Tuple[
        np.ndarray, np.ndarray, np.ndarray, float]:
        """
        Takes a votable file and returns data for ZTF data

        :param votable: parsed votable file for ZTF light curve
        :param band_to_get: Band g, r or i to get from ZTF data
        :return: 3 2D arrays with time, magnitude and error for g, r and i ZTF bands
        """
        if votable is None:
            return np.asarray([]), np.asarray([]), np.asarray([]), 0

        source_ids = votable.array["source_id"].astype(str)

        indices_to_use = np.where(source_ids == source_id)[0]

        if np.size(indices_to_use) == 0:
            return np.asarray([]), np.asarray([]), np.asarray([]), 0

        # ra = votable.array["ra"].astype(float)
        # dec = votable.array["dec"].astype(float)

        flags = votable.array["ph_qual"].astype(str)

        if band_to_get == "w1":
            mag = votable.array["w1mpro"].astype(float)
            mag_err = votable.array["w1sigmpro"].astype(float)
        else:
            mag = votable.array["w2mpro"].astype(float)
            mag_err = votable.array["w2sigmpro"].astype(float)
        mjd_time = votable.array["mjd"].astype(float)

        mag = np.asarray(mag[indices_to_use])
        mag_err = np.asarray(mag_err[indices_to_use])
        mjd_time = np.asarray(mjd_time[indices_to_use])
        flags = np.asarray(flags[indices_to_use])

        flags = flags.view('U1').reshape(np.size(flags), -1)[:, 0:2]
        if band_to_get == "w1":
            flags = flags[:, 0]
        else:
            flags = flags[:, 1]

        indices_to_use = np.invert(np.logical_or.reduce((flags == "X", flags == "U")))
        mag_good = mag[indices_to_use]
        mag_err_good = mag_err[indices_to_use]
        mjd_time_good = mjd_time[indices_to_use]

        return mjd_time_good, mag_good, mag_err_good, np.size(mjd_time_good)


def remove_close_ztf_measurements(time: np.ndarray, mag: np.ndarray, err: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Removes any ZTF measurements that are too close. For example, if limit is 1 hour and measurements are taken at
    [0, 0.5, 1, 1.5, 1.7, 2.1], then what will remain is measurements at [0, 1, 2.1]
    :param time: Array with times of measurements
    :param mag: Corresponding magnitude
    :param err: Corresponding error in mag
    :return: Time, mag, error arrays
    """
    sorted_ind = np.argsort(time)   # sorts time
    time = time[sorted_ind]
    mag = mag[sorted_ind]
    err = err[sorted_ind]

    diff_time = np.where(np.ediff1d(time) < min_time_between_measurements)[0]
    if np.size(diff_time > 0):
        start_index = diff_time[0]
        # time, mag, err = recursive_remove_close_measurements(time, mag, err, start_index)
        time, mag, err = remove_close_measurements_v2(time, mag, err, start_index)

    return time, mag, err


def remove_close_measurements_v2(time: np.ndarray, mag: np.ndarray, err: np.ndarray, start_index: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    The function that does actual removal of close measurements
    :param time: Array with times of measurements
    :param mag: Corresponding magnitude
    :param err: Corresponding error in mag
    :param start_index: At what index to start removal (i.e. assumes that indices before this one are not too close)
    :return: Time, mag, error arrays
    """
    ind_to_delete = np.array([]).astype(int)  # indices to delete from original array

    while True:
        if start_index + 1 < np.size(time):  # make sure that start index not out of range
            time_diff_to_1st = time[start_index + 1:] - time[start_index]
            # find difference between time measurements only for latter part of the array, the one not checked yet
            min_time_indices = np.where(time_diff_to_1st < min_time_between_measurements)[0] + start_index
            # find points which are very close in time to a data point at start_index

            if np.size(min_time_indices) > 0:  # if such points exists
                ind_to_delete = np.append(ind_to_delete, min_time_indices + 1)  # then will need to delete them

                diff_time = np.where(np.ediff1d(time[min_time_indices[-1] + 2:]) < min_time_between_measurements)[0]
                # now skip points that are going to be deleted. Are there any other points that are too close to each other
                # + 2 comes from two following reasons: 1) because time differences is off by 1. 2) because we want the next index after deleted indices
                if np.size(diff_time > 0):  # if yes
                    start_index = diff_time[0] + min_time_indices[
                        -1] + 2  # then continue deleting this points after last deleted point
                else:
                    break
            else:
                break

    time, mag, err = np.delete(time, ind_to_delete), np.delete(mag, ind_to_delete), np.delete(err, ind_to_delete)
    # delete all points at once at the end
    return time, mag, err


def recursive_remove_close_measurements(time, mag, err, start_index):
    # OLD to be deleted
    if start_index + 1 < np.size(time):
        time_diff_to_1st = time[start_index + 1:] - time[start_index]
        min_time_indices = np.where(time_diff_to_1st < min_time_between_measurements)[0] + start_index

        if np.size(min_time_indices) > 0:
            time, mag, err = np.delete(time, min_time_indices + 1), np.delete(mag, min_time_indices + 1), np.delete(err,
                                                                                                                    min_time_indices + 1)

        diff_time = np.where(np.ediff1d(time) < min_time_between_measurements)[0]
        if np.size(diff_time > 0):
            start_index = diff_time[0]
            time, mag, err = recursive_remove_close_measurements(time, mag, err, start_index)

    return time, mag, err


def bin_nights_with_lots_of_measurements(time: np.ndarray, mag: np.ndarray, err: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Bins together all measurements into a mean of a single night if it has more than a set amount of measurements
    :param time: Array with times of measurements
    :param mag: Corresponding magnitude
    :param err: Corresponding error in mag
    :return: Time, mag, error arrays
    """
    time_int = np.copy(time).astype(int)
    unique, counts = np.unique(time_int, return_counts=True)
    where_too_many_light_curve_measurements = np.where(counts >= max_measurements_per_night)[0]
    if np.size(where_too_many_light_curve_measurements) > 0:
        args_to_dlt = np.array([]).astype(int)
        for index in where_too_many_light_curve_measurements:
            args_to_bin = np.where(time_int == unique[index])[0]

            mean_time = np.mean(time[args_to_bin])
            mean_mag = np.mean(mag[args_to_bin])
            mean_err = np.mean(err[args_to_bin])

            time[args_to_bin[0]], mag[args_to_bin[0]], err[args_to_bin[0]] = mean_time, mean_mag, mean_err

            args_to_dlt = np.append(args_to_dlt, args_to_bin[1:-1])

        time = np.delete(time, args_to_dlt)
        mag = np.delete(mag, args_to_dlt)
        err = np.delete(err, args_to_dlt)

    return time, mag, err


def separate_gaia_light_curve(band: np.ndarray, band_used: str, flux: np.ndarray,
                              flux_err: np.ndarray, mag: np.ndarray, rej_by_var: np.ndarray,
                              time: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Separates data into "clean" and "bad" data

    :param band: 1D array with bands used
    :param band_used: The band that will be taken from the data
    :param flux: 1D array with fluxes
    :param flux_err: 1D array with flux errors
    :param mag: 1D array with magnitude values
    :param rej_by_var: 1D array with True/False depending whether it is rejected or not (True: data not taken as clean
    data no matter what)
    :param time: 1D array with times of corresponding values
    :return: Two 2D array with the following property: Each row represents [time mag error_in_mag]. First 2D array: clean data from corresponding band that was not rejected. Second 2D array: data from corresponding band that was rejected by rej_by_var array
    """

    def calc_mag_error(gaia_flux: float, gaia_error: float) -> float:
        """
        Calculates an approximate error of the Gaia band, given its magnitude and flux with error

        :param gaia_flux: flux corresponding to that magnitude
        :param gaia_error: float error of corresponding flux
        :return: float the error in the magnitude in units of mag
        """

        return abs(-2.5 / np.log(10) * gaia_error / gaia_flux)

    clean_data2 = []  # time mag error
    dirty_data = []
    for i in range(0, np.size(band)):
        if band[i] == band_used and not rej_by_var[i]:  # and rej_by_phot[i] == "false"
            clean_data2.append([time[i], mag[i], calc_mag_error(flux[i], flux_err[i])])
        elif band[i] == band_used and rej_by_var[i]:
            dirty_data.append([time[i], mag[i], calc_mag_error(flux[i], flux_err[i])])
    clean_data = np.asarray(clean_data2)
    dirty_data = np.asarray(dirty_data)
    return clean_data, dirty_data


def fit_sin_with_const_function(time: np.ndarray, y_values: np.ndarray, period: float, phase: float, fit_phase=False) -> dict:
    """
    Fits sin given different parameters and return the dictionary with fits. Period is constant and is not optimized
    :param time: 1D array with time/x coordinates
    :param y_values: 1D array with corresponding y coordinates
    :param period: Uses that period as a constant for fitting. I.e. period cannot vary during optimization
    :param phase: Phase to fit
    :param fit_phase: If True, then ignores given phase and fits it by itself
    :return: Returns fitting parameters as dictionary with keys: "amp", "omega", "phase", "offset", "freq", "period", "fitfunc", "maxcov", "rawres"
    """
    time = np.array(time)
    y_values = np.array(y_values)

    ang_freq = 2. * np.pi / period

    guess_amp = np.std(y_values) * 2. ** 0.5
    guess_offset = np.mean(y_values)

    if fit_phase:
        guess = np.array([guess_amp, 0, guess_offset])
    else:
        guess = np.array([guess_amp, guess_offset])

    def sinfunc_new_phase(t, A, new_phase, c):
        return A * np.sin(ang_freq * t + new_phase) + c

    def sinfunc(t, A, c):
        return A * np.sin(ang_freq * t + phase) + c

    if fit_phase:
        popt, pcov = scipy.optimize.curve_fit(sinfunc_new_phase, time, y_values, method='lm', p0=guess)
        A, phase, c = popt
    else:
        popt, pcov = scipy.optimize.curve_fit(sinfunc, time, y_values, method='lm', p0=guess)
        A, c = popt

    fitfunc = lambda t: A * np.sin(ang_freq * t + phase) + c
    return {"amp": A, "omega": ang_freq, "phase": phase, "offset": c, "freq": 1 / period, "period": period,
            "fitfunc": fitfunc, "maxcov": np.max(pcov), "rawres": (guess, popt, pcov)}


def fit_sin_with_const_period(time: np.ndarray, y_values: np.ndarray, error: np.ndarray, period: float) -> dict:
    """
    Fits sin given different parameters and return the dictionary with fits. Period is constant and is not optimized

    :param time: 1D array with time/x coordinates
    :param y_values: 1D array with corresponding y coordinates
    :param error: 1D array with corresponding errors (or zeroes if none)
    :param period: Uses that period as a constant for fitting. I.e. period cannot vary during optimization
    :return: Returns fitting parameters as dictionary with keys: "amp", "omega", "phase", "offset", "freq", "period", "fitfunc", "maxcov", "rawres"
    """
    time = np.array(time)
    y_values = np.array(y_values)

    ang_freq = 2. * np.pi / period

    guess_amp = np.std(y_values) * 2. ** 0.5
    guess_offset = np.mean(y_values)
    guess = np.array([guess_amp, 0., guess_offset])

    def sinfunc(t, A, p, c):
        return A * np.sin(ang_freq * t + p) + c

    try:
        popt, pcov = scipy.optimize.curve_fit(sinfunc, time, y_values, method='lm', p0=guess)
        A, p, c = popt
        fitfunc = lambda t: A * np.sin(ang_freq * t + p) + c
        fit_success = True
    except:
        A, w, p, c, f, fit_values_cov, fit_values, function_fit, period_fit = 0, 0, 0, 0, 0, 0, 0, 0, 0
        fitfunc = None
        pcov = np.array([0])
        popt = 0
        fit_success = False

    return {"amp": A, "omega": ang_freq, "phase": p, "offset": c, "freq": 1 / period, "period": period,
            "fitfunc": fitfunc, "maxcov": np.max(pcov), "rawres": (guess, popt, pcov), 'fit_success': fit_success}


def non_linear_fit_sin(time: np.ndarray, y_values: np.ndarray, error: np.ndarray, manual_period_guess: float,
                       lightcurve_type, fit_mean=True, center_data=True, normalization='standard', method='cython') \
        -> Tuple[dict, np.ndarray, np.ndarray, float]:
    """
    Fits sin given different parameters and return the dictionary with fits

    :param time: Time/x coordinates
    :param y_values: Corresponding y coordinates
    :param error: Corresponding errors (or zeroes if none)
    :param manual_period_guess: If 0 is given, then the guess will be done based on Lomb-Scargle method, otherwise manual guess is used
    :param lightcurve_type: The type of light curve class (e.g. LightCurveZTF, LightCurveGaia)
    :param fit_mean: Fit mean for astropy LS parameter
    :param center_data: Center data for astropy LS parameter
    :param normalization: Normalisation for astropy LS, 'standard', 'psd'
    :param method: Method for the LS calculation. Default 'cython'.
    :return: Returns fitting parameters as dictionary with keys: "amp", "omega", "phase", "offset", "freq", "period", "fitfunc", "maxcov", "rawres", "fit_success", 2 arrays from LS algorithm with powers/frequencies and guess frequency
    """

    time, y_values = np.asarray(time), np.asarray(y_values)

    ls, power = calculate_periodogram_powers(time, y_values, ls_freqs_input, fit_mean=fit_mean, center_data=center_data,
                                             normalization=normalization, method=method)

    if remove_1_day_peak_manually_ztf:
        if lightcurve_type is LightCurveZTF:
            indices_with_period_1_day = np.where(np.logical_and(1 / 1.025 < ls_freqs_input, ls_freqs_input < 1 / 0.975))
            power[indices_with_period_1_day] = 0

    ls_power_arg_max = np.argmax(power)
    guess_freq = ls_freqs_input[ls_power_arg_max]
    if center_data and ls is not None:
        ls_fap = ls.false_alarm_probability(power[ls_power_arg_max])
    else:
        ls_fap = cf.dflt_no_vle

    if manual_period_guess != 0:  # Override if manual
        guess_freq = 1 / manual_period_guess

    guess_amp = np.std(y_values) * 2. ** 0.5  # Guess for the scipy
    guess_offset = np.mean(y_values)
    guess = np.array([guess_amp, 2. * np.pi * guess_freq, 0., guess_offset])

    def sin_func(t, a, w, p, c):  # Function plot
        return a * np.sin(w * t + p) + c

    try:  # Try to optimize
        fit_values, fit_values_cov = scipy.optimize.curve_fit(sin_func, time, y_values, method='lm', p0=guess)
        amp, omega, phase, const = fit_values
        f = omega / (2. * np.pi)
        period_fit = abs(1. / f)
        function_fit: Callable[[List[float]], List[float]] = lambda t: amp * np.sin(omega * t + phase) + const
        fit_success = True
    except:  # If optimization took too long, abort and go here
        amp, omega, phase, const, f, fit_values_cov, fit_values, function_fit, period_fit = 0, 0, 0, 0, 0, 0, 0, 0, 0
        fit_success = False

    return {"amp": amp, "omega": omega, "phase": phase, "offset": const, "freq": f, "period": period_fit,
            "fitfunc": function_fit, "maxcov": np.max(fit_values_cov), "rawres": (guess, fit_values, fit_values_cov),
            "fit_success": fit_success, "ls_fap": ls_fap}, ls_freqs_input, power, guess_freq


def draw_ztf_and_gaia_light_curve_graphs(star_obj, show_image: bool, save_image: bool):
    """
    Plot the fits and folded fits for ZTF data together with normal Gaia data

    :param star_obj: Star class object
    :param show_image: If want to show images
    :param save_image: If want to save images
    :return: nothing
    """

    if show_image or save_image:
        gaia_g_light_curve, ztf_g_light_curve, ztf_r_light_curve, ztf_i_light_curve = star_obj.gaia_g_light_curve, star_obj.ztf_g_light_curve, star_obj.ztf_r_light_curve, star_obj.ztf_i_light_curve

        alpha_of_points = 0.3  # For plotting, what is transparency of points

        time_gaia_g, y_gaia_g, error_gaia_g, points_amount_gaia_g = gaia_g_light_curve.get_values()

        time_ztf_g, y_ztf_g, error_ztf_g, points_amount_ztf_g = ztf_g_light_curve.get_values()
        time_ztf_r, y_ztf_r, error_ztf_r, points_amount_ztf_r = ztf_r_light_curve.get_values()
        time_ztf_i, y_ztf_i, error_ztf_i, points_amount_ztf_i = ztf_i_light_curve.get_values()

        gaia_g_y_min, gaia_g_y_max = 1000, 0  # Temp variables, to know the min/max magnitude of Gaia g-band
        ztf_r_shift, ztf_g_shift = 0, 0  # Temp variables, about how much to shift g/r curves up/down

        # Draw G light curve with fit
        if points_amount_gaia_g > 0:  # Plot G points
            gaia_g_y_min = np.min(y_gaia_g)
            gaia_g_y_max = np.max(y_gaia_g)
            draw_points_err_bar(time_gaia_g, y_gaia_g, error_gaia_g, linewidth=2)
            plot_function(time_gaia_g, gaia_g_light_curve.fit_result["fitfunc"],
                          f"Gaia G: {round(gaia_g_light_curve.period_fit, 2)} d", alpha=0.5, color='black',
                          draw_function=gaia_g_light_curve.fitting_successful)

        # Draw ZTF on top
        if points_amount_ztf_g > 0:
            if np.min(y_ztf_g) < gaia_g_y_max:
                ztf_g_shift = abs(np.min(y_ztf_g) - gaia_g_y_max) + 0.5
            draw_points_err_bar(time_ztf_g, y_ztf_g + ztf_g_shift, error_ztf_g, color="green", alpha=alpha_of_points)

            if ztf_g_shift > 0:
                label_ztf_g = f"ZTF g: {round(ztf_g_light_curve.period_fit, 2)} d\nwith shift {round(ztf_g_shift, 1)}"
            else:
                label_ztf_g = f"ZTF g: {round(ztf_g_light_curve.period_fit, 2)} d"

            plot_function(time_ztf_g, ztf_g_light_curve.fit_result["fitfunc"],
                          label_ztf_g,
                          shift=ztf_g_shift, color="green", alpha=0.5,
                          draw_function=ztf_g_light_curve.fitting_successful)

        if points_amount_ztf_r > 0:
            if np.max(y_ztf_r) > gaia_g_y_min:
                ztf_r_shift = abs(np.max(y_ztf_r) - gaia_g_y_min) + 0.5
            draw_points_err_bar(time_ztf_r, y_ztf_r - ztf_r_shift, error_ztf_r, color="red", alpha=alpha_of_points)

            if ztf_r_shift > 0:
                label_ztf_r = f"ZTF r: {round(ztf_r_light_curve.period_fit, 2)} d\nwith shift -{round(ztf_r_shift, 1)}"
            else:
                label_ztf_r = f"ZTF r: {round(ztf_r_light_curve.period_fit, 2)} d"

            plot_function(time_ztf_r, ztf_r_light_curve.fit_result["fitfunc"],
                          label_ztf_r,
                          shift=-ztf_r_shift, color="red", alpha=0.5,
                          draw_function=ztf_r_light_curve.fitting_successful)

        draw_points_err_bar(time_ztf_i, y_ztf_i, error_ztf_i, color="purple", alpha=alpha_of_points)
        plot_function(time_ztf_i, ztf_i_light_curve.fit_result["fitfunc"],
                      f"ZTF i: {round(ztf_i_light_curve.period_fit, 2)} d", color="purple", alpha=0.5,
                      draw_function=ztf_i_light_curve.fitting_successful)

        prepare_plot("Gaia and ZTF", "Offset time [days]", "Offset magnitudes [mag]", show_image, save_image,
                     image_output_png=star_obj.ztf_output_pictures_raw)

        # Draw folded
        draw_ztf_and_gaia_light_curve_graphs_folded(star_obj, show_image, save_image)


def draw_ztf_and_gaia_light_curve_graphs_folded(star_obj, show_image: bool, save_image: bool, plot_fit=True):
    """
    Plot the fits and folded fits for ZTF data together with normal Gaia data

    :param star_obj: Star class object
    :param show_image: If want to show images
    :param save_image: If want to save images
    :param plot_fit: Whether to draw fits in folded curves
    :return: nothing
    """

    if show_image or save_image:
        gaia_g_light_curve, ztf_g_light_curve, ztf_r_light_curve, ztf_i_light_curve = star_obj.gaia_g_light_curve, star_obj.ztf_g_light_curve, star_obj.ztf_r_light_curve, star_obj.ztf_i_light_curve

        alpha_of_points = 0.3  # For plotting, what is transparency of points

        time_gaia_g, y_gaia_g, error_gaia_g, points_amount_gaia_g = gaia_g_light_curve.get_values()

        time_ztf_g, y_ztf_g, error_ztf_g, points_amount_ztf_g = ztf_g_light_curve.get_values()
        time_ztf_r, y_ztf_r, error_ztf_r, points_amount_ztf_r = ztf_r_light_curve.get_values()
        time_ztf_i, y_ztf_i, error_ztf_i, points_amount_ztf_i = ztf_i_light_curve.get_values()

        gaia_g_y_min, gaia_g_y_max = 1000, 0  # Temp variables, to know the min/max magnitude of Gaia g-band
        ztf_r_shift, ztf_g_shift = 0, 0  # Temp variables, about how much to shift g/r curves up/down
        gaia_g_point_label, ztf_g_point_label, ztf_r_point_label, ztf_i_point_label = None, None, None, None

        if points_amount_gaia_g > 0:
            gaia_g_y_min = np.min(y_gaia_g)
            gaia_g_y_max = np.max(y_gaia_g)

        # Draw folded period for G
        if gaia_g_light_curve.fitting_successful:
            
            if not plot_fit:
                gaia_g_point_label = f"Gaia G"

            draw_points_err_bar(time_gaia_g, y_gaia_g, error_gaia_g, folding_period=gaia_g_light_curve.period_fit, label=gaia_g_point_label)

            if plot_fit:
                plot_function(time_gaia_g, gaia_g_light_curve.fit_result["fitfunc"],
                              f"Gaia G: {round(gaia_g_light_curve.period_fit, 2)} d", alpha=0.5, color="black",
                              folding_period=gaia_g_light_curve.period_fit)

        # Draw folded ZTF on top
        if ztf_g_light_curve.fitting_successful:
            if np.min(y_ztf_g) < gaia_g_y_max:
                ztf_g_shift = abs(np.min(y_ztf_g) - gaia_g_y_max) + 0.5

            if ztf_g_shift > 0:
                label_ztf_g = f"ZTF g: {round(ztf_g_light_curve.period_fit, 2)} d\nwith shift {round(ztf_g_shift, 1)}"
                if not plot_fit:
                    ztf_g_point_label = f"ZTF g with shift {round(ztf_g_shift, 1)}"
            else:
                label_ztf_g = f"ZTF g: {round(ztf_g_light_curve.period_fit, 2)} d"
                if not plot_fit:
                    ztf_g_point_label = "ZTF g"



            draw_points_err_bar(time_ztf_g, y_ztf_g + ztf_g_shift, error_ztf_g, color="green", alpha=alpha_of_points,
                                folding_period=ztf_g_light_curve.period_fit, label=ztf_g_point_label)

            if plot_fit:
                plot_function(time_ztf_g, ztf_g_light_curve.fit_result["fitfunc"],
                              label_ztf_g,
                              folding_period=ztf_g_light_curve.period_fit, shift=ztf_g_shift, color="green", alpha=0.5)

        if ztf_r_light_curve.fitting_successful:
            if np.max(y_ztf_r) > gaia_g_y_min:
                ztf_r_shift = abs(np.max(y_ztf_r) - gaia_g_y_min) + 0.5

            if ztf_r_shift > 0:
                label_ztf_r = f"ZTF r: {round(ztf_r_light_curve.period_fit, 2)} d\nwith shift -{round(ztf_r_shift, 1)}"
                if not plot_fit:
                    ztf_r_point_label = f"ZTF r with shift -{round(ztf_r_shift, 1)}"
            else:
                label_ztf_r = f"ZTF r: {round(ztf_r_light_curve.period_fit, 2)} d"
                if not plot_fit:
                    ztf_r_point_label = "ZTF r"

            draw_points_err_bar(time_ztf_r, y_ztf_r - ztf_r_shift, error_ztf_r, color="red", alpha=alpha_of_points,
                                folding_period=ztf_r_light_curve.period_fit, label=ztf_r_point_label)

            if plot_fit:
                plot_function(time_ztf_r, ztf_r_light_curve.fit_result["fitfunc"],
                              label_ztf_r,
                              folding_period=ztf_r_light_curve.period_fit, shift=-ztf_r_shift, color="red", alpha=0.5)

        if ztf_i_light_curve.fitting_successful:
            if not plot_fit:
                ztf_i_point_label = "ZTF i"

            draw_points_err_bar(time_ztf_i, y_ztf_i, error_ztf_i, color="purple", alpha=alpha_of_points,
                                folding_period=ztf_i_light_curve.period_fit, label=ztf_i_point_label)

            if plot_fit:
                plot_function(time_ztf_i, ztf_i_light_curve.fit_result["fitfunc"],
                              f"ZTF i: {round(ztf_i_light_curve.period_fit, 2)} d",
                              folding_period=ztf_i_light_curve.period_fit, color="purple", alpha=0.5)

        if not plot_fit:
            title = f"Gaia and ZTF, folded with period {round(gaia_g_light_curve.period_fit, 4)} d"
        else:
            title = "Gaia and ZTF, folded"

        prepare_plot(title, "Phase", "Offset magnitudes [mag]", show_image, save_image,
                     image_output_png=star_obj.ztf_output_pictures_folded)


def find_correlation_of_time_series(light_curve_1: LightCurve, light_curve_2: LightCurve) -> float:
    """
    Find correlation value between two light curves. Only takes light curves that have same amount of points and have
    measurements taken within 1 minute. Assumes that measurements are taken at the same time (first light curve time)
    :param light_curve_1: First light curve
    :param light_curve_2: Second light curve
    :return: Correlation coefficient between two time light curves. If light curves too different, returns -9999
    """
    if light_curve_1.data_length == light_curve_2.data_length:
        time_difference = np.abs(light_curve_1.data_t - light_curve_2.data_t)
        if np.max(time_difference) < 1 / 24 / 60:  # Time difference of measurements less than 1 minute
            series_1 = pandas.Series(data=light_curve_1.data_y, index=light_curve_1.data_t)
            series_2 = pandas.Series(data=light_curve_2.data_y, index=light_curve_1.data_t)
            correlation_value = series_1.corr(series_2)
            return correlation_value
    return cf.dflt_no_vle


def plot_three_gaia_curves(star_obj, image_output_png=None):
    """
    Draws a big plot of light curve with all 3 Gaia bands
    :param star_obj: Star object
    :param image_output_png: Where to save .png of the fit
    """
    if star_obj.gaia_g_light_curve.save_image or star_obj.gaia_g_light_curve.show_image:
        plt.figure(figsize=(8, 10), dpi=80)

        draw_points_err_bar(star_obj.gaia_g_light_curve.data_t, star_obj.gaia_g_light_curve.data_y,
                            star_obj.gaia_g_light_curve.data_error, label="Gaia G")
        draw_points_err_bar(star_obj.gaia_g_light_curve.data_rej_t, star_obj.gaia_g_light_curve.data_rej_y,
                            star_obj.gaia_g_light_curve.data_rej_error, fmt='x', color="green")
        draw_points_err_bar(star_obj.gaia_bp_light_curve.data_t, star_obj.gaia_bp_light_curve.data_y,
                            star_obj.gaia_bp_light_curve.data_error, label="Gaia BP", color="blue")
        draw_points_err_bar(star_obj.gaia_bp_light_curve.data_rej_t, star_obj.gaia_bp_light_curve.data_rej_y,
                            star_obj.gaia_bp_light_curve.data_rej_error, fmt='x', color="green")
        draw_points_err_bar(star_obj.gaia_rp_light_curve.data_t, star_obj.gaia_rp_light_curve.data_y,
                            star_obj.gaia_rp_light_curve.data_error, label="Gaia RP", color="red")
        draw_points_err_bar(star_obj.gaia_rp_light_curve.data_rej_t, star_obj.gaia_rp_light_curve.data_rej_y,
                            star_obj.gaia_rp_light_curve.data_rej_error, fmt='x', color="green")

        draw_empty_graph([star_obj.gaia_g_light_curve.data_length, star_obj.gaia_g_light_curve.data_rej_length,
                          star_obj.gaia_bp_light_curve.data_length, star_obj.gaia_bp_light_curve.data_rej_length,
                          star_obj.gaia_rp_light_curve.data_length, star_obj.gaia_rp_light_curve.data_rej_length])

        prepare_plot("All Gaia plots", "Time [days]", "Gaia band [mag]", star_obj.gaia_g_light_curve.show_image,
                     star_obj.gaia_g_light_curve.save_image, image_output_png=image_output_png)


def multi_band_periodogram(light_curves: List[LightCurve], light_curve_filters: List[str], save_image: bool,
                           show_image: bool, save_pdfs=False, image_output_png=None, image_output_pdf=None) -> Tuple[float, float, float, float, float]:
    """
    Creates multi band periodogram based on light curves passed in a list
    :param light_curves: List with light curves classes
    :param light_curve_filters: List with the names of the light curves filters
    :param save_image: Whether to save image
    :param show_image: Whether to show image
    :param save_pdfs: Whether to save image as pdf
    :param image_output_png: Output .png file location string
    :param image_output_pdf: Output .pdf file location string
    :return: Period of 1st peak, peak value of 1st peak, period of 2nd peak, peak value of 2nd peak, median of original LS
    """
    model = LombScargleMultiband()

    minimum_light_curve_length = 10

    t = []
    y = []
    dy = []
    filters = []

    filter_all_names = ""
    for j in range(len(light_curve_filters)):
        filter_all_names = filter_all_names + light_curve_filters[j] + " "

    for i in range(len(light_curves)):
        if light_curves[i].data_length > minimum_light_curve_length:
            t = t + list(light_curves[i].data_t)
            y = y + list(light_curves[i].data_y)
            dy = dy + list(light_curves[i].data_error)
            filters = filters + [light_curve_filters[i]] * light_curves[i].data_length

    if len(t) > minimum_light_curve_length:

        model.fit(np.asarray(t), np.asarray(y), dy=None, filts=np.asarray(filters))

        ls_freqs = np.arange(ls_freq_start, ls_freq_stop + ls_freq_time_step, ls_freq_time_step)
        ls_periods = 1 / ls_freqs

        ls_power = model.periodogram(ls_periods)

        indices_with_period_1_day = np.where(np.logical_and(1 / 1.025 < ls_freqs, ls_freqs < 1 / 0.975))
        ls_power[indices_with_period_1_day] = 0

        ls_guess_freq = ls_freqs[np.argmax(np.asarray(ls_power))]
        # plt.plot(1 / periods, ls_power)
        # plt.xscale('log')
        # plt.show()

        if save_image or show_image:
            y_lim_max = np.max(ls_power)

            # plt.semilogx(self.ls_freqs, self.ls_power, label="Frequencies power", color="green")
            plt.plot(ls_freqs, ls_power, label="Frequencies power", color="green")

            plt.vlines(ls_guess_freq, 0, y_lim_max, label="Guess period " + str(round(1. / ls_guess_freq, 2)) + " days",
                       color="black", alpha=alpha_periodogram_vlines)

            """def one_over(x):
                x = np.array(x).astype(float)
                near_zero = np.isclose(x, 0)
                x[near_zero] = np.inf
                x[~near_zero] = 1 / x[~near_zero]
                return x
    
            # the function "1/x" is its own inverse
            inverse = one_over
    
            secax = ax.secondary_xaxis('top', functions=(one_over, inverse))
            secax.set_xticks([10, 15, 20, 30, 40, 50, 100, 500])
            secax.set_xlabel('Period [days]')"""

            prepare_plot(f"LS Multi-band Frequency Periodogram with {filter_all_names}", 'Frequency [1/days]',
                         'Lomb-Scargle Power', show_image, save_image, image_output_png=image_output_png,
                         image_output_pdf=image_output_pdf, save_pdf=save_pdfs, invert_yaxis=False,
                         xlim=(np.min(ls_freqs) * 0.95, np.max(ls_freqs) * 1.05), ylim=(0, y_lim_max))

        freq1, peak1_value, freq2, peak2_value = find_two_peaks_periodogram(ls_freqs, ls_power)
        return 1. / freq1, peak1_value, 1. / freq2, peak2_value, np.median(ls_power)
    else:
        draw_empty_graph([0], label="Not enough points to make periodogram")
        prepare_plot(f"LS Multi-band Frequency Periodogram with {filter_all_names}", 'Frequency [1/days]',
                     'Lomb-Scargle Power', show_image, save_image, image_output_png=image_output_png,
                     image_output_pdf=image_output_pdf, save_pdf=save_pdfs, invert_yaxis=False)
        return cf.dflt_no_vle, cf.dflt_no_vle, cf.dflt_no_vle, cf.dflt_no_vle, cf.dflt_no_vle


def find_two_peaks_periodogram(freqs: np.ndarray, power: np.ndarray) -> Tuple[float, float, float, float]:
    """
    Find locations of two peaks with their frequency position. Does not remove peaks by fits, simply finds peaks in OG LS
    :param freqs: The frequencies of the LS
    :param power: The powers of the LS
    :return: Frequency of 1st peak, peak value of 1st peak, frequency of 2nd peak, peak value of 2nd peak
    """
    frequencies, powers = find_n_peaks_periodogram(freqs, power, 2)

    guess_freq, guess_freq_2 = frequencies[0], frequencies[1]
    pow_1, pow_2 = powers[0], powers[1]

    return guess_freq, pow_1, guess_freq_2, pow_2


def calculate_periodogram_powers(time: np.ndarray, mag: np.ndarray, freqs: np.ndarray, fit_mean=True, center_data=True,
                                 normalization='standard', method='cython', error=None) -> Tuple[astropyLS, np.ndarray]:
    """
    Calculates the powers in the LS
    :param time: Time array
    :param mag: Magnitudes array
    :param freqs: Array with frequencies where to calculate LS
    :param fit_mean: Whether to fit mean
    :param center_data: Whether to center data
    :param normalization: Normalisation to use for astorpy LS. Usually 'standard' or 'psd'
    :param method: Method to use. 'scipy' means to use Scipy packacge, otherwise 'cython'/'slow'/'auto' etc uses astropy
    :param error: Error of the magnitudes if want to take errors into account. Usually not expected
    :return: astropy LS class and computed powers. Use LS class for e.g. FAP calculation
    """
    if method == "scipy":
        ls_power = scipy_lombscargle(time, mag, freqs * 2. * np.pi, precenter=center_data)
        return None, ls_power
    else:
        ls = astropyLS(time, mag, dy=error, fit_mean=fit_mean, center_data=center_data, normalization=normalization)
        power = ls.power(freqs, method=method)
        return ls, power


def slider_folded_light_curve(lightcurve: LightCurve, valmin: float, valmax: float):
    """
    Slider for folded light curve to try to guess period by eye
    :param lightcurve: Light curve class
    :param valmin: Minimum period value for the slider
    :param valmax: Maximum period value for the slider
    """

    # Define initial parameters
    init_period = (valmax + valmin) / 2

    # Create the figure and the line that we will manipulate
    fig, ax = plt.subplots()
    line, = plt.plot((np.asarray(lightcurve.data_t) % init_period) / init_period, lightcurve.data_y, 'o')
    ax.set_xlabel('Time [s]')

    # adjust the main plot to make room for the sliders
    plt.subplots_adjust(left=0.25, bottom=0.25)
    plt.gca().invert_yaxis()
    ax.set_title("Type number to set minimum slider value")

    # Make a horizontal slider to control the period.
    axfreq = plt.axes([0.25, 0.1, 0.65, 0.03])
    period_slider = Slider(
        ax=axfreq,
        label='Period [days]',
        valmin=valmin,
        valmax=valmax,
        valinit=init_period,
    )

    # The function to be called anytime a slider's value changes
    def update(val):
        line.set_xdata((lightcurve.data_t % period_slider.val) / period_slider.val)
        fig.canvas.draw_idle()

    def update_slider(evt):
        print(evt.key)
        try:
            val = int(evt.key)
            period_slider.valmin = val - 0.5
            period_slider.valmin = val + 0.5
            fig.canvas.draw_idle()
        except:
            pass

    # register the update function with each slider
    period_slider.on_changed(update)
    fig.canvas.mpl_connect('key_press_event', update_slider)

    # Create a `matplotlib.widgets.Button` to reset the sliders to initial values.
    resetax = plt.axes([0.8, 0.025, 0.1, 0.04])
    button = Button(resetax, 'Reset', hovercolor='0.975')

    def reset(event):
        period_slider.reset()

    button.on_clicked(reset)

    plt.show()
    plt.close('all')


def slider_folded_light_curve_freq(lightcurve: LightCurve, valmin, valmax):
    """
    Slider for folded light curve to try to guess frequency by eye
    :param lightcurve: Light curve class
    :param valmin: Minimum period value for the slider
    :param valmax: Maximum period value for the slider
    """

    # Define initial parameters
    init_period = (valmax + valmin) / 2

    # Create the figure and the line that we will manipulate
    fig, ax = plt.subplots()
    line, = plt.plot((lightcurve.data_t % (1 / init_period)) / (1 / init_period), lightcurve.data_y, 'o')
    ax.set_xlabel('Time [s]')

    # adjust the main plot to make room for the sliders
    plt.subplots_adjust(left=0.25, bottom=0.25)
    plt.gca().invert_yaxis()
    ax.set_title("Type number to set minimum slider value")

    # Make a horizontal slider to control the period.
    axfreq = plt.axes([0.25, 0.1, 0.65, 0.03])
    period_slider = Slider(
        ax=axfreq,
        label='Period [days]',
        valmin=valmin,
        valmax=valmax,
        valinit=init_period,
    )

    # The function to be called anytime a slider's value changes
    def update(val):
        line.set_xdata((lightcurve.data_t % (1 / period_slider.val)) / (1 / period_slider.val))
        fig.canvas.draw_idle()

    def update_slider(evt):
        print(evt.key)
        try:
            val = int(evt.key)
            period_slider.valmin = val / 1.1
            period_slider.valmin = val * 1.1
            fig.canvas.draw_idle()
        except:
            pass

    # register the update function with each slider
    period_slider.on_changed(update)
    fig.canvas.mpl_connect('key_press_event', update_slider)

    # Create a `matplotlib.widgets.Button` to reset the sliders to initial values.
    resetax = plt.axes([0.8, 0.025, 0.1, 0.04])
    button = Button(resetax, 'Reset', hovercolor='0.975')

    def reset(event):
        period_slider.reset()

    button.on_clicked(reset)

    plt.show()


def find_n_peaks_periodogram(freqs: np.ndarray, power_og: np.ndarray, n_of_peaks: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Finds peaks locations and powers in the periodogram (simply finds peaks, does not do fits or anything)
    :param freqs: Frequencies where to compute LS
    :param power_og: Calculated powers of the LS
    :param n_of_peaks: how many peaks to find
    :return: Locations of peaks, powers of the peaks
    """
    peaks_freqs = np.zeros(n_of_peaks)
    peaks_powers = np.zeros(n_of_peaks)

    power = np.copy(power_og)

    argmax_power = np.argmax(power)

    peaks_freqs[0] = freqs[argmax_power]
    peaks_powers[0] = power_og[argmax_power]

    for i in range(1, n_of_peaks):
        j = argmax_power
        while True:
            previous_power = power[j]
            power[j] = 0
            if j + 1 >= np.size(power):
                break
            if previous_power > power[j + 1]:
                j = j + 1
            else:
                break

        j = argmax_power - 1
        while True:
            previous_power = power[j]
            power[j] = 0
            if j - 1 < 0:
                break
            if previous_power > power[j - 1]:
                j = j - 1
            else:
                break

        argmax_power = np.argmax(power)
        peaks_freqs[i] = freqs[argmax_power]
        peaks_powers[i] = power_og[argmax_power]

    return peaks_freqs, peaks_powers


def draw_points_err_bar(x: np.ndarray, y: np.ndarray, error: np.ndarray, label=None, fmt='o', color='black',
                        linewidth=2.0, alpha=1.0, folding_period=None):
    """
    Draws the points with error bars on plt. Can draw folded light curve if folding period is passed and non-zero
    :param x: x of points
    :param y: y of points
    :param error: errors of points
    :param label: label fot points, default none
    :param fmt: symbol for points, default circle
    :param color: colour to use, default black
    :param linewidth: linewidth of points, default 2
    :param alpha: alpha of points (0 to 1), default 1
    :param folding_period: period to fold, if need to fold points
    """
    if np.size(x) > 0:  # Check if any points exist at all
        if folding_period is not None and folding_period != 0:
            x = (x % abs(folding_period)) / abs(folding_period)
        plt.errorbar(x, y, yerr=error, fmt=fmt, label=label, linewidth=linewidth, color=color, alpha=alpha)


def draw_empty_graph(number_of_points_list: List[int], label="No points in this band"):
    """
    Draws empty graph if list is full of 0-s, which indicates the number of points in already drawn graphs
    :param number_of_points_list: List of number of points in each light curve
    :param label: Label to use if there are no points in the plot
    """
    number_of_points_list = np.asarray(number_of_points_list)
    number_of_points = np.sum(number_of_points_list)
    if number_of_points == 0:
        plt.errorbar([0], [0], yerr=[0], fmt='o', label=label, linewidth=2)


def plot_function(time: np.ndarray, function: Callable[[List[float]], List[float]], label: str, shift=0.0,
                  folding_period=None, color='black', linewidth=2, alpha=1.0, function_amount_of_points=5000,
                  draw_function=True):
    """
    Plots a function on the plt. Can shift function or fold it into 1 period, then x is phase 0 to 1
    :param time: Time is where the points are drawn, so the function will be drawn at these times too
    :param function: Function to plot
    :param label: Label name of the function
    :param shift: If non-zero, then shifts function up/down
    :param folding_period: Period to fold, if need to fold
    :param color: colour to use, default black
    :param linewidth: linewidth of points, default 2
    :param alpha: alpha of points (0 to 1), default 1
    :param function_amount_of_points: How many points to draw in a function, default 5000
    :param draw_function: Whethet to darw function in the first place (useful if want to pass bool to not draw function sometimes)
    """
    if draw_function:
        if folding_period is not None and folding_period != 0:
            folding_period = abs(folding_period)
            time = time % folding_period
            x_func = np.linspace(np.min(time), np.max(time), function_amount_of_points)
            x = x_func / folding_period
        else:
            x_func = np.linspace(np.min(time), np.max(time), function_amount_of_points)
            x = x_func

        plt.plot(x, function(x_func) + shift, label=label, linewidth=linewidth, color=color, alpha=alpha)


def prepare_plot(title: str, xlabel: str, ylabel: str, show_image: bool, save_image: bool, image_output_png=None,
                 image_output_pdf=None, save_pdf=False, invert_yaxis=True, xlim=None, ylim=None, show_legend=True):
    """
    Prepares plot and shows it or saves it
    :param title: Title of the plot
    :param xlabel: x label
    :param ylabel: y label
    :param show_image: Whether to show image
    :param save_image: Whether to save image
    :param image_output_png: str where to save image in png
    :param image_output_pdf: str where to save image in pdf
    :param save_pdf: Whether to save pdf even if save image
    :param invert_yaxis: Whether to invert y-axis, default True
    :param xlim: Tuple of the x limits, default none
    :param ylim: Tuple of the y limits, default none
    :param show_legend: show legend or not, default True
    """
    plt.title(title)
    if show_legend:
        plt.legend(loc="best")
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.xlim(xlim)
    plt.ylim(ylim)
    if invert_yaxis:
        plt.gca().invert_yaxis()
    if save_image:
        if image_output_png is not None:
            plt.savefig(image_output_png)
        if save_pdf and image_output_pdf is not None:
            plt.savefig(image_output_pdf)
    if show_image:
        plt.show()
    plt.close('all')


def plot_g_r_variation(light_curve_g: LightCurveZTF, light_curve_r: LightCurveZTF):
    binned_time, mag_r, error_r, g_r, g_r_error = g_r_variation(light_curve_g, light_curve_r)

    draw_points_err_bar(binned_time, g_r, g_r_error, "g-r", linewidth=2, fmt='.')
    #draw_points_err_bar(light_curve_r.data_t, (light_curve_r.data_y - np.mean(light_curve_r.data_y)) / np.mean(light_curve_r.data_y) * -1 + np.mean(g_r) - 0.3, np.zeros(np.size(light_curve_r.data_y)), "Offset light curve", linewidth=0.01, color='red', alpha=0.1, fmt='.')
    prepare_plot(None, "Time [days]", "g-r [mag]", True, False, invert_yaxis=False, show_legend=False)

    a_ztf_g = 1.1837965102373564
    a_ztf_r = 0.8352016801352571

    # r = m * (g-r)
    # extinction_vector_slope = a_ztf_r / (a_ztf_g - a_ztf_r)

    #draw_points_err_bar(g_r, mag_r, np.zeros(np.size(binned_time)), None, linewidth=1, fmt='.')
    cm = plt.cm.get_cmap('brg')
    z = binned_time

    #cmap = cm.viridis
    norm = Normalize(vmin=z.min(), vmax=z.max())

    plt.errorbar(g_r, mag_r, xerr=g_r_error, yerr=error_r, fmt="", marker="", ls="", ecolor=cm(norm(z)))

    sc = plt.scatter(g_r, mag_r, c=z, cmap=cm, linewidth=1, zorder=100)
    #plt.errorbar(g_r, mag_r, xerr=g_r_error, yerr=error_r, zorder=0)#, fmt=None, marker=None, zorder=0)
    plt.colorbar(sc, label='Time [days]')



    #origin = np.array([np.min(g_r), np.max(mag_r)])
    #plt.arrow(np.min(g_r), np.min(mag_r), a_ztf_g - a_ztf_r, a_ztf_r, width=0.001, color="k",
    #          head_width=0.1, head_length=0.15, overhang=1)

    prop = dict(arrowstyle="->,head_width=0.3,head_length=0.5", shrinkA=0, shrinkB=0)

    x_offset = 0.5 * (np.max(g_r) - np.min(g_r))

    plt.annotate("", xy=(np.min(g_r) + x_offset + (a_ztf_g - a_ztf_r), np.min(mag_r) + a_ztf_r),
                 xytext=(np.min(g_r) + x_offset, np.min(mag_r)), arrowprops=prop)  # extinction arrow

    prepare_plot(None, "g-r [mag]", "r [mag]", True, False, invert_yaxis=True, show_legend=False)


def draw_g_r_raw_light_curve(light_curve_g: LightCurveZTF, light_curve_r: LightCurveZTF):
    draw_points_err_bar(light_curve_g.data_t, light_curve_g.data_y, light_curve_g.data_error, label="ZTF g band", color='green', alpha=0.6, linewidth=2, fmt=".")
    draw_points_err_bar(light_curve_r.data_t, light_curve_r.data_y, light_curve_r.data_error, label="ZTF r band", color='red', alpha=0.6, linewidth=2, fmt=".")
    prepare_plot(None, "Time [days]", "Band [mag]", True, False)


def g_r_variation(light_curve_g: LightCurveZTF, light_curve_r: LightCurveZTF):
    binned_time_g, binned_mag_g, binned_error_g = bin_nights(light_curve_g.data_t, light_curve_g.data_y, light_curve_g.data_error)
    binned_time_r, binned_mag_r, binned_error_r = bin_nights(light_curve_r.data_t, light_curve_r.data_y, light_curve_r.data_error)

    min_time_to_use = max(np.min(binned_time_g), np.min(binned_time_r))
    max_time_to_use = min(np.max(binned_time_g), np.max(binned_time_r))

    args_to_use_g = np.logical_and.reduce((binned_time_g <= max_time_to_use, min_time_to_use <= binned_time_g))
    args_to_use_r = np.logical_and.reduce((binned_time_r <= max_time_to_use, min_time_to_use <= binned_time_r))

    binned_mag_g = binned_mag_g[args_to_use_g]
    binned_error_g = binned_error_g[args_to_use_g]
    binned_mag_r = binned_mag_r[args_to_use_r]
    binned_error_r = binned_error_r[args_to_use_r]
    binned_time_g = binned_time_g[args_to_use_g]

    g_r = binned_mag_g - binned_mag_r
    g_r_error = binned_error_g + binned_error_r

    g_r_args_to_use = ~np.isnan(g_r)

    binned_time_g = binned_time_g[g_r_args_to_use]
    g_r = g_r[g_r_args_to_use]
    binned_mag_r = binned_mag_r[g_r_args_to_use]
    g_r_error = g_r_error[g_r_args_to_use]
    binned_error_r = binned_error_r[g_r_args_to_use]

    return binned_time_g, binned_mag_r, binned_error_r, g_r, g_r_error


def bin_nights(time, mag, error):
    time_int = time.astype(int)
    bins_amount = np.max(time_int) - np.min(time_int) + 1

    median_values, bin_edges, bin_numbers = binned_statistic(time_int, mag, statistic='median', bins=bins_amount,
                                                             range=(np.min(time_int), np.max(time_int) + 1))
    mean_error_values, _, __ = binned_statistic(time_int, error, statistic='mean', bins=bins_amount,
                                                             range=(np.min(time_int), np.max(time_int) + 1))

    if False:
        time_binned = bin_edges[:-1][~np.isnan(median_values)]
        mag_binned = median_values[~np.isnan(median_values)]
    else:
        time_binned = bin_edges[:-1]
        mag_binned = median_values
        error_binned = mean_error_values

    return time_binned, mag_binned, error_binned

