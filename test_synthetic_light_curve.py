import numpy as np
import tools
import lightcurve_class
from astropy.io.votable import parse_single_table
import get_gaia_table
import config_file as cf
from sine_curve_test import simulate_sine_curve_1
from sine_curve_2 import simulate_sine_curve_normal
import matplotlib.pyplot as plt
import scipy.fftpack
from scipy.fft import fft, fftfreq
from scipy.signal import lombscargle
import matplotlib

matplotlib.use('macosx')

min_day = 0.5
max_day = 1000

freqs_to_try = np.linspace(1 / max_day, 1 / min_day, 10000)
periods_to_try = 1 / freqs_to_try
#periods_to_try = np.linspace(0.5, 1000, 10000)

periods_to_try = np.append(np.linspace(min_day, max_day, 2500), 1 / freqs_to_try)

filename = "test_half_to_1k_days_with_half_to_1k_days_higher_noise_v6"
filename = "test_one_star_newest"

const_noise = False
draw_periodogram = True
#periods_to_try = [23.515]

noise = 1


def simulate_sine_curve_2(amp: float, period: float, tt: np.ndarray, offset: float, noise: float) -> np.ndarray:
    if np.size(tt) == 0:
        return []
    omega = 2 * np.pi / period
    phase = np.random.random() * np.pi
    yy = amp * np.sin(omega * tt + phase) + offset
    yynoise = yy + noise * (np.random.random(len(tt)) - 0.5)
    return yynoise


def test_many_synthetic_data(textfile_name):
    gaia_lpv_table = get_gaia_lpv_table.get_data_from_table(cf.input_gaia_table_path)  # Get main Gaia info (GDR2 + GEDR3)
    ids = tools.get_specific_column_from_data_str(gaia_lpv_table, "source_id").astype(str)

    path_gaia = cf.input_light_curve_to_analyse_path
    path_ztf = cf.input_ztf_lightcurves_to_analyse

    tools.save_in_txt_topcat(["# source_id\ttrue_period\tgaia_g_per\tztf_g_per\tztf_r_per"], "synthetic_data/" + textfile_name + ".txt")

    for source_id in ids:
        if source_id == "2207196101581030784":
        #if True:
            gaia_votable_data = parse_single_table(path_gaia + source_id + ".xml")
            ztf_votable_data = parse_single_table(path_ztf + source_id + ".vot")

            """ztf_g_time = tools.load_data("synthetic_data/ztf_g_synthetic_1836765033547214336.txt")[:, 0].astype(float)
            ztf_r_time = tools.load_data("synthetic_data/ztf_r_synthetic_1836765033547214336.txt")[:, 0].astype(float)
            gaia_g_time = tools.load_data("synthetic_data/gaia_g_synthetic_1836765033547214336.txt").astype(float)
    
            gaia_g_mean = 14.5
            gaia_g_amp = 0.3
    
            ztf_g_mean = 16.5
            ztf_g_amp = 0.45
    
            ztf_r_mean = 14.9
            ztf_r_amp = 0.35"""

            periods_to_try_2 = np.random.choice(periods_to_try, 10000)

            for true_period in periods_to_try_2:
                gaia_g_light_curve = lightcurve_class.LightCurveGaia(gaia_votable_data, "G", show_image=False)
                ztf_g_light_curve = lightcurve_class.LightCurveZTF(ztf_votable_data, "g", show_image=False)
                ztf_r_light_curve = lightcurve_class.LightCurveZTF(ztf_votable_data, "r", show_image=False)

                if const_noise:
                    gaia_g_light_curve.data_y = simulate_sine_curve_normal(1, true_period, gaia_g_light_curve.data_t, gaia_g_light_curve.mean, 1, noise)
                    gaia_g_light_curve.fit_light_curve()

                    ztf_g_light_curve.data_y = simulate_sine_curve_normal(1, true_period, ztf_g_light_curve.data_t, ztf_g_light_curve.mean, 1, noise)
                    ztf_g_light_curve.fit_light_curve()

                    ztf_r_light_curve.data_y = simulate_sine_curve_normal(1, true_period, ztf_r_light_curve.data_t, ztf_r_light_curve.mean, 1, noise)
                    ztf_r_light_curve.fit_light_curve()
                else:
                    sigma = 5

                    gaia_g_light_curve.data_y = simulate_sine_curve_normal(gaia_g_light_curve.percentile_q95_5 / 2, true_period,
                                                                           gaia_g_light_curve.data_t, gaia_g_light_curve.mean,
                                                                           1, gaia_g_light_curve.avg_error * sigma)
                    gaia_g_light_curve.fit_light_curve()

                    ztf_g_light_curve.data_y = simulate_sine_curve_normal(ztf_g_light_curve.percentile_q95_5 / 2, true_period,
                                                                          ztf_g_light_curve.data_t, ztf_g_light_curve.mean,
                                                                          1, ztf_g_light_curve.avg_error * sigma)
                    ztf_g_light_curve.fit_light_curve()

                    ztf_r_light_curve.data_y = simulate_sine_curve_normal(ztf_r_light_curve.percentile_q95_5 / 2, true_period,
                                                                          ztf_r_light_curve.data_t, ztf_r_light_curve.mean,
                                                                          1, ztf_r_light_curve.avg_error * sigma)
                    ztf_r_light_curve.fit_light_curve()

                #print(gaia_g_light_curve.percentile_q95_5 - gaia_g_light_curve.avg_error, ztf_g_light_curve.percentile_q95_5 - ztf_g_light_curve.avg_error,ztf_r_light_curve.percentile_q95_5 - ztf_r_light_curve.avg_error)
                print(true_period, gaia_g_light_curve.period_fit,ztf_g_light_curve.period_fit,ztf_r_light_curve.period_fit)
                tools.save_in_txt_topcat([source_id, true_period, gaia_g_light_curve.period_fit,ztf_g_light_curve.period_fit,ztf_r_light_curve.period_fit], "synthetic_data/" + textfile_name + ".txt")


def test_one_synthetic(textfile_name):
    ztf_g_time = tools.load_data("synthetic_data/ztf_g_synthetic_1836765033547214336.txt")[:, 0].astype(float)
    ztf_r_time = tools.load_data("synthetic_data/ztf_r_synthetic_1836765033547214336.txt")[:, 0].astype(float)
    gaia_g_time = tools.load_data("synthetic_data/gaia_g_synthetic_1836765033547214336.txt").astype(float)

    gaia_g_mean = 14.5
    gaia_g_amp = 0.3

    ztf_g_mean = 16.5
    ztf_g_amp = 0.45

    ztf_r_mean = 14.9
    ztf_r_amp = 0.35

    tools.save_in_txt_topcat(["# true_period\tgaia_g_per\tztf_g_per\tztf_r_per"], "synthetic_data/" + textfile_name + ".txt")

    for true_period in periods_to_try:
        gaia_g_light_curve = lightcurve_class.LightCurveGaia(None, "G", show_image=False)
        ztf_g_light_curve = lightcurve_class.LightCurveZTF(None, "g", show_image=False)
        ztf_r_light_curve = lightcurve_class.LightCurveZTF(None, "r", show_image=False)

        gaia_g_light_curve.data_t = gaia_g_time
        gaia_g_light_curve.data_y = simulate_sine_curve_1(gaia_g_amp, true_period, gaia_g_time, gaia_g_mean, noise)
        gaia_g_light_curve.data_length = np.size(gaia_g_time)
        gaia_g_light_curve.data_error = np.zeros(np.size(gaia_g_time))
        gaia_g_light_curve.fit_light_curve()

        ztf_g_light_curve.data_t = ztf_g_time
        ztf_g_light_curve.data_y = simulate_sine_curve_1(ztf_g_amp, true_period, ztf_g_time, ztf_g_mean, noise)
        ztf_g_light_curve.data_length = np.size(ztf_g_time)
        ztf_g_light_curve.data_error = np.zeros(np.size(ztf_g_time))
        ztf_g_light_curve.fit_light_curve()

        ztf_r_light_curve.data_t = ztf_r_time
        ztf_r_light_curve.data_y = simulate_sine_curve_1(ztf_r_amp, true_period, ztf_r_time, ztf_r_mean, noise)
        ztf_r_light_curve.data_length = np.size(ztf_r_time)
        ztf_r_light_curve.data_error = np.zeros(np.size(ztf_r_time))
        ztf_r_light_curve.fit_light_curve()

        print(true_period, gaia_g_light_curve.period_fit,ztf_g_light_curve.period_fit,ztf_r_light_curve.period_fit)
        if not draw_periodogram:
            tools.save_in_txt_topcat([true_period, gaia_g_light_curve.period_fit,ztf_g_light_curve.period_fit,ztf_r_light_curve.period_fit], "synthetic_data/" + textfile_name + ".txt")

        if draw_periodogram:
            gaia_g_light_curve.show_image = True
            gaia_g_light_curve.draw_frequency_gram()
            gaia_g_light_curve.draw_fitted_light_curve()

            ztf_g_light_curve.show_image = True
            ztf_g_light_curve.draw_frequency_gram()
            ztf_g_light_curve.draw_fitted_light_curve()

            ztf_r_light_curve.show_image = True
            ztf_r_light_curve.draw_frequency_gram()
            ztf_r_light_curve.draw_fitted_light_curve()


def spectral_window_many():
    gaia_lpv_table = get_gaia_lpv_table.get_data_from_table(
        cf.input_gaia_table_path)  # Get main Gaia info (GDR2 + GEDR3)
    ids = tools.get_specific_column_from_data_str(gaia_lpv_table, "source_id").astype(str)

    path_gaia = cf.input_light_curve_to_analyse_path
    path_ztf = cf.input_ztf_lightcurves_to_analyse

    freqs = np.arange(lightcurve_class.ls_freq_start, lightcurve_class.ls_freq_stop + lightcurve_class.ls_freq_time_step, lightcurve_class.ls_freq_time_step / 10)
    ang_freqs = freqs * 2 * np.pi

    for source_id in ids:
        gaia_votable_data = parse_single_table(path_gaia + source_id + ".xml")
        ztf_votable_data = parse_single_table(path_ztf + source_id + ".vot")

        gaia_g_light_curve = lightcurve_class.LightCurveGaia(gaia_votable_data, "G", show_image=False)
        ztf_g_light_curve = lightcurve_class.LightCurveZTF(ztf_votable_data, "g", show_image=False)
        ztf_r_light_curve = lightcurve_class.LightCurveZTF(ztf_votable_data, "r", show_image=False)

        if source_id == "4488894969319094912":
            gaia_g_light_curve.data_y = np.ones(np.size(gaia_g_light_curve.data_t))
            gaia_g_light_curve.show_image = True
            gaia_g_light_curve.draw_frequency_gram()
            gaia_g_light_curve.draw_periodogram()

        if ztf_g_light_curve.data_length < 1500 and ztf_g_light_curve.data_length > 10:
            print(source_id, ztf_g_light_curve.data_length)
            ztf_g_light_curve.data_y = np.ones(np.size(ztf_g_light_curve.data_t))
            ztf_g_light_curve.show_image = True
            ztf_g_light_curve.fit_light_curve(fit_mean=False, center_data=False, normalization='psd')
            ztf_g_light_curve.draw_frequency_gram()
            ztf_g_light_curve.draw_periodogram()
            try:
                #lightcurve_class.multi_band_periodogram([ztf_g_light_curve], ["ztf_g"], False, True)
                #ztf_g_light_curve.data_y = simulate_sine_curve_1(2, 23.5, ztf_g_light_curve.data_t, 15, 0)
                """ls_powers = lombscargle(ztf_g_light_curve.data_t, ztf_g_light_curve.data_y, ang_freqs, normalize=False, precenter=False)
                plt.plot(freqs, ls_powers)
                plt.title("ztf g freqs")
                plt.show()

                plt.semilogx(1/freqs, ls_powers)
                plt.title("ztf g period")
                plt.show()"""
                #ztf_g_light_curve.draw_frequency_gram()
                #ztf_g_light_curve.draw_periodogram()
                pass
            except:
                pass

        if ztf_r_light_curve.data_length > 10:
            ztf_r_light_curve.data_y = np.ones(np.size(ztf_r_light_curve.data_t))
            ztf_r_light_curve.show_image = True
            ztf_g_light_curve.fit_light_curve(fit_mean=False, center_data=False, normalization='psd')
            ztf_g_light_curve.draw_frequency_gram()
            ztf_g_light_curve.draw_periodogram()
            try:
                #lightcurve_class.multi_band_periodogram([ztf_r_light_curve], ["ztf_r"], False, True)
                """ls_powers = lombscargle(ztf_r_light_curve.data_t, ztf_r_light_curve.data_y, ang_freqs, normalize=False, precenter=False)
                plt.plot(freqs, ls_powers)
                plt.title("ztf r freqs")
                plt.show()

                plt.semilogx(1 / freqs, ls_powers)
                plt.title("ztf r period")
                plt.show()"""
                """ztf_r_light_curve.draw_frequency_gram()
                ztf_r_light_curve.draw_periodogram()"""
                pass
            except:
                pass


def spectral_window_many_noise():
    gaia_lpv_table = get_gaia_lpv_table.get_data_from_table(
        cf.input_gaia_table_path)  # Get main Gaia info (GDR2 + GEDR3)
    ids = tools.get_specific_column_from_data_str(gaia_lpv_table, "source_id").astype(str)

    path_gaia = cf.input_light_curve_to_analyse_path
    path_ztf = cf.input_ztf_lightcurves_to_analyse

    for source_id in ids:
        gaia_votable_data = parse_single_table(path_gaia + source_id + ".xml")
        ztf_votable_data = parse_single_table(path_ztf + source_id + ".vot")

        gaia_g_light_curve = lightcurve_class.LightCurveGaia(gaia_votable_data, "G", show_image=False)
        ztf_g_light_curve = lightcurve_class.LightCurveZTF(ztf_votable_data, "g", show_image=False)
        ztf_r_light_curve = lightcurve_class.LightCurveZTF(ztf_votable_data, "r", show_image=False)

        if source_id == "4488894969319094912":
            gaia_g_light_curve.data_y = np.ones(np.size(gaia_g_light_curve.data_t))
            gaia_g_light_curve.show_image = True
            gaia_g_light_curve.draw_frequency_gram()
            gaia_g_light_curve.draw_periodogram()

        if ztf_g_light_curve.data_length < 15000 and ztf_g_light_curve.data_length > 10:
            print(source_id, ztf_g_light_curve.data_length)
            ztf_g_light_curve.data_y = np.ones(np.size(ztf_g_light_curve.data_t)) + np.random.normal(scale=0.1, size=np.size(ztf_g_light_curve.data_t))
            ztf_g_light_curve.show_image = True
            ztf_g_light_curve.fit_light_curve(fit_mean=False, center_data=False, normalization='psd')
            ztf_g_light_curve.draw_frequency_gram()
            ztf_g_light_curve.draw_periodogram()

        if ztf_r_light_curve.data_length > 10:
            print(source_id, ztf_r_light_curve.data_length)
            ztf_r_light_curve.data_y = np.ones(np.size(ztf_r_light_curve.data_t)) + np.random.normal(scale=0.3, size=np.size(ztf_r_light_curve.data_t))
            ztf_r_light_curve.show_image = True
            ztf_r_light_curve.fit_light_curve(fit_mean=False, center_data=False, normalization='psd')
            ztf_r_light_curve.draw_frequency_gram()
            ztf_r_light_curve.draw_periodogram()


def spectral_window_check_fitted_curves():
    gaia_lpv_table = get_gaia_lpv_table.get_data_from_table(
        cf.input_gaia_table_path)  # Get main Gaia info (GDR2 + GEDR3)
    ids = tools.get_specific_column_from_data_str(gaia_lpv_table, "source_id").astype(str)

    path_gaia = cf.input_light_curve_to_analyse_path
    path_ztf = cf.input_ztf_lightcurves_to_analyse

    fitted_data = tools.load_data("output_textfiles/2022-03-11 11_43_47.057904_all_periods_final_v10_combined.txt")
    source_ids_to_check = fitted_data[:, 0].astype(str)
    fitted_period_gaia_g = fitted_data[:, 21].astype(float)
    fitted_period_ztf_g = fitted_data[:, 83].astype(float)
    fitted_period_ztf_r = fitted_data[:, 87].astype(float)

    for source_id in ids:
        if source_id in source_ids_to_check:
            index_of_data_in_fitted = np.where(source_id == source_ids_to_check)[0][0]

            print(source_id)
            gaia_votable_data = parse_single_table(path_gaia + source_id + ".xml")
            ztf_votable_data = parse_single_table(path_ztf + source_id + ".vot")

            gaia_g_light_curve = lightcurve_class.LightCurveGaia(gaia_votable_data, "G", show_image=False)
            ztf_g_light_curve = lightcurve_class.LightCurveZTF(ztf_votable_data, "g", show_image=False)
            ztf_r_light_curve = lightcurve_class.LightCurveZTF(ztf_votable_data, "r", show_image=False)

            #check_spectral_window_with_fitted_data_one(fitted_period_gaia_g, gaia_g_light_curve,
            #                                           index_of_data_in_fitted)
            check_spectral_window_with_fitted_data_one(fitted_period_ztf_g, ztf_g_light_curve,
                                                       index_of_data_in_fitted)
            check_spectral_window_with_fitted_data_one(fitted_period_ztf_r, ztf_r_light_curve,
                                                       index_of_data_in_fitted)



def check_spectral_window_with_fitted_data_one(fitted_periods, light_curve, index_of_data_in_fitted):
    light_curve.data_y = np.ones(np.size(light_curve.data_t))
    light_curve.show_image = True
    light_curve.fit_light_curve(fit_mean=False, center_data=False, normalization='psd')
    if light_curve.fitting_successful:
        spectral_freqs_peaks, _ = lightcurve_class.find_n_peaks_periodogram(light_curve.ls_freqs,
                                                                            light_curve.ls_power, 20)
        nearest_freq_peak = tools.find_nearest(spectral_freqs_peaks, 1 / fitted_periods[index_of_data_in_fitted])
        if tools.compare_two_floats(nearest_freq_peak, 1 / fitted_periods[index_of_data_in_fitted], 0.01):
            print(1 / nearest_freq_peak, nearest_freq_peak)
            print(1 / spectral_freqs_peaks)
            print(spectral_freqs_peaks)
            print(np.where(spectral_freqs_peaks == nearest_freq_peak)[0][0])
            print(fitted_periods[index_of_data_in_fitted], 1 / fitted_periods[index_of_data_in_fitted])
            light_curve.draw_frequency_gram()
            light_curve.draw_periodogram()
        else:
            print("all_good", light_curve.light_curve_name, light_curve.band_name)


def spectral_window_one():
    ztf_g_time = tools.load_data("synthetic_data/ztf_g_synthetic_1836765033547214336.txt")[:, 0].astype(float)
    ztf_r_time = tools.load_data("synthetic_data/ztf_r_synthetic_1836765033547214336.txt")[:, 0].astype(float)
    gaia_g_time = tools.load_data("synthetic_data/gaia_g_synthetic_1836765033547214336.txt").astype(float)

    gaia_g_mean = 14.5
    gaia_g_amp = 0.3

    ztf_g_mean = 16.5
    ztf_g_amp = 0.45

    ztf_r_mean = 14.9
    ztf_r_amp = 0.35

    gaia_g_light_curve = lightcurve_class.LightCurveGaia(None, "G", show_image=False)
    ztf_g_light_curve = lightcurve_class.LightCurveZTF(None, "g", show_image=False)
    ztf_r_light_curve = lightcurve_class.LightCurveZTF(None, "r", show_image=False)

    gaia_g_light_curve.data_t = gaia_g_time
    gaia_g_light_curve.data_y = simulate_sine_curve_1(1, 1, gaia_g_time, gaia_g_mean, 0)
    gaia_g_light_curve.data_y = np.ones(np.size(gaia_g_time))
    gaia_g_light_curve.data_length = np.size(gaia_g_time)
    gaia_g_light_curve.data_error = np.zeros(np.size(gaia_g_time))
    gaia_g_light_curve.fit_light_curve()

    gaia_g_light_curve.show_image = True
    gaia_g_light_curve.draw_periodogram()
    gaia_g_light_curve.draw_frequency_gram()

    ztf_g_light_curve.data_t = ztf_g_time
    ztf_g_light_curve.data_y = simulate_sine_curve_1(1, 1, ztf_g_time, ztf_g_mean, 0)
    ztf_g_light_curve.data_length = np.size(ztf_g_time)
    ztf_g_light_curve.data_y = np.ones(np.size(ztf_g_time))
    ztf_g_light_curve.data_error = np.zeros(np.size(ztf_g_time))
    ztf_g_light_curve.fit_light_curve()

    ztf_g_light_curve.show_image = True
    ztf_g_light_curve.draw_periodogram()
    ztf_g_light_curve.draw_frequency_gram()

    ztf_r_light_curve.data_t = ztf_r_time
    ztf_r_light_curve.data_y = simulate_sine_curve_1(1, 1, ztf_r_time, ztf_r_mean, 0)
    ztf_r_light_curve.data_y = np.ones(np.size(ztf_r_time))
    ztf_r_light_curve.data_length = np.size(ztf_r_time)
    ztf_r_light_curve.data_error = np.zeros(np.size(ztf_r_time))
    ztf_r_light_curve.fit_light_curve()

    ztf_r_light_curve.show_image = True
    ztf_r_light_curve.draw_periodogram()
    ztf_r_light_curve.draw_folded_light_curve()
    ztf_r_light_curve.draw_frequency_gram()
    ztf_r_light_curve.draw_raw_light_curve()

    N = 10000
    T = 100

    #yf = scipy.fftpack.fft(ztf_r_light_curve.data_t)
    #xf = np.linspace(0.0, N * T, N)
    timestep = 0.1
    #n = np.size(yf)
    #freq = np.fft.fftfreq(n, d=timestep)
    #xf = fftfreq(N, T)[:N // 2]

    #plt.subplot(2, 1, 1)
    #plt.plot(freq, yf)
    #plt.xlim(-0.1, 0.1)
    #plt.show()
    #plt.subplot(2, 1, 2)
    #plt.plot(xf[1:], 2.0 / N * np.abs(yf[0:N / 2])[1:])



def main():
    #test_one_synthetic(filename)
    #test_many_synthetic_data(filename)
    #spectral_window_one()
    #spectral_window_many()
    #spectral_window_check_fitted_curves()
    spectral_window_many_noise()


if __name__ == '__main__':
    main()
