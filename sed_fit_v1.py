from astropy import units as u
from sedfitter import (fit, plot, plot_params_1d, plot_params_2d,
                       write_parameters, write_parameter_ranges)
from sedfitter.extinction import Extinction
import numpy as np
import config_file as cf
from sedfitter.filter import Filter
from sedfitter.convolve import convolve_model_dir
import datetime
import tools


def bb_model_fit_with_image(sedfitter_file_data, filters_total, sedfitter_models_dir, parameters_output_path, min_dist, max_dist):
    fitinfo_file_output = 'sedfitter/fitinfo/' + str(datetime.datetime.now()).replace(':', '_') + '_.fitinfo'
    filters = filters_total

    # "GDR2_BP", "GEDR3_BP", "GEDR3_G", "GDR2_G", "GEDR3_RP", "GDR2_RP", "2J", "2H", "2K", "WISE1",
    #                            "I1", "I2", "WISE2", "I3", "I4", "WISE3", "WISE4", "M1", "PACS1", "M2", "PACS2"]

    apertures = [0.23, 0.23, 0.23, 0.23, 0.23, 0.23, 4, 4, 4, 8.25, 2.4, 2.4, 8.25, 2.4, 2.4, 8.25, 16.5, 3.5, 11.4, 16, 13.7] * u.arcsec
    apertures = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 4, 4, 4, 8.25, 2.4, 2.4, 8.25, 2.4, 2.4, 8.25, 16.5, 3.5, 11.4, 16, 13.7] * u.arcsec
    apertures = [1.0, 1.0, 1.0, 4, 4, 4, 8.25, 2.4, 2.4, 8.25, 2.4, 2.4, 8.25, 16.5, 3.5, 11.4, 16, 13.7] * u.arcsec
    #apertures = [1.0, 1.0, 1.0, 4, 4, 4, 8.25, 2.4] * u.arcsec

    if len(filters) != len(apertures):
        raise ValueError("Filters != apertures")

    # Read in extinction law. We read in columns 1 (the wavelength in microns) and
    # 4 (the opacity in cm^2/g)
    extinction = Extinction.from_file('kmh94.par', columns=[0, 3], wav_unit=u.micron, chi_unit=u.cm**2 / u.g)
    extinction = Extinction.from_file('whitney.txt', columns=[0, 3], wav_unit=u.micron, chi_unit=u.cm**2 / u.g)

    #dist1 = 1000 / (star_obj.parallax_ge3 + star_obj.parallax_error_ge3)
    #dist2 = 1000 / (star_obj.parallax_ge3 - star_obj.parallax_error_ge3)

    #min_dist = min(dist1, dist2) / 1000
    #max_dist = max(dist1, dist2) / 1000

    if min_dist == 0 and max_dist == 0:  # If none given, assume relatively closeby
        min_dist = 0.01
        max_dist = 1.5
    else:       # Give some extra space in case the parallax is off by up to 15%
        min_dist = min_dist / 1000 * (1 / 1.1)  # * 0.9
        if min_dist == 0:
            min_dist = 0.01
        max_dist = max_dist / 1000 * (1 / 0.9)  # * 1.1

    # Run the fitting
    fit(sedfitter_file_data, filters, apertures, sedfitter_models_dir, fitinfo_file_output,
        extinction_law=extinction, distance_range=[min_dist, max_dist] * u.kpc, av_range=[0., 5])

    # For the remaining commands, we always select the models with chi^2-chi_best^2
    # per datapoint less than 0.005.
    select_format = ('F', 0.07)

    # Make SED plots
    #plot(fitinfo_file_output, output_dir='sedfitter/pdf_out/' + str(datetime.datetime.now()).replace(':', '_').replace(' ', '_').replace('.', '_'), plot_max=100, plot_mode='A', show_sed=True)

    #write_parameters(fitinfo_file_output, cf.sedfitter_output_parameters_folder + str(min_dist) + 'parameters.txt') # old
    write_parameters(fitinfo_file_output, cf.sedfitter_output_parameters + 'parameters.txt')

    # Write out the min/max ranges corresponding to the above file
    write_parameter_ranges(fitinfo_file_output, parameters_output_path + 'parameters_ranges.txt', select_format=select_format)

    if sedfitter_models_dir == 'models_r06':
        # Make histograms of the disk mass
        plot_params_1d(fitinfo_file_output, 'MDISK', output_dir='sedfitter/pdf_out/' + str(datetime.datetime.now()).replace(':', '_').replace(' ', '_').replace('.', '_') + "plotsmdisk/", log_x=True, select_format=select_format)

        # Make 2-d plots of the envelope infall rate vs disk mass
        plot_params_2d(fitinfo_file_output, 'MDISK', 'MDOT', output_dir='sedfitter/pdf_out/' + str(datetime.datetime.now()).replace(':', '_').replace(' ', '_').replace('.', '_') + "plots_mdot_mdisk/", log_x=True, log_y=True, select_format=select_format)


def write_parameters_for_calculated_models(fitinfo_file_output, parameters_output_path):
    select_format = ('F', 0.07)
    #write_parameters(fitinfo_file_output, cf.sedfitter_output_parameters + 'parameters.txt')
    write_parameter_ranges(fitinfo_file_output, parameters_output_path + 'parameters_ranges.txt', select_format=select_format)


def create_wise_gaia_pacs_filters(model_dir):
    wlg, g, gbp, grp = np.loadtxt('response_functions/GaiaDR2_RevisedPassbands.dat', unpack=True, usecols=(0, 1, 3, 5))
    g = np.asarray(g)
    gbp = np.asarray(gbp)
    grp = np.asarray(grp)
    wlg_g = np.asarray(wlg)
    wlg_bp = np.copy(wlg_g)
    wlg_rp = np.copy(wlg_g)

    wlg_g = wlg_g[np.where(g < 90)]
    g = list((g[np.where(g < 90)])[::-1])

    wlg_bp = wlg_bp[np.where(gbp < 90)]
    gbp = list((gbp[np.where(gbp < 90)])[::-1])

    wlg_rp = wlg_rp[np.where(grp < 90)]
    grp = list((grp[np.where(grp < 90)])[::-1])

    f = Filter()
    f.name = "GDR2_G"
    f.central_wavelength = cf.g_dr2 * pow(10, 6) * u.micron
    f.nu = list(np.array((cf.light_speed / (wlg_g * 1e-9)))[::-1]) * u.Hz
    f.response = g
    f.normalize()

    f2 = Filter()
    f2.name = "GDR2_BP"
    f2.central_wavelength = cf.bp_dr2 * pow(10, 6) * u.micron
    f2.nu = list(np.array((cf.light_speed / (wlg_bp * 1e-9)))[::-1]) * u.Hz
    f2.response = gbp
    f2.normalize()

    f3 = Filter()
    f3.name = "GDR2_RP"
    f3.central_wavelength = cf.rp_dr2 * pow(10, 6) * u.micron
    f3.nu = list(np.array((cf.light_speed / (wlg_rp * 1e-9)))[::-1]) * u.Hz
    f3.response = grp
    f3.normalize()

    wlg, g, gbp, grp = np.loadtxt('response_functions/EDR3_passband.dat', unpack=True, usecols=(0, 1, 3, 5))
    g = np.asarray(g)
    gbp = np.asarray(gbp)
    grp = np.asarray(grp)
    wlg_g = np.asarray(wlg)
    wlg_bp = np.copy(wlg_g)
    wlg_rp = np.copy(wlg_g)

    wlg_g = wlg_g[np.where(g < 90)]
    g = list((g[np.where(g < 90)])[::-1])

    wlg_bp = wlg_bp[np.where(gbp < 90)]
    gbp = list((gbp[np.where(gbp < 90)])[::-1])

    wlg_rp = wlg_rp[np.where(grp < 90)]
    grp = list((grp[np.where(grp < 90)])[::-1])

    f4 = Filter()
    f4.name = "GEDR3_G"
    f4.central_wavelength = cf.g_edr3 * pow(10, 6) * u.micron
    f4.nu = list(np.array((cf.light_speed / (wlg_g * 1e-9)))[::-1]) * u.Hz
    f4.response = g
    f4.normalize()

    f5 = Filter()
    f5.name = "GEDR3_BP"
    f5.central_wavelength = cf.bp_edr3 * pow(10, 6) * u.micron
    f5.nu = list(np.array((cf.light_speed / (wlg_bp * 1e-9)))[::-1]) * u.Hz
    f5.response = gbp
    f5.normalize()

    f6 = Filter()
    f6.name = "GEDR3_RP"
    f6.central_wavelength = cf.rp_edr3 * pow(10, 6) * u.micron
    f6.nu = list(np.array((cf.light_speed / (wlg_rp * 1e-9)))[::-1]) * u.Hz
    f6.response = grp
    f6.normalize()

    wlg, resp, _ = np.loadtxt('response_functions/Response_WISE1.txt', unpack=True, usecols=(0, 1, 2))

    f7 = Filter()
    f7.name = "WISE1"
    f7.central_wavelength = cf.W1 * pow(10, 6) * u.micron
    f7.nu = list(np.array((cf.light_speed / (wlg * 1e-6)))[::-1]) * u.Hz
    f7.response = resp[::-1]
    f7.normalize()

    wlg, resp, _ = np.loadtxt('response_functions/Response_WISE2.txt', unpack=True, usecols=(0, 1, 2))

    f8 = Filter()
    f8.name = "WISE2"
    f8.central_wavelength = cf.W2 * pow(10, 6) * u.micron
    f8.nu = list(np.array((cf.light_speed / (wlg * 1e-6)))[::-1]) * u.Hz
    f8.response = resp[::-1]
    f8.normalize()

    wlg, resp, _ = np.loadtxt('response_functions/Response_WISE3.txt', unpack=True, usecols=(0, 1, 2))

    f9 = Filter()
    f9.name = "WISE3"
    f9.central_wavelength = cf.W3 * pow(10, 6) * u.micron
    f9.nu = list(np.array((cf.light_speed / (wlg * 1e-6)))[::-1]) * u.Hz
    f9.response = resp[::-1]
    f9.normalize()

    wlg, resp, _ = np.loadtxt('response_functions/Response_WISE4.txt', unpack=True, usecols=(0, 1, 2))

    f10 = Filter()
    f10.name = "WISE4"
    f10.central_wavelength = cf.W4 * pow(10, 6) * u.micron
    f10.nu = list(np.array((cf.light_speed / (wlg * 1e-6)))[::-1]) * u.Hz
    f10.response = resp[::-1]
    f10.normalize()

    resp, wlg = np.loadtxt('response_functions/Response_PACS70.txt', unpack=True, usecols=(0, 1))

    f11 = Filter()
    f11.name = "PACS1"
    f11.central_wavelength = cf.pacs70 * pow(10, 6) * u.micron
    f11.nu = list(np.array((cf.light_speed / (wlg * 1e-6)))[::-1]) * u.Hz
    f11.response = resp[::-1]
    f11.normalize()

    resp, wlg = np.loadtxt('response_functions/Response_PACS100.txt', unpack=True, usecols=(0, 1))

    f12 = Filter()
    f12.name = "PACS2"
    f12.central_wavelength = cf.pacs100 * pow(10, 6) * u.micron
    f12.nu = list(np.array((cf.light_speed / (wlg * 1e-6)))[::-1]) * u.Hz
    f12.response = resp[::-1]
    f12.normalize()

    convolve_model_dir(model_dir, [f, f2, f3, f4, f5, f6, f7, f8, f9, f10, f11, f12])


def convolve_gaia(model_dir):
    wlg, g, gbp, grp = np.loadtxt('response_functions/GaiaDR2_RevisedPassbands.dat', unpack=True, usecols=(0, 1, 3, 5))
    g = np.asarray(g)
    gbp = np.asarray(gbp)
    grp = np.asarray(grp)
    wlg_g = np.asarray(wlg)
    wlg_bp = np.copy(wlg_g)
    wlg_rp = np.copy(wlg_g)

    wlg_g = wlg_g[np.where(g < 90)]
    g = list((g[np.where(g < 90)])[::-1])

    wlg_bp = wlg_bp[np.where(gbp < 90)]
    gbp = list((gbp[np.where(gbp < 90)])[::-1])

    wlg_rp = wlg_rp[np.where(grp < 90)]
    grp = list((grp[np.where(grp < 90)])[::-1])

    f = Filter()
    f.name = "GDR2_G"
    f.central_wavelength = cf.g_dr2 * pow(10, 6) * u.micron
    f.nu = list(np.array((cf.light_speed / (wlg_g * 1e-9)))[::-1]) * u.Hz
    f.response = g
    f.normalize()

    f2 = Filter()
    f2.name = "GDR2_BP"
    f2.central_wavelength = cf.bp_dr2 * pow(10, 6) * u.micron
    f2.nu = list(np.array((cf.light_speed / (wlg_bp * 1e-9)))[::-1]) * u.Hz
    f2.response = gbp
    f2.normalize()

    f3 = Filter()
    f3.name = "GDR2_RP"
    f3.central_wavelength = cf.rp_dr2 * pow(10, 6) * u.micron
    f3.nu = list(np.array((cf.light_speed / (wlg_rp * 1e-9)))[::-1]) * u.Hz
    f3.response = grp
    f3.normalize()

    wlg, g, gbp, grp = np.loadtxt('response_functions/EDR3_passband.dat', unpack=True, usecols=(0, 1, 3, 5))
    g = np.asarray(g)
    gbp = np.asarray(gbp)
    grp = np.asarray(grp)
    wlg_g = np.asarray(wlg)
    wlg_bp = np.copy(wlg_g)
    wlg_rp = np.copy(wlg_g)

    wlg_g = wlg_g[np.where(g < 90)]
    g = list((g[np.where(g < 90)])[::-1])

    wlg_bp = wlg_bp[np.where(gbp < 90)]
    gbp = list((gbp[np.where(gbp < 90)])[::-1])

    wlg_rp = wlg_rp[np.where(grp < 90)]
    grp = list((grp[np.where(grp < 90)])[::-1])

    f4 = Filter()
    f4.name = "GEDR3_G"
    f4.central_wavelength = cf.g_edr3 * pow(10, 6) * u.micron
    f4.nu = list(np.array((cf.light_speed / (wlg_g * 1e-9)))[::-1]) * u.Hz
    f4.response = g
    f4.normalize()

    f5 = Filter()
    f5.name = "GEDR3_BP"
    f5.central_wavelength = cf.bp_edr3 * pow(10, 6) * u.micron
    f5.nu = list(np.array((cf.light_speed / (wlg_bp * 1e-9)))[::-1]) * u.Hz
    f5.response = gbp
    f5.normalize()

    f6 = Filter()
    f6.name = "GEDR3_RP"
    f6.central_wavelength = cf.rp_edr3 * pow(10, 6) * u.micron
    f6.nu = list(np.array((cf.light_speed / (wlg_rp * 1e-9)))[::-1]) * u.Hz
    f6.response = grp
    f6.normalize()

    convolve_model_dir(model_dir, [f, f2, f3, f4, f5, f6])


if __name__ == '__main__':
    print("hello")
    # create_wise_gaia_pacs_filters('sp--s-i')
    # convolve_gaia('spu-smi')