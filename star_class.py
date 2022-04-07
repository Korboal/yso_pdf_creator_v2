import os.path

import numpy as np
import config_file as cf
import lightcurve_class
from xmatched_tables_class import XMatchTables
from lightcurve_class import LightCurveZTF, LightCurveGaia, LightCurve, LightCurveNeowise
from astropy.io.votable import parse_single_table
import get_closest_nebula


class Star:
    def __init__(self, source_id: str, x_matched_tables_obj: XMatchTables):
        """
        Initializes the star object. Only info created at this stage is ID and initalized x-matched tables class.

        :param source_id: ID of the star, string object
        :param x_matched_tables_obj: The tables with the all x-matched stars; not necessarily including this star
        """
        self.source_id = source_id
        self.x_matched_tables_obj = x_matched_tables_obj

        # PDF canvas directory
        self.pdf_dir = cf.output_pdf_files + self.source_id + ".pdf"

        # Light curve parameters
        self.gaia_g_light_curve, self.gaia_bp_light_curve, self.gaia_rp_light_curve = None, None, None

        self.light_curve_output_period_textfile = cf.output_fitted_period_text
        self.light_curve_gaia_input_file_path = cf.input_light_curve_to_analyse_path + self.source_id + ".xml"
        self.periodogram_gaia_g_png_directory = cf.output_periodogram_gaia_g_png + self.source_id + ".png"
        self.periodogram_gaia_g_pdf_directory = cf.output_periodogram_gaia_g_pdf + self.source_id + ".pdf"
        self.fitted_curve_gaia_g_output_png = cf.output_light_curve_g_band_png + self.source_id + ".png"
        self.fitted_curve_gaia_g_output_pdf = cf.output_light_curve_g_band_pdf + self.source_id + ".pdf"
        self.raw_data_gaia_g_band_output_png = cf.output_raw_data_g_band_png + self.source_id + ".png"
        self.raw_data_gaia_g_band_output_pdf = cf.output_raw_data_g_band_pdf + self.source_id + ".pdf"
        self.folded_light_curve_gaia_g_output_png = cf.output_folded_light_curve_png + self.source_id + ".png"
        self.folded_light_curve_gaia_g_output_pdf = cf.output_folded_light_curve_pdf + self.source_id + ".pdf"
        self.raw_data_gaia_rp_band_output_png = cf.output_rp_raw_data_png + self.source_id + ".png"
        self.raw_data_gaia_rp_band_output_pdf = cf.output_rp_raw_data_pdf + self.source_id + ".pdf"
        self.raw_data_gaia_bp_band_output_png = cf.output_bp_raw_data_png + self.source_id + ".png"
        self.raw_data_gaia_bp_band_output_pdf = cf.output_bp_raw_data_pdf + self.source_id + ".pdf"
        self.folded_light_curve_gaia_bp_with_gaia_g_output_png = cf.output_folded_light_curve_png + self.source_id + "bp_with_gaia_g.png"
        self.folded_light_curve_gaia_bp_with_gaia_g_output_pdf = cf.output_folded_light_curve_pdf + self.source_id + "bp_with_gaia_g.pdf"
        self.folded_light_curve_gaia_rp_with_gaia_g_output_png = cf.output_folded_light_curve_png + self.source_id + "rp_with_gaia_g.png"
        self.folded_light_curve_gaia_rp_with_gaia_g_output_pdf = cf.output_folded_light_curve_pdf + self.source_id + "rp_with_gaia_g.pdf"
        self.output_gaia_all_bands_raw_data_png = cf.output_gaia_all_bands_raw_data_png + self.source_id + ".png"
        self.output_frequency_periodogram_gaia_g_png = cf.output_frequency_periodogram_gaia_g_png + self.source_id + ".png"
        self.output_frequency_periodogram_gaia_g_pdf = cf.output_frequency_periodogram_gaia_g_pdf + self.source_id + ".pdf"

        self.output_txt_params = cf.output_fitted_param

        # ZTF Light curve parameters
        self.ztf_g_light_curve, self.ztf_r_light_curve, self.ztf_i_light_curve = None, None, None

        self.ztf_light_curve_file_path = cf.input_ztf_lightcurves_to_analyse + self.source_id + ".vot"
        self.ztf_output_period = cf.output_ztf_lightcurves_period
        self.ztf_output_pictures_raw = cf.output_ztf_lightcurves_pictures + self.source_id + "_raw.png"
        self.ztf_output_pictures_folded = cf.output_ztf_lightcurves_pictures + self.source_id + "_folded.png"
        self.ztf_output_pictures_folded_g_with_gaia_g_fit = cf.output_ztf_lightcurves_pictures + self.source_id + "_folded_ztf_g_with_gaia_g.png"
        self.ztf_output_pictures_folded_r_with_gaia_g_fit = cf.output_ztf_lightcurves_pictures + self.source_id + "_folded_ztf_g_with_gaia_r.png"
        self.ztf_output_pictures_folded_g_with_ztf_r_fit = cf.output_ztf_lightcurves_pictures + self.source_id + "_folded_ztf_g_with_ztf_r.png"
        self.ztf_output_pictures_folded_r_with_ztf_g_fit = cf.output_ztf_lightcurves_pictures + self.source_id + "_folded_ztf_r_with_ztf_g.png"
        self.ztf_output_pictures_fit_ztf_g = cf.output_ztf_fit + self.source_id + "_ztf_g.png"
        self.ztf_output_pictures_fit_ztf_r = cf.output_ztf_fit + self.source_id + "_ztf_r.png"
        self.ztf_output_pictures_folded_ztf_g = cf.output_ztf_folded + self.source_id + "_ztf_g.png"
        self.ztf_output_pictures_folded_ztf_r = cf.output_ztf_folded + self.source_id + "_ztf_r.png"
        self.ztf_output_pictures_periodogram_ztf_g_png = cf.output_periodogram_ztf_g_png + self.source_id + ".png"
        self.ztf_output_pictures_periodogram_ztf_g_pdf = cf.output_periodogram_ztf_g_pdf + self.source_id + ".pdf"
        self.ztf_output_pictures_periodogram_ztf_r_png = cf.output_periodogram_ztf_r_png + self.source_id + ".png"
        self.ztf_output_pictures_periodogram_ztf_r_pdf = cf.output_periodogram_ztf_r_pdf + self.source_id + ".pdf"
        self.output_frequency_periodogram_ztf_g_png = cf.output_frequency_periodogram_ztf_g_png + self.source_id + ".png"
        self.output_frequency_periodogram_ztf_g_pdf = cf.output_frequency_periodogram_ztf_g_pdf + self.source_id + ".pdf"
        self.output_frequency_periodogram_ztf_r_png = cf.output_frequency_periodogram_ztf_r_png + self.source_id + ".png"
        self.output_frequency_periodogram_ztf_r_pdf = cf.output_frequency_periodogram_ztf_r_pdf + self.source_id + ".pdf"

        self.output_multiband_frequency_periodogram_gaia_png = cf.output_multiband_frequency_periodogram_png + self.source_id + "_gaia.png"
        self.output_multiband_frequency_periodogram_ztf_png = cf.output_multiband_frequency_periodogram_png + self.source_id + "_ztf.png"
        self.output_multiband_frequency_periodogram_all_png = cf.output_multiband_frequency_periodogram_png + self.source_id + "_all.png"

        self.output_multiband_gaia_fit_gaia_g_png = cf.output_multiband_fits_png + self.source_id + "_gaia_multiband_gaia_g.png"
        self.output_multiband_gaia_fit_gaia_bp_png = cf.output_multiband_fits_png + self.source_id + "_gaia_multiband_gaia_bp.png"
        self.output_multiband_gaia_fit_gaia_rp_png = cf.output_multiband_fits_png + self.source_id + "_gaia_multiband_gaia_rp.png"

        self.output_multiband_ztf_fit_ztf_g_png = cf.output_multiband_fits_png + self.source_id + "_ztf_multiband_ztf_g.png"
        self.output_multiband_ztf_fit_ztf_r_png = cf.output_multiband_fits_png + self.source_id + "_ztf_multiband_ztf_r.png"

        self.output_multiband_all_fit_gaia_g_png = cf.output_multiband_fits_png + self.source_id + "_all_multiband_gaia_g.png"
        self.output_multiband_all_fit_gaia_bp_png = cf.output_multiband_fits_png + self.source_id + "_all_multiband_gaia_bp.png"
        self.output_multiband_all_fit_gaia_rp_png = cf.output_multiband_fits_png + self.source_id + "_all_multiband_gaia_rp.png"
        self.output_multiband_all_fit_ztf_g_png = cf.output_multiband_fits_png + self.source_id + "_all_multiband_ztf_g.png"
        self.output_multiband_all_fit_ztf_r_png = cf.output_multiband_fits_png + self.source_id + "_all_multiband_ztf_r.png"


        # SED line fit parameters
        self.ir_slope25 = None
        self.ir_slope20 = None

        self.sed_points_prepared = False
        self.enough_ir_points = None
        self.sed_vizier_data_file_path = cf.input_sed_files_to_analyse_path + self.source_id + ".vot"
        self.x_good, self.y_good, self.error_good, self.x_upper, self.y_upper, self.error_upper, self.x_viz, \
        self.y_viz, self.error_viz, self.separation_arcsec, self.separation_table = None, None, None, None, None, \
                                                                                    None, None, None, None, None, None
        self.x_sed_linear_fit, self.y_sed_linear_fit, self.error_sed_linear_fit = None, None, None
        self.x_sed_all_ir, self.y_sed_all_ir, self.error_sed_all_ir = None, None, None
        self.x_sed_mir, self.y_sed_mir, self.error_sed_mir = None, None, None
        self.sed_linefit_rayleigh_jeans_const = None

        self.ir_slope_directory_png = cf.output_sed_ir_slope_fit_png + self.source_id + ".png"
        self.ir_slope_directory_pdf = cf.output_sed_ir_slope_fit_pdf + self.source_id + ".pdf"
        self.sed_integrated_excess_png = cf.output_sed_integrated_excess_figure_png + self.source_id + ".png"
        self.sed_integrated_excess_pdf = cf.output_sed_integrated_excess_figure_pdf + self.source_id + ".pdf"
        self.sed_fit_directory_png = cf.output_sed_fit_png + self.source_id + ".png"
        self.sed_fit_directory_pdf = cf.output_sed_fit_pdf + self.source_id + ".pdf"
        self.sed_bar_dir_png = cf.output_sed_bar_png + self.source_id + ".png"
        self.sed_bar_dir_pdf = cf.output_sed_bar_pdf + self.source_id + ".pdf"

        # SED BB fit
        self.output_sed_temp_pic = cf.output_sed_temp_pic + self.source_id + ".png"
        self.output_sed_temp_fits_txt = cf.output_sed_temp_fits_txt

    def __setstate__(self, state: dict):
        """
        Inserts the parameters for the star based on its Gaia information from both GDR2 and GEDR3

        :param state: Dictionary with Gaia information table with GDR2 and GEDR3
        """

        self.ra = state['ra']
        self.dec = state['dec']
        self.distance_ge3 = state['dist']
        self.parallax_ge3 = state['parallax_ge3']
        self.parallax_error_ge3 = state['parallax_error_ge3']
        self.l = state['l']
        self.b = state['b']
        self.pmra = state['pmra']
        self.pmdec = state['pmdec']
        self.g_mag_g2 = state["g_mag_g2"]
        self.bp_mag_g2 = state["bp_mag_g2"]
        self.rp_mag_g2 = state["rp_mag_g2"]
        self.rad_vel_g2 = state["rad_vel_g2"]
        self.period_g2 = state["period_g2"]
        self.period_err_g2 = state["period_err_g2"]
        self.extinction_g2 = state["extinction_g2"]
        self.teff_template_g2 = state["teff_template_g2"]
        self.teff_val_g2 = state["teff_val_g2"]
        self.name_simbad = state["name_simbad"]
        self.main_type_simbad = state["main_type_simbad"]
        self.other_types_simbad = state["other_types_simbad"]
        self.g_mag_ge3 = state["g_mag_ge3"]
        self.bp_mag_ge3 = state["bp_mag_ge3"]
        self.rp_mag_ge3 = state["rp_mag_ge3"]
        self.radius_g2 = state["radius_g2"]
        self.radius_lower_g2 = state["radius_lower_g2"]
        self.radius_upper_g2 = state["radius_upper_g2"]

    def print_source_id(self):
        """
        Print the source id
        """
        print(self.source_id)

    def lightcurve_fit_and_plot(self, save_variables=False, save_images=False, show_images=False, print_variables=False,
                                manual_period_guess=0):
        """
        Does Gaia light curve fit using LS guess with non-linear sin fit.

        :param save_variables: If want to save variables
        :param save_images: If want to save images
        :param show_images: If want to show images
        :param print_variables: If want to print images
        :param manual_period_guess: If want to do a manual guess fit for the Gaia light curve
        """
        from tools import save_in_txt_topcat

        if os.path.isfile(self.light_curve_gaia_input_file_path):
            gaia_votable_data = parse_single_table(self.light_curve_gaia_input_file_path)
        else:
            print("Cannot find Gaia light curve for", self.source_id)
            gaia_votable_data = None

        self.gaia_g_light_curve = LightCurveGaia(gaia_votable_data, "G", save_image=save_images, show_image=show_images,
                                                 save_pdfs=cf.save_pdfs)
        self.gaia_bp_light_curve = LightCurveGaia(gaia_votable_data, "BP", save_image=save_images,
                                                  show_image=show_images, save_pdfs=cf.save_pdfs)
        self.gaia_rp_light_curve = LightCurveGaia(gaia_votable_data, "RP", save_image=save_images,
                                                  show_image=show_images, save_pdfs=cf.save_pdfs)

        if cf.detrend:
            self.gaia_g_light_curve.remove_long_term_trend()
            self.gaia_bp_light_curve.remove_long_term_trend()
            self.gaia_rp_light_curve.remove_long_term_trend()

        self.gaia_g_light_curve.fit_light_curve(print_variables=print_variables, manual_period_guess=manual_period_guess)

        gaia_g_bp_correlation = lightcurve_class.find_correlation_of_time_series(self.gaia_g_light_curve, self.gaia_bp_light_curve)
        gaia_g_rp_correlation = lightcurve_class.find_correlation_of_time_series(self.gaia_g_light_curve, self.gaia_rp_light_curve)

        fit_func_bp_with_gaia_g, error_bp_with_gaia_g, amp_bp_with_gaia_g, nrmse_bp_with_gaia_g = self.gaia_bp_light_curve.draw_folded_light_curve_with_other_function(
            self.gaia_g_light_curve.fit_result,
            "Gaia G",
            image_output_png=self.folded_light_curve_gaia_bp_with_gaia_g_output_png,
            image_output_pdf=self.folded_light_curve_gaia_bp_with_gaia_g_output_pdf)
        fit_func_rp_with_gaia_g, error_rp_with_gaia_g, amp_rp_with_gaia_g, nrmse_rp_with_gaia_g = self.gaia_rp_light_curve.draw_folded_light_curve_with_other_function(
            self.gaia_g_light_curve.fit_result,
            "Gaia G",
            image_output_png=self.folded_light_curve_gaia_rp_with_gaia_g_output_png,
            image_output_pdf=self.folded_light_curve_gaia_rp_with_gaia_g_output_pdf)

        self.gaia_bp_light_curve.nrmse_using_gaia_g = nrmse_bp_with_gaia_g
        self.gaia_rp_light_curve.nrmse_using_gaia_g = nrmse_rp_with_gaia_g

        peak1_period, peak1_value, peak2_period, peak2_value = self.gaia_g_light_curve.draw_periodogram(image_output_png=self.periodogram_gaia_g_png_directory,
                                                    image_output_pdf=self.periodogram_gaia_g_pdf_directory,
                                                    second_fit_periods=[self.period_g2], second_fit_names=["GDR2"], find_peaks=True)

        if save_images or show_images:
            self.gaia_g_light_curve.draw_raw_light_curve(image_output_png=self.raw_data_gaia_g_band_output_png,
                                                         image_output_pdf=self.raw_data_gaia_g_band_output_pdf)
            self.gaia_bp_light_curve.draw_raw_light_curve(image_output_png=self.raw_data_gaia_bp_band_output_png,
                                                          image_output_pdf=self.raw_data_gaia_bp_band_output_pdf)
            self.gaia_rp_light_curve.draw_raw_light_curve(image_output_png=self.raw_data_gaia_rp_band_output_png,
                                                          image_output_pdf=self.raw_data_gaia_rp_band_output_pdf)

            self.gaia_g_light_curve.draw_folded_light_curve(image_output_png=self.folded_light_curve_gaia_g_output_png,
                                                            image_output_pdf=self.folded_light_curve_gaia_g_output_pdf)

            self.gaia_g_light_curve.draw_fitted_light_curve(image_output_png=self.fitted_curve_gaia_g_output_png, image_output_pdf=self.fitted_curve_gaia_g_output_pdf, do_second_fit=True, second_fit_period=self.period_g2, second_fit_name="GDR2")

            self.gaia_g_light_curve.draw_frequency_gram(image_output_png=self.output_frequency_periodogram_gaia_g_png, image_output_pdf=self.output_frequency_periodogram_gaia_g_pdf,
                                                        second_fit_periods=[self.period_g2], second_fit_names=["GDR2"])

            lightcurve_class.plot_three_gaia_curves(self, image_output_png=self.output_gaia_all_bands_raw_data_png)

        if save_variables:
            gaia_g_median_ls_power_old_psd, gaia_g_median_ls_power_new_psd, gaia_g_ls_power_arg_max_old_psd, gaia_g_ls_power_arg_max_new_psd = self.get_median_ls_power(
                self.gaia_g_light_curve, normalization='psd', recalculate_old_periodogram=True)

            gaia_g_median_ls_power_old_normal, gaia_g_median_ls_power_new_normal, gaia_g_ls_power_arg_max_old_normal, gaia_g_ls_power_arg_max_new_normal = self.get_median_ls_power(
                self.gaia_g_light_curve, normalization='standard', recalculate_old_periodogram=False)
            save_in_txt_topcat(
                [self.source_id, self.gaia_g_light_curve.period_fit, self.gaia_g_light_curve.max_diff,
                 self.gaia_g_light_curve.percentile_q95_5,  self.gaia_g_light_curve.st_dev,
                 self.gaia_g_light_curve.avg_error, self.gaia_g_light_curve.fit_function_error,
                 self.gaia_g_light_curve.fit_result['amp'], self.gaia_g_light_curve.data_length,
                 self.gaia_g_light_curve.skewness, self.gaia_g_light_curve.total_obs_time,
                 self.gaia_bp_light_curve.data_length, self.gaia_rp_light_curve.data_length,
                 gaia_g_bp_correlation, gaia_g_rp_correlation,
                 error_bp_with_gaia_g, amp_bp_with_gaia_g, nrmse_bp_with_gaia_g,
                 error_rp_with_gaia_g, amp_rp_with_gaia_g, nrmse_rp_with_gaia_g,
                 peak1_period, peak1_value, peak2_period, peak2_value,
                 self.gaia_g_light_curve.fit_result["ls_fap"], np.median(self.gaia_g_light_curve.ls_power),
                 self.gaia_g_light_curve.nrmse_fit, gaia_g_median_ls_power_old_psd, gaia_g_median_ls_power_new_psd,
                 gaia_g_ls_power_arg_max_old_psd, gaia_g_ls_power_arg_max_new_psd, gaia_g_median_ls_power_old_normal,
                 gaia_g_median_ls_power_new_normal, gaia_g_ls_power_arg_max_old_normal,
                 gaia_g_ls_power_arg_max_new_normal], self.light_curve_output_period_textfile)

    def prepare_sed_points(self):
        from sed_linefit_v4_0 import clean_sed_points
        from tools import take_within_x_boundaries

        self.x_good, self.y_good, self.error_good, self.x_upper, self.y_upper, self.error_upper, self.x_viz, \
        self.y_viz, self.error_viz, self.separation_arcsec, self.separation_table = \
            clean_sed_points(self.sed_vizier_data_file_path, self.source_id, self.x_matched_tables_obj)

        if np.size(np.where(np.logical_and(cf.ir_excess_start <= self.x_good, self.x_good <= cf.ir_excess_end))) < 2:
            self.enough_ir_points = False
        else:
            self.enough_ir_points = True
        # For fit of the linear SED line with slope -3 (Rayleigh Jeans)
        self.x_sed_linear_fit, self.y_sed_linear_fit, self.error_sed_linear_fit = take_within_x_boundaries(self.x_good,
                                                                                                           self.y_good,
                                                                                                           self.error_good,
                                                                                                           cf.sed_linear_line_wavelength_start,
                                                                                                           cf.sed_linear_line_wavelength_end,
                                                                                                           0)

        if np.size(self.x_sed_linear_fit) == 0:
            from tools import find_nearest_index
            index = find_nearest_index(self.x_good,
                                       (cf.sed_linear_line_wavelength_start + cf.sed_linear_line_wavelength_end) / 2)
            self.x_sed_linear_fit = np.array([self.x_good[index]])
            self.y_sed_linear_fit = np.array([self.y_good[index]])
            self.error_sed_linear_fit = np.array([self.error_good[index]])

        # For finding average ratio, without limit on where excess ends
        self.x_sed_all_ir, self.y_sed_all_ir, self.error_sed_all_ir = take_within_x_boundaries(self.x_good,
                                                                                               self.y_good,
                                                                                               self.error_good,
                                                                                               cf.sed_linear_line_wavelength_start,
                                                                                               pow(10.0, 99), 2)
        # For finding average ratio, with limit being 25 um
        self.x_sed_mir, self.y_sed_mir, self.error_sed_mir = take_within_x_boundaries(self.x_good,
                                                                                      self.y_good,
                                                                                      self.error_good,
                                                                                      cf.sed_linear_line_wavelength_start,
                                                                                      cf.ir_excess_end, 2)

        self.sed_points_prepared = True

    def fit_sed_linear_fit(self):
        from sed_linefit_v4_0 import fit_sed_linear_fit
        self.sed_linefit_rayleigh_jeans_const = fit_sed_linear_fit(self)

    def sed_line_fit_and_plot(self, fits_to_do: str, save_variables=False,
                              save_images=False, show_images=False, print_variables=False):
        from sed_linefit_v4_0 import calculate_sed_excess_from_points, calculate_ir_slope, \
            extrapolate_and_integrate_sed_excess, plot_image
        from tools import save_in_txt_topcat, identify_yso_class
        if not self.sed_points_prepared:
            self.prepare_sed_points()

        if len(fits_to_do) < 7:  # Checks if fits_to_do is not long enough. If not, just makes it so other plots are not done
            fits_to_do += "0000000000"

        param_to_save = [self.source_id, str(self.enough_ir_points)]

        if fits_to_do[0] == "1":
            self.fit_sed_linear_fit()

        if fits_to_do[1] == "1":
            if self.sed_linefit_rayleigh_jeans_const is None:
                self.fit_sed_linear_fit()
            sed_excess = calculate_sed_excess_from_points(self.x_sed_all_ir, self.y_sed_all_ir, self.error_sed_all_ir,
                                                          self.sed_linefit_rayleigh_jeans_const)
            param_to_save += sed_excess

        if fits_to_do[2] == "1":
            if self.sed_linefit_rayleigh_jeans_const is None:
                self.fit_sed_linear_fit()
            sed_excess_2 = calculate_sed_excess_from_points(self.x_sed_mir, self.y_sed_mir, self.error_sed_mir,
                                                            self.sed_linefit_rayleigh_jeans_const)
            param_to_save += sed_excess_2

        if fits_to_do[3] == "1":
            ir_slope = calculate_ir_slope(self, print_variables,
                                          show_images, save_images, ir_slope_end=cf.ir_slope_end)
            self.ir_slope25 = ir_slope[0]
            param_to_save += ir_slope
            param_to_save += [identify_yso_class(self.ir_slope25)]

        if fits_to_do[4] == "1":
            ir_slope2 = calculate_ir_slope(self, print_variables,
                                           show_images, save_images, ir_slope_end=cf.ir_slope_end2)
            self.ir_slope20 = ir_slope2[0]
            param_to_save += ir_slope2
            param_to_save += [identify_yso_class(self.ir_slope20)]

        if fits_to_do[5] == "1":
            if self.sed_linefit_rayleigh_jeans_const is None:
                self.fit_sed_linear_fit()
            excess = extrapolate_and_integrate_sed_excess(self, print_variables, show_images, save_images)
            param_to_save += excess

        if fits_to_do[6] == "1":
            if self.sed_linefit_rayleigh_jeans_const is None:
                self.fit_sed_linear_fit()
            plot_image(self, show_images, save_images)

        if save_variables:
            save_in_txt_topcat(param_to_save, self.output_txt_params)

    def create_pdf_canvas_for_a_star(self, gaia_lpv_table):
        from create_pdf_from_images_and_text_v3 import create_new_canvas_for_a_star
        create_new_canvas_for_a_star(self, gaia_lpv_table)

    def sed_bb_fit(self, save_image: bool, show_image: bool, save_variables: bool):
        from my_sedfit_v6 import bb_model_fit_with_image
        if not self.sed_points_prepared:
            self.prepare_sed_points()
        bb_fits_values = bb_model_fit_with_image(self, save_image, show_image)
        if save_variables:
            from tools import save_in_txt_topcat
            save_in_txt_topcat(bb_fits_values, self.output_sed_temp_fits_txt)

    def sed_fitter_fit(self, file_sed_points, filters_total):
        if not self.sed_points_prepared:
            self.prepare_sed_points()

        from tools import save_in_txt_topcat, get_sedfitter_star_data

        star_info = get_sedfitter_star_data(self, filters_total)
        save_in_txt_topcat(list(star_info), file_sed_points)

    def print_amount_of_ir_points(self):
        from tools import take_within_x_boundaries
        if not self.sed_points_prepared:
            self.prepare_sed_points()
        x, y, err = take_within_x_boundaries(self.x_good, self.y_good, self.error_good, cf.ir_excess_start,
                                             cf.ir_excess_end, 0)
        print(self.source_id + "\t" + str(np.size(x)))

    def analyse_ztf_lightcurves(self, save_variables, show_variables, save_pictures, show_pictures, do_multiband_periodogram, manual_period_guess=0):
        from tools import save_in_txt_topcat

        if self.gaia_g_light_curve is None:
            self.lightcurve_fit_and_plot(save_variables=False, save_images=False, show_images=False, print_variables=False, manual_period_guess=manual_period_guess)

        if os.path.isfile(self.ztf_light_curve_file_path):
            ztf_votable_data = parse_single_table(self.ztf_light_curve_file_path)
        else:
            print("Cannot find ZTF light curve for", self.source_id)
            ztf_votable_data = None

        self.ztf_g_light_curve = LightCurveZTF(ztf_votable_data, "g", save_image=save_pictures, save_pdfs=cf.save_pdfs,
                                               show_image=show_pictures)
        self.ztf_r_light_curve = LightCurveZTF(ztf_votable_data, "r", save_image=save_pictures, save_pdfs=cf.save_pdfs,
                                               show_image=show_pictures)
        self.ztf_i_light_curve = LightCurveZTF(ztf_votable_data, "i", save_image=save_pictures, save_pdfs=cf.save_pdfs,
                                               show_image=show_pictures)

        """max_time = np.max(self.ztf_r_light_curve.data_t)
        dt = int(50)
        for i in range(0, int(max_time), dt):
            self.ztf_r_light_curve.choose_spectral_window(i, i+dt)
            self.ztf_r_light_curve.fit_light_curve(show_variables, manual_period_guess=manual_period_guess)
            self.ztf_r_light_curve.draw_folded_light_curve(image_output_png=self.ztf_output_pictures_folded_ztf_r)
            self.ztf_r_light_curve = LightCurveZTF(ztf_votable_data, "r", save_image=save_pictures,
                                                   save_pdfs=cf.save_pdfs,
                                                   show_image=show_pictures)"""

        #difference_between_gr = self.ztf_g_light_curve.mean - self.ztf_r_light_curve.mean
        #if np.size(self.ztf_r_light_curve.data_y) > 0:
        #    self.ztf_r_light_curve.data_y = self.ztf_r_light_curve.data_y + difference_between_gr

        #self.ztf_g_light_curve.data_t = np.append(self.ztf_g_light_curve.data_t, self.ztf_r_light_curve.data_t)    # combining g and r bands
        #self.ztf_g_light_curve.data_y = np.append(self.ztf_g_light_curve.data_y, self.ztf_r_light_curve.data_y)
        #self.ztf_g_light_curve.data_error = np.append(self.ztf_g_light_curve.data_error, self.ztf_r_light_curve.data_error)
        #self.ztf_g_light_curve.calculate_light_curve_properties()

        if cf.detrend:
            self.ztf_g_light_curve.remove_long_term_trend()
            self.ztf_r_light_curve.remove_long_term_trend()
            self.ztf_i_light_curve.remove_long_term_trend()

        lightcurve_class.slider_folded_light_curve(self.ztf_r_light_curve, 25, 35)  # 9.353221424477987

        """lightcurve_class.slider_folded_light_curve(self.ztf_r_light_curve, 150, 500)
        #lightcurve_class.slider_folded_light_curve_freq(self.ztf_g_light_curve, 0.05, 1)

        other_func = {"period": (53 * 2), "phase": 0} #220.18971637106395
        self.gaia_g_light_curve.show_image = True
        self.gaia_g_light_curve.draw_folded_light_curve_with_other_function(other_func, "specific", fit_phase=True)
        self.ztf_g_light_curve.draw_folded_light_curve_with_other_function(other_func, "specific", fit_phase=True)
        self.ztf_r_light_curve.draw_folded_light_curve_with_other_function(other_func, "specific", fit_phase=True)"""

        if cf.debug_mode and False:
            self.gaia_g_light_curve.show_image = True
            self.gaia_g_light_curve.fit_two_periods_new_version()
            self.ztf_g_light_curve.fit_two_periods_new_version()
            self.ztf_r_light_curve.fit_two_periods_new_version()

        #self.ztf_g_light_curve.draw_linear_trend_removal_example()

        self.ztf_r_light_curve.fit_light_curve(show_variables, manual_period_guess=manual_period_guess)
        self.ztf_g_light_curve.fit_light_curve(show_variables, manual_period_guess=manual_period_guess)
        self.ztf_i_light_curve.fit_light_curve(show_variables, manual_period_guess=manual_period_guess)

        self.ztf_r_light_curve.draw_spectral_window()

        #self.gaia_g_light_curve.draw_folded_with_colored_time()
        #self.ztf_g_light_curve.draw_folded_with_colored_time()
        #self.ztf_r_light_curve.draw_folded_with_colored_time()

        lightcurve_class.plot_g_r_variation(self.ztf_g_light_curve, self.ztf_r_light_curve)

        #self.ztf_r_light_curve.fitting_example_frequency_periodogram()

        #self.lightcurve_fit_and_plot(save_variables=cf.lightcurve_fit_save_variables, save_images=cf.lightcurve_fit_save_images, show_images=cf.lightcurve_fit_show_images, print_variables=cf.lightcurve_fit_print_variables, manual_period_guess=self.ztf_r_light_curve.period_fit)

        #other_func = {"period": self.ztf_r_light_curve.period_fit, "phase": 0}
        #self.gaia_g_light_curve.show_image = True
        #self.gaia_g_light_curve.draw_folded_light_curve_with_other_function(other_func, "ZTF", fit_phase=True)


        # 2 PERIODS
        #self.gaia_g_light_curve.fit_light_curve_using_two_periods(period1=161, period2=81)
        #self.ztf_g_light_curve.fit_light_curve_using_two_periods(period1=161, period2=81)
        #self.ztf_r_light_curve.fit_light_curve_using_two_periods(period1=161, period2=81)

        fit_func_g_with_gaia_g, error_g_with_gaia_g, amp_g_with_gaia_g, nrmse_g_with_gaia_g = self.ztf_g_light_curve.draw_folded_light_curve_with_other_function(self.gaia_g_light_curve.fit_result,
                                                                               "Gaia G",
                                                                               image_output_png=self.ztf_output_pictures_folded_g_with_gaia_g_fit, fit_phase=True)
        fit_func_r_with_gaia_g, error_r_with_gaia_g, amp_r_with_gaia_g, nrmse_r_with_gaia_g = self.ztf_r_light_curve.draw_folded_light_curve_with_other_function(self.gaia_g_light_curve.fit_result,
                                                                               "Gaia G",
                                                                               image_output_png=self.ztf_output_pictures_folded_r_with_gaia_g_fit, fit_phase=True)

        self.ztf_g_light_curve.nrmse_using_gaia_g = nrmse_g_with_gaia_g
        self.ztf_r_light_curve.nrmse_using_gaia_g = nrmse_r_with_gaia_g

        fit_func_g_with_ztf_r, error_g_with_ztf_r, amp_g_with_ztf_r, nrmse_g_with_ztf_r = self.ztf_g_light_curve.draw_folded_light_curve_with_other_function(
                self.ztf_r_light_curve.fit_result,
                "ZTF r",
                image_output_png=self.ztf_output_pictures_folded_g_with_ztf_r_fit, fit_phase=False)
        fit_func_r_with_ztf_g, error_r_with_ztf_g, amp_r_with_ztf_g, nrmse_r_with_ztf_g = self.ztf_r_light_curve.draw_folded_light_curve_with_other_function(
                self.ztf_g_light_curve.fit_result,
                "ZTF g",
                image_output_png=self.ztf_output_pictures_folded_r_with_ztf_g_fit, fit_phase=False)

        self.ztf_g_light_curve.nrmse_using_other_ztf = nrmse_g_with_ztf_r
        self.ztf_r_light_curve.nrmse_using_other_ztf = nrmse_r_with_ztf_g

        peak1_period_ztf_g, peak1_value_ztf_g, peak2_period_ztf_g, peak2_value_ztf_g = self.ztf_g_light_curve.draw_periodogram(image_output_png=self.ztf_output_pictures_periodogram_ztf_g_png,
                                                     image_output_pdf=self.ztf_output_pictures_periodogram_ztf_g_pdf,
                                                     second_fit_periods=[self.period_g2, self.gaia_g_light_curve.period_fit, self.ztf_r_light_curve.period_fit],
                                                     second_fit_names=["GDR2", "Gaia G fit", "ZTF r"], find_peaks=True)

        peak1_period_ztf_r, peak1_value_ztf_r, peak2_period_ztf_r, peak2_value_ztf_r = self.ztf_r_light_curve.draw_periodogram(image_output_png=self.ztf_output_pictures_periodogram_ztf_r_png,
                                                image_output_pdf=self.ztf_output_pictures_periodogram_ztf_r_pdf,
                                                second_fit_periods=[self.period_g2,
                                                                    self.gaia_g_light_curve.period_fit,
                                                                    self.ztf_g_light_curve.period_fit],
                                                second_fit_names=["GDR2", "Gaia G fit", "ZTF g"], find_peaks=True)

        if do_multiband_periodogram:
            # Multiband periodograms
            # Multiband of Gaia G, BP, RP
            peak1_period_multiband_gaia, peak1_value_multiband_gaia, peak2_period_multiband_gaia, peak2_value_multiband_gaia, median_ls_power_multiband_gaia = lightcurve_class.multi_band_periodogram(
                [self.gaia_g_light_curve, self.gaia_bp_light_curve, self.gaia_rp_light_curve], ["G", "BP", "RP"], save_pictures,
                show_pictures, image_output_png=self.output_multiband_frequency_periodogram_gaia_png)
            # Multiband of ZTF g, r, i
            peak1_period_multiband_ztf, peak1_value_multiband_ztf, peak2_period_multiband_ztf, peak2_value_multiband_ztf, median_ls_power_multiband_ztf = lightcurve_class.multi_band_periodogram(
                [self.ztf_g_light_curve, self.ztf_r_light_curve, self.ztf_i_light_curve], ["ztf_g", "ztf_r", "ztf_i"], save_pictures,
                show_pictures, image_output_png=self.output_multiband_frequency_periodogram_ztf_png)
            # Multiband of Gaia and ZTF all bands
            peak1_period_multiband_all, peak1_value_multiband_all, peak2_period_multiband_all, peak2_value_multiband_all, median_ls_power_multiband_all = lightcurve_class.multi_band_periodogram(
                [self.gaia_g_light_curve, self.gaia_bp_light_curve, self.gaia_rp_light_curve, self.ztf_g_light_curve,
                 self.ztf_r_light_curve, self.ztf_i_light_curve], ["G", "BP", "RP", "ztf_g", "ztf_r", "ztf_i"], save_pictures,
                show_pictures, image_output_png=self.output_multiband_frequency_periodogram_all_png)
        else:
            peak1_period_multiband_gaia, peak1_value_multiband_gaia, peak2_period_multiband_gaia, peak2_value_multiband_gaia = cf.dflt_no_vle, cf.dflt_no_vle, cf.dflt_no_vle, cf.dflt_no_vle
            peak1_period_multiband_ztf, peak1_value_multiband_ztf, peak2_period_multiband_ztf, peak2_value_multiband_ztf = cf.dflt_no_vle, cf.dflt_no_vle, cf.dflt_no_vle, cf.dflt_no_vle
            peak1_period_multiband_all, peak1_value_multiband_all, peak2_period_multiband_all, peak2_value_multiband_all = cf.dflt_no_vle, cf.dflt_no_vle, cf.dflt_no_vle, cf.dflt_no_vle
            median_ls_power_multiband_gaia, median_ls_power_multiband_ztf, median_ls_power_multiband_all = cf.dflt_no_vle, cf.dflt_no_vle, cf.dflt_no_vle

        if show_pictures or save_pictures:
            from lightcurve_class import draw_ztf_and_gaia_light_curve_graphs

            draw_ztf_and_gaia_light_curve_graphs(self, show_pictures, save_pictures)

            self.ztf_g_light_curve.draw_fitted_light_curve(image_output_png=self.ztf_output_pictures_fit_ztf_g)
            self.ztf_g_light_curve.draw_folded_light_curve(image_output_png=self.ztf_output_pictures_folded_ztf_g)

            self.ztf_r_light_curve.draw_fitted_light_curve(image_output_png=self.ztf_output_pictures_fit_ztf_r)
            self.ztf_r_light_curve.draw_folded_light_curve(image_output_png=self.ztf_output_pictures_folded_ztf_r)

            self.gaia_g_light_curve.save_image = save_pictures
            self.gaia_g_light_curve.show_image = show_pictures
            self.gaia_g_light_curve.draw_periodogram(image_output_png=self.periodogram_gaia_g_png_directory,
                                                     image_output_pdf=self.periodogram_gaia_g_pdf_directory,
                                                     second_fit_periods=[self.period_g2, self.ztf_g_light_curve.period_fit, self.ztf_r_light_curve.period_fit],
                                                     second_fit_names=["GDR2", "ZTF g", "ZTF r"])

            self.gaia_g_light_curve.draw_frequency_gram(image_output_png=self.output_frequency_periodogram_gaia_g_png,
                                                     image_output_pdf=self.output_frequency_periodogram_gaia_g_pdf,
                                                     second_fit_periods=[self.period_g2, self.ztf_g_light_curve.period_fit, self.ztf_r_light_curve.period_fit],
                                                     second_fit_names=["GDR2", "ZTF g", "ZTF r"])

            self.ztf_g_light_curve.draw_frequency_gram(image_output_png=self.output_frequency_periodogram_ztf_g_png,
                                                     image_output_pdf=self.output_frequency_periodogram_ztf_g_pdf,
                                                     second_fit_periods=[self.period_g2, self.gaia_g_light_curve.period_fit, self.ztf_r_light_curve.period_fit],
                                                     second_fit_names=["GDR2", "Gaia G fit", "ZTF r"])

            self.ztf_r_light_curve.draw_frequency_gram(image_output_png=self.output_frequency_periodogram_ztf_r_png,
                                                    image_output_pdf=self.output_frequency_periodogram_ztf_r_pdf,
                                                    second_fit_periods=[self.period_g2,
                                                                        self.gaia_g_light_curve.period_fit,
                                                                        self.ztf_g_light_curve.period_fit],
                                                    second_fit_names=["GDR2", "Gaia G fit", "ZTF g"])

            if do_multiband_periodogram:
                """self.gaia_g_light_curve.save_image = True
                self.gaia_bp_light_curve.save_image = True
                self.gaia_rp_light_curve.save_image = True
                self.ztf_g_light_curve.save_image = True
                self.ztf_r_light_curve.save_image = True"""

                # Multiband periodogram fits
                multiband_gaia = {"period": peak1_period_multiband_gaia, "phase": 0}
                self.gaia_g_light_curve.draw_folded_light_curve_with_other_function(multiband_gaia, "Multiband Gaia",
                                                                                    image_output_png=self.output_multiband_gaia_fit_gaia_g_png,
                                                                                    fit_phase=True)
                self.gaia_bp_light_curve.draw_folded_light_curve_with_other_function(multiband_gaia, "Multiband Gaia",
                                                                                     image_output_png=self.output_multiband_gaia_fit_gaia_bp_png,
                                                                                     fit_phase=True)
                self.gaia_rp_light_curve.draw_folded_light_curve_with_other_function(multiband_gaia, "Multiband Gaia",
                                                                                     image_output_png=self.output_multiband_gaia_fit_gaia_rp_png,
                                                                                     fit_phase=True)

                multiband_ztf = {"period": peak1_period_multiband_ztf, "phase": 0}
                self.ztf_g_light_curve.draw_folded_light_curve_with_other_function(multiband_ztf, "Multiband ZTF",
                                                                                   image_output_png=self.output_multiband_ztf_fit_ztf_g_png,
                                                                                   fit_phase=True)
                self.ztf_r_light_curve.draw_folded_light_curve_with_other_function(multiband_ztf, "Multiband ZTF",
                                                                                   image_output_png=self.output_multiband_ztf_fit_ztf_r_png,
                                                                                   fit_phase=True)

                multiband_all = {"period": peak1_period_multiband_all, "phase": 0}
                self.gaia_g_light_curve.draw_folded_light_curve_with_other_function(multiband_all, "Multiband Gaia and ZTF",
                                                                                    image_output_png=self.output_multiband_all_fit_gaia_g_png,
                                                                                    fit_phase=True)
                self.gaia_bp_light_curve.draw_folded_light_curve_with_other_function(multiband_all,
                                                                                     "Multiband Gaia and ZTF",
                                                                                     image_output_png=self.output_multiband_all_fit_gaia_bp_png,
                                                                                     fit_phase=True)
                self.gaia_rp_light_curve.draw_folded_light_curve_with_other_function(multiband_all,
                                                                                     "Multiband Gaia and ZTF",
                                                                                     image_output_png=self.output_multiband_all_fit_gaia_rp_png,
                                                                                     fit_phase=True)
                self.ztf_g_light_curve.draw_folded_light_curve_with_other_function(multiband_all, "Multiband Gaia and ZTF",
                                                                                   image_output_png=self.output_multiband_all_fit_ztf_g_png,
                                                                                   fit_phase=True)
                self.ztf_r_light_curve.draw_folded_light_curve_with_other_function(multiband_all, "Multiband Gaia and ZTF",
                                                                                   image_output_png=self.output_multiband_all_fit_ztf_r_png,
                                                                                   fit_phase=True)

        ztf_g_median_ls_power_old_psd, ztf_g_median_ls_power_new_psd, ztf_g_ls_power_arg_max_old_psd, ztf_g_ls_power_arg_max_new_psd = self.get_median_ls_power(
            self.ztf_g_light_curve, normalization='psd', recalculate_old_periodogram=True)
        ztf_r_median_ls_power_old_psd, ztf_r_median_ls_power_new_psd, ztf_r_ls_power_arg_max_old_psd, ztf_r_ls_power_arg_max_new_psd = self.get_median_ls_power(
            self.ztf_r_light_curve, normalization='psd', recalculate_old_periodogram=True)
        ztf_i_median_ls_power_old_psd, ztf_i_median_ls_power_new_psd, ztf_i_ls_power_arg_max_old_psd, ztf_i_ls_power_arg_max_new_psd = self.get_median_ls_power(
            self.ztf_i_light_curve, normalization='psd', recalculate_old_periodogram=True)

        ztf_g_median_ls_power_old_normal, ztf_g_median_ls_power_new_normal, ztf_g_ls_power_arg_max_old_normal, ztf_g_ls_power_arg_max_new_normal = self.get_median_ls_power(
            self.ztf_g_light_curve, normalization='standard', recalculate_old_periodogram=False)
        ztf_r_median_ls_power_old_normal, ztf_r_median_ls_power_new_normal, ztf_r_ls_power_arg_max_old_normal, ztf_r_ls_power_arg_max_new_normal = self.get_median_ls_power(
            self.ztf_r_light_curve, normalization='standard', recalculate_old_periodogram=False)
        ztf_i_median_ls_power_old_normal, ztf_i_median_ls_power_new_normal, ztf_i_ls_power_arg_max_old_normal, ztf_i_ls_power_arg_max_new_normal = self.get_median_ls_power(
            self.ztf_i_light_curve, normalization='standard', recalculate_old_periodogram=False)
        if save_variables:
            save_in_txt_topcat([self.source_id, self.ztf_g_light_curve.period_fit, self.ztf_g_light_curve.data_length,
                                self.ztf_r_light_curve.period_fit, self.ztf_r_light_curve.data_length,
                                self.ztf_i_light_curve.period_fit, self.ztf_i_light_curve.data_length,
                                self.ztf_g_light_curve.max_diff, self.ztf_g_light_curve.percentile_q95_5,
                                self.ztf_g_light_curve.st_dev, self.ztf_g_light_curve.avg_error,
                                self.ztf_r_light_curve.max_diff, self.ztf_r_light_curve.percentile_q95_5,
                                self.ztf_r_light_curve.st_dev, self.ztf_r_light_curve.avg_error,
                                self.ztf_i_light_curve.max_diff, self.ztf_i_light_curve.percentile_q95_5,
                                self.ztf_i_light_curve.st_dev, self.ztf_i_light_curve.avg_error,
                                self.ztf_g_light_curve.fit_function_error, self.ztf_r_light_curve.fit_function_error,
                                self.ztf_i_light_curve.fit_function_error, self.ztf_g_light_curve.fit_result['amp'],
                                self.ztf_r_light_curve.fit_result['amp'], self.ztf_i_light_curve.fit_result['amp'],
                                self.ztf_g_light_curve.mean, self.ztf_r_light_curve.mean,
                                self.ztf_i_light_curve.mean,
                                self.ztf_g_light_curve.skewness, self.ztf_r_light_curve.skewness,
                                self.ztf_i_light_curve.skewness,
                                self.ztf_g_light_curve.total_obs_time,
                                self.ztf_r_light_curve.total_obs_time,
                                self.ztf_i_light_curve.total_obs_time,
                                error_g_with_gaia_g, amp_g_with_gaia_g, nrmse_g_with_gaia_g,
                                error_r_with_gaia_g, amp_r_with_gaia_g, nrmse_r_with_gaia_g,
                                error_g_with_ztf_r, amp_g_with_ztf_r, nrmse_g_with_ztf_r,
                                error_r_with_ztf_g, amp_r_with_ztf_g, nrmse_r_with_ztf_g,
                                peak1_period_ztf_g, peak1_value_ztf_g, peak2_period_ztf_g, peak2_value_ztf_g,
                                peak1_period_ztf_r, peak1_value_ztf_r, peak2_period_ztf_r, peak2_value_ztf_r,
                                peak1_period_multiband_gaia, peak1_value_multiband_gaia, peak2_period_multiband_gaia, peak2_value_multiband_gaia,
                                peak1_period_multiband_ztf, peak1_value_multiband_ztf, peak2_period_multiband_ztf, peak2_value_multiband_ztf,
                                peak1_period_multiband_all, peak1_value_multiband_all, peak2_period_multiband_all, peak2_value_multiband_all,
                                self.ztf_g_light_curve.fit_result["ls_fap"], self.ztf_r_light_curve.fit_result["ls_fap"],
                                ztf_g_median_ls_power_old_psd, self.ztf_g_light_curve.nrmse_fit,
                                ztf_r_median_ls_power_old_psd, self.ztf_r_light_curve.nrmse_fit,
                                ztf_i_median_ls_power_old_psd, self.ztf_i_light_curve.nrmse_fit,
                                median_ls_power_multiband_gaia, median_ls_power_multiband_ztf,
                                median_ls_power_multiband_all, ztf_g_median_ls_power_new_psd,
                                ztf_r_median_ls_power_new_psd, ztf_i_median_ls_power_new_psd,
                                ztf_g_ls_power_arg_max_old_psd, ztf_g_ls_power_arg_max_new_psd,
                                ztf_r_ls_power_arg_max_old_psd, ztf_r_ls_power_arg_max_new_psd,
                                ztf_i_ls_power_arg_max_old_psd, ztf_i_ls_power_arg_max_new_psd,
                                ztf_g_median_ls_power_old_normal, ztf_g_median_ls_power_new_normal,
                                ztf_g_ls_power_arg_max_old_normal, ztf_g_ls_power_arg_max_new_normal,
                                ztf_r_median_ls_power_old_normal, ztf_r_median_ls_power_new_normal,
                                ztf_r_ls_power_arg_max_old_normal, ztf_r_ls_power_arg_max_new_normal,
                                ztf_i_median_ls_power_old_normal, ztf_i_median_ls_power_new_normal,
                                ztf_i_ls_power_arg_max_old_normal, ztf_i_ls_power_arg_max_new_normal
                                ], self.ztf_output_period)

    def analyse_neowise_lightcurves(self, parsed_neowise_table, save_variables, show_variables, save_pictures, show_pictures, manual_period_guess=0):
        from tools import save_in_txt_topcat

        if self.gaia_g_light_curve is None:
            self.lightcurve_fit_and_plot(save_variables=False, save_images=False, show_images=False, print_variables=False)
        if self.ztf_g_light_curve is None:
            self.analyse_ztf_lightcurves(False, False, False, False, False)

        self.neowise_w1_light_curve = LightCurveNeowise(parsed_neowise_table, self.source_id, "w1", "NEOWISE", save_image=save_pictures, show_image=show_pictures)
        self.neowise_w2_light_curve = LightCurveNeowise(parsed_neowise_table, self.source_id, "w2", "NEOWISE", save_image=save_pictures, show_image=show_pictures)

        self.neowise_w1_light_curve.fit_light_curve(show_variables, manual_period_guess=manual_period_guess)
        self.neowise_w2_light_curve.fit_light_curve(show_variables, manual_period_guess=manual_period_guess)

        peak1_period_neowise_w1, peak1_value_neowise_w1, peak2_period_neowise_w1, peak2_value_neowise_w1 = self.neowise_w1_light_curve.draw_periodogram(image_output_png=None,
                                                     image_output_pdf=None,
                                                     second_fit_periods=[self.period_g2, self.gaia_g_light_curve.period_fit, self.ztf_g_light_curve.period_fit, self.ztf_r_light_curve.period_fit],
                                                     second_fit_names=["GDR2", "Gaia G fit", "ZTF g", "ZTF r"], find_peaks=True)

        peak1_period_neowise_w2, peak1_value_neowise_w2, peak2_period_neowise_w2, peak2_value_neowise_w2 = self.neowise_w2_light_curve.draw_periodogram(image_output_png=None,
                                                image_output_pdf=None,
                                                second_fit_periods=[self.period_g2,
                                                                    self.gaia_g_light_curve.period_fit,
                                                                    self.ztf_g_light_curve.period_fit, self.ztf_r_light_curve.period_fit],
                                                second_fit_names=["GDR2", "Gaia G fit", "ZTF g", "ZTF r"], find_peaks=True)

        if show_pictures or save_pictures:
            self.neowise_w1_light_curve.draw_fitted_light_curve(image_output_png=self.ztf_output_pictures_fit_ztf_g)
            self.neowise_w1_light_curve.draw_folded_light_curve(image_output_png=self.ztf_output_pictures_folded_ztf_g)

            self.neowise_w2_light_curve.draw_fitted_light_curve(image_output_png=self.ztf_output_pictures_fit_ztf_r)
            self.neowise_w2_light_curve.draw_folded_light_curve(image_output_png=self.ztf_output_pictures_folded_ztf_r)


            self.neowise_w1_light_curve.draw_frequency_gram(image_output_png=self.output_frequency_periodogram_ztf_g_png,
                                                     image_output_pdf=self.output_frequency_periodogram_ztf_g_pdf,
                                                     second_fit_periods=[self.period_g2, self.gaia_g_light_curve.period_fit, self.ztf_r_light_curve.period_fit],
                                                     second_fit_names=["GDR2", "Gaia G fit", "ZTF r"])

            self.neowise_w2_light_curve.draw_frequency_gram(image_output_png=self.output_frequency_periodogram_ztf_r_png,
                                                    image_output_pdf=self.output_frequency_periodogram_ztf_r_pdf,
                                                    second_fit_periods=[self.period_g2,
                                                                        self.gaia_g_light_curve.period_fit,
                                                                        self.ztf_g_light_curve.period_fit],
                                                    second_fit_names=["GDR2", "Gaia G fit", "ZTF g"])


        """ztf_g_median_ls_power_old_psd, ztf_g_median_ls_power_new_psd, ztf_g_ls_power_arg_max_old_psd, ztf_g_ls_power_arg_max_new_psd = self.get_median_ls_power(
            self.ztf_g_light_curve, normalization='psd', recalculate_old_periodogram=True)
        ztf_r_median_ls_power_old_psd, ztf_r_median_ls_power_new_psd, ztf_r_ls_power_arg_max_old_psd, ztf_r_ls_power_arg_max_new_psd = self.get_median_ls_power(
            self.ztf_r_light_curve, normalization='psd', recalculate_old_periodogram=True)
        ztf_i_median_ls_power_old_psd, ztf_i_median_ls_power_new_psd, ztf_i_ls_power_arg_max_old_psd, ztf_i_ls_power_arg_max_new_psd = self.get_median_ls_power(
            self.ztf_i_light_curve, normalization='psd', recalculate_old_periodogram=True)

        ztf_g_median_ls_power_old_normal, ztf_g_median_ls_power_new_normal, ztf_g_ls_power_arg_max_old_normal, ztf_g_ls_power_arg_max_new_normal = self.get_median_ls_power(
            self.ztf_g_light_curve, normalization='standard', recalculate_old_periodogram=False)
        ztf_r_median_ls_power_old_normal, ztf_r_median_ls_power_new_normal, ztf_r_ls_power_arg_max_old_normal, ztf_r_ls_power_arg_max_new_normal = self.get_median_ls_power(
            self.ztf_r_light_curve, normalization='standard', recalculate_old_periodogram=False)
        ztf_i_median_ls_power_old_normal, ztf_i_median_ls_power_new_normal, ztf_i_ls_power_arg_max_old_normal, ztf_i_ls_power_arg_max_new_normal = self.get_median_ls_power(
            self.ztf_i_light_curve, normalization='standard', recalculate_old_periodogram=False)"""
        """if save_variables:
            save_in_txt_topcat([self.source_id, self.ztf_g_light_curve.period_fit, self.ztf_g_light_curve.data_length,
                                self.ztf_r_light_curve.period_fit, self.ztf_r_light_curve.data_length,
                                self.ztf_i_light_curve.period_fit, self.ztf_i_light_curve.data_length,
                                self.ztf_g_light_curve.max_diff, self.ztf_g_light_curve.percentile_q95_5,
                                self.ztf_g_light_curve.st_dev, self.ztf_g_light_curve.avg_error,
                                self.ztf_r_light_curve.max_diff, self.ztf_r_light_curve.percentile_q95_5,
                                self.ztf_r_light_curve.st_dev, self.ztf_r_light_curve.avg_error,
                                self.ztf_i_light_curve.max_diff, self.ztf_i_light_curve.percentile_q95_5,
                                self.ztf_i_light_curve.st_dev, self.ztf_i_light_curve.avg_error,
                                self.ztf_g_light_curve.fit_function_error, self.ztf_r_light_curve.fit_function_error,
                                self.ztf_i_light_curve.fit_function_error, self.ztf_g_light_curve.fit_result['amp'],
                                self.ztf_r_light_curve.fit_result['amp'], self.ztf_i_light_curve.fit_result['amp'],
                                self.ztf_g_light_curve.mean, self.ztf_r_light_curve.mean,
                                self.ztf_i_light_curve.mean,
                                self.ztf_g_light_curve.skewness, self.ztf_r_light_curve.skewness,
                                self.ztf_i_light_curve.skewness,
                                self.ztf_g_light_curve.total_obs_time,
                                self.ztf_r_light_curve.total_obs_time,
                                self.ztf_i_light_curve.total_obs_time,
                                error_g_with_gaia_g, amp_g_with_gaia_g, nrmse_g_with_gaia_g,
                                error_r_with_gaia_g, amp_r_with_gaia_g, nrmse_r_with_gaia_g,
                                error_g_with_ztf_r, amp_g_with_ztf_r, nrmse_g_with_ztf_r,
                                error_r_with_ztf_g, amp_r_with_ztf_g, nrmse_r_with_ztf_g,
                                peak1_period_ztf_g, peak1_value_ztf_g, peak2_period_ztf_g, peak2_value_ztf_g,
                                peak1_period_ztf_r, peak1_value_ztf_r, peak2_period_ztf_r, peak2_value_ztf_r,
                                peak1_period_multiband_gaia, peak1_value_multiband_gaia, peak2_period_multiband_gaia, peak2_value_multiband_gaia,
                                peak1_period_multiband_ztf, peak1_value_multiband_ztf, peak2_period_multiband_ztf, peak2_value_multiband_ztf,
                                peak1_period_multiband_all, peak1_value_multiband_all, peak2_period_multiband_all, peak2_value_multiband_all,
                                self.ztf_g_light_curve.fit_result["ls_fap"], self.ztf_r_light_curve.fit_result["ls_fap"],
                                ztf_g_median_ls_power_old_psd, self.ztf_g_light_curve.nrmse_fit,
                                ztf_r_median_ls_power_old_psd, self.ztf_r_light_curve.nrmse_fit,
                                ztf_i_median_ls_power_old_psd, self.ztf_i_light_curve.nrmse_fit,
                                median_ls_power_multiband_gaia, median_ls_power_multiband_ztf,
                                median_ls_power_multiband_all, ztf_g_median_ls_power_new_psd,
                                ztf_r_median_ls_power_new_psd, ztf_i_median_ls_power_new_psd,
                                ztf_g_ls_power_arg_max_old_psd, ztf_g_ls_power_arg_max_new_psd,
                                ztf_r_ls_power_arg_max_old_psd, ztf_r_ls_power_arg_max_new_psd,
                                ztf_i_ls_power_arg_max_old_psd, ztf_i_ls_power_arg_max_new_psd,
                                ztf_g_median_ls_power_old_normal, ztf_g_median_ls_power_new_normal,
                                ztf_g_ls_power_arg_max_old_normal, ztf_g_ls_power_arg_max_new_normal,
                                ztf_r_median_ls_power_old_normal, ztf_r_median_ls_power_new_normal,
                                ztf_r_ls_power_arg_max_old_normal, ztf_r_ls_power_arg_max_new_normal,
                                ztf_i_median_ls_power_old_normal, ztf_i_median_ls_power_new_normal,
                                ztf_i_ls_power_arg_max_old_normal, ztf_i_ls_power_arg_max_new_normal
                                ], self.ztf_output_period)"""

    @staticmethod
    def get_median_ls_power(light_curve: LightCurve, normalization, recalculate_old_periodogram):
        if np.size(light_curve.ls_power) > 0:
            if recalculate_old_periodogram:
                _, ls_pow_new = lightcurve_class.calculate_periodogram_powers(light_curve.data_t,
                                                                              light_curve.data_y, light_curve.ls_freqs,
                                                                              normalization=normalization)
                median_ls_power_old = np.median(ls_pow_new)
                ls_power_arg_max_old = np.max(ls_pow_new)
            else:
                median_ls_power_old = np.median(light_curve.ls_power)
                ls_power_arg_max_old = np.max(light_curve.ls_power)
            if light_curve.fitting_successful:
                temp_data_y = light_curve.data_y - light_curve.fit_result[
                    "fitfunc"](light_curve.data_t)
                _, ls_pow_new = lightcurve_class.calculate_periodogram_powers(light_curve.data_t,
                                                                          temp_data_y, light_curve.ls_freqs,
                                                                          normalization=normalization)
                median_ls_power_new = np.median(ls_pow_new)
                ls_power_arg_max_new = np.max(ls_pow_new)
            else:
                median_ls_power_new = cf.dflt_no_vle
                ls_power_arg_max_new = cf.dflt_no_vle
        else:
            median_ls_power_old = cf.dflt_no_vle
            median_ls_power_new = cf.dflt_no_vle
            ls_power_arg_max_old = cf.dflt_no_vle
            ls_power_arg_max_new = cf.dflt_no_vle
        return median_ls_power_old, median_ls_power_new, ls_power_arg_max_old, ls_power_arg_max_new


    def get_closest_nebula(self, nebula_data):
        get_closest_nebula.find_nebula(self, nebula_data, cf.output_textfile_nebulae)

