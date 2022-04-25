import datetime
import matplotlib
import numpy as np

file_category = "all"               # Purely for naming purposes of the outputs. Useful if need different samples
gaia_table_version = 46             # For input Gaia table naming. Can in theory be ignored, check the usage to see where used
save_pdfs = False                   # If want to save images as PDFs, increases the run time quite a bit
debug_mode = True                   # Debug mode: show parameters and images, do NOT save anything
save_images_if_not_debug = False    # If want to save images, while not debugging
use_vizier_data = False             # If want to also plot data from Vizier

lightcurve_fit = False       # Do light curve fit for Gaia data
ztf_lightcurve_fit = False    # Do light curve fit for ZTF curves
do_multiband_periodogram = False

neowise_lightcurve_fit = False

sed_line_fit = True        # Do SED linear fits (i.e. IR slopes); fast, nothing complex
do_pdf_creation = False     # Make PDFs for each star with their information and fits
do_bb_fit = False           # Do fits using my own blackbody fits (bad)
save_bb_variables = False   # Save variables for blackbody fit
do_sedfitter = False        # Use SEDFitter to fit star's models. Only on fast PC
find_nebula = False

do_irsa_astroquery = False
do_xmm_astroquery = False
do_vvvdr4_astroquery = False
recalculate_sedfitter_param_ranges = False
recalculate_folder = "sedfitter/fitinfo/spusmi/" #"sedfitter/fitinfo/spsi/"

sed_fits_to_do = "0001001"  # Which fits to do in sed_line_fit. 1 if want to do, 0 if not.
#sed_fits_to_do = "0000001"

remove_ztf_peak_periodogram_days = []   # 1

# Explanation of each bit of "sed_fits_to_do":
# 1st bit: line fit
# 2nd bit: sed excess
# 3rd bit: sed mir excess
# 4th bit: ir slope 25
# 5th bit: ir slope 20
# 6th bit: integral of the excess
# 7th bit: plot image

ids_to_calculate = np.array([]).astype(str)

detrend = False
do_fit_with_constant_period = False
period_manual = 0  # 9.353221424477987*2

if not debug_mode:      # Save images, variables, do not show anything
    matplotlib.use('Agg')
    lightcurve_fit_show_images = False
    lightcurve_fit_save_images = True
    lightcurve_fit_print_variables = False
    lightcurve_fit_save_variables = True

    ztf_save_variables = True
    ztf_show_variables = False
    ztf_save_pictures = True
    ztf_show_pictures = False

    sed_line_fit_show_images = False
    sed_line_fit_save_images = True
    sed_line_fit_print_variables = False
    sed_line_fit_save_variables = True

    bb_fit_show_images = False
    bb_fit_save_images = False
else:       # Show variables images, do not save anything
    lightcurve_fit_show_images = True
    lightcurve_fit_save_images = False
    lightcurve_fit_print_variables = True
    lightcurve_fit_save_variables = False

    ztf_save_variables = False
    ztf_show_variables = True
    ztf_save_pictures = False
    ztf_show_pictures = True

    sed_line_fit_show_images = True
    sed_line_fit_save_images = False
    sed_line_fit_print_variables = True
    sed_line_fit_save_variables = False

    bb_fit_show_images = True
    bb_fit_save_images = False


if not save_images_if_not_debug and not debug_mode:     # If one does not want to save images regardless
    lightcurve_fit_save_images = False
    sed_line_fit_save_images = False
    bb_fit_save_images = False
    ztf_save_pictures = False


# Input Gaia table (includes the gaia_table_version from above)

input_gaia_table_path = f"../../Unige/2021-2022_Master_project/Data/Processed_tables/ge3_yso_pot_10_pp_g2_ge3_" \
                        f"simbad_v{gaia_table_version}.fits"

# Config for SED linear fit:

sed_linear_line_wavelength_start = 2.0 * np.power(10.0, -6)     # Line fit of slope below, where to start & end
sed_linear_line_wavelength_end = 4.0 * np.power(10.0, -6)

line_fit_power_rayleigh_jeans = -3.0    # Slope at big wavelength. -3 is standard Rayleigh Jeans, but can use different value

ir_slope_start = 2.2 * pow(10, -6)      # Where want to do the IR slopes
ir_slope_end = 25.0 * pow(10, -6)
ir_slope_end2 = 20.0 * pow(10, -6)

ir_excess_start = 2.2 * pow(10, -6)     # Done as an indication of where fits are usually done, generally
ir_excess_end = 25.0 * pow(10, -6)      # should not be changed from these values

# Classification of YSO objects based on starting slope values:
yso_classes = [["Class_1", "0.0"], ["Class_2", "-1.5"], ["Class_3", "-20.0"], ["Non_YSO", "-20.5"]]
#2D array. Each component: first name, then at what slope the classification starts (it ends wherever next class starts)

# Config for light curves

minimum_data_points_to_fit_light_curve = 10

# Input for SED Vizier data

input_sed_files_to_analyse_path = f"sed/{file_category}/"

# Input for Gaia light curves

input_light_curve_to_analyse_path = f"xml_files/{file_category}/"

# Output for files

date_time_to_use = str(datetime.datetime.now()).replace(':', '_')

textfiles_output_folder = "output_textfiles/"
output_text_any_param = f"{textfiles_output_folder}{date_time_to_use}_"
output_pdf_files = f"output_pdf/{file_category}/"

# Output files for light curves:

lightcurve_gaia_txt_to_save = "#source_id_1\tperiod_gaia_g\tmax_amp_gaia_g\tq95_m_q5_gaia_g\tst_dev_gaia_g" \
                              "\tavg_err_gaia_g\tfit_error_gaia_g\tamp_fit_gaia_g\tpoints_gaia_g\tskewness_gaia_g" \
                              "\ttotal_obs_time_gaia_g\tpoints_gaia_bp\tpoints_gaia_rp\tgaia_g_bp_corr\tgaia_g_rp_corr" \
                              "\terr_fit_using_gaia_g__gaia_bp\tamp_fit_using_gaia_g__gaia_bp\tnrmse_fit_using_gaia_g__gaia_bp" \
                              "\terr_fit_using_gaia_g__gaia_rp\tamp_fit_using_gaia_g__gaia_rp\tnrmse_fit_using_gaia_g__gaia_rp" \
                              "\tperiodogram_gaia_g_peak1_period\tperiodogram_gaia_g_peak1_value" \
                              "\tperiodogram_gaia_g_peak2_period\tperiodogram_gaia_g_peak2_value\tgaia_g_ls_fap" \
                              "\tgaia_g_median_ls_powers_old\tgaia_g_fit_nrmse\tgaia_g_median_ls_powers_old_psd" \
                              "\tgaia_g_median_ls_powers_new_psd\tgaia_g_ls_peak_old_psd\tgaia_g_ls_peak_new_psd" \
                              "\tgaia_g_median_ls_powers_old_normal\tgaia_g_median_ls_powers_new_normal" \
                              "\tgaia_g_ls_peak_old_normal\tgaia_g_ls_peak_new_normal"

output_fitted_param = f"{textfiles_output_folder}{date_time_to_use}_{file_category}_fitted_param.txt"

output_periodogram_gaia_g_png = f"output_pictures/{file_category}/periodogram_gaia_g_png/"
output_periodogram_gaia_g_pdf = f"output_pictures/{file_category}/periodogram_gaia_g_pdf/"

output_frequency_periodogram_gaia_g_png = f"output_pictures/{file_category}/frequency_periodogram_gaia_g_png/"
output_frequency_periodogram_gaia_g_pdf = f"output_pictures/{file_category}/frequency_periodogram_gaia_g_pdf/"

output_light_curve_g_band_png = f"output_pictures/{file_category}/lightcurves_gaia_g_png/"
output_light_curve_g_band_pdf = f"output_pictures/{file_category}/lightcurves_gaia_g_pdf/"

output_raw_data_g_band_png = f"output_pictures/{file_category}/rawdata_gaia_g_png/"
output_raw_data_g_band_pdf = f"output_pictures/{file_category}/rawdata_gaia_g_pdf/"

output_folded_light_curve_png = f"output_pictures/{file_category}/folded_lightcurves_gaia_g_png/"
output_folded_light_curve_pdf = f"output_pictures/{file_category}/folded_lightcurves_gaia_g_pdf/"

output_rp_raw_data_png = f"output_pictures/{file_category}/rawdata_RP_png/"
output_rp_raw_data_pdf = f"output_pictures/{file_category}/rawdata_RP_pdf/"

output_bp_raw_data_png = f"output_pictures/{file_category}/rawdata_BP_png/"
output_bp_raw_data_pdf = f"output_pictures/{file_category}/rawdata_BP_pdf/"

output_gaia_all_bands_raw_data_png = f"output_pictures/{file_category}/all_gaia_rawdata_png/"

output_fitted_period_text = f"{textfiles_output_folder}{date_time_to_use}_{file_category}_periods.txt"

# ZTF light curves

ztf_lightcurve_txt_to_save = "#source_id_123\tperiod_ztf_g\tpoints_ztf_g\tperiod_ztf_r\tpoints_ztf_r" \
                             "\tperiod_ztf_i\tpoints_ztf_i" \
                             "\tmax_amp_ztf_g\tq95_m_q5_ztf_g\tst_dev_ztf_g\tavg_err_ztf_g" \
                             "\tmax_amp_ztf_r\tq95_m_q5_ztf_r\tst_dev_ztf_r\tavg_err_ztf_r" \
                             "\tmax_amp_ztf_i\tq95_m_q5_ztf_i\tst_dev_ztf_i\tavg_err_ztf_i" \
                             "\tfit_error_ztf_g\tfit_error_ztf_r\tfit_error_ztf_i" \
                             "\tamp_fit_ztf_g\tamp_fit_ztf_r\tamp_fit_ztf_i" \
                             "\tband_mean_ztf_g\tband_mean_ztf_r\tband_mean_ztf_i" \
                             "\tskewness_ztf_g\tskewness_ztf_r\tskewness_ztf_i" \
                             "\ttotal_obs_time_ztf_g\ttotal_obs_time_ztf_r\ttotal_obs_time_ztf_i," \
                             "\terr_fit_using_gaia_g__ztf_g\tamp_fit_using_gaia_g__ztf_g\tnrmse_fit_using_gaia_g__ztf_g" \
                             "\terr_fit_using_gaia_g__ztf_r\tamp_fit_using_gaia_g__ztf_r\tnrmse_fit_using_gaia_g__ztf_r" \
                             "\terr_fit_using_ztf_r__ztf_g\tamp_fit_using_ztf_r__ztf_g\tnrmse_fit_using_ztf_r__ztf_g" \
                             "\terr_fit_using_ztf_g__ztf_r\tamp_fit_using_ztf_g__ztf_r\tnrmse_fit_using_ztf_g__ztf_r" \
                             "\tperiodogram_ztf_g_peak1_period\tperiodogram_ztf_g_peak1_value" \
                             "\tperiodogram_ztf_g_peak2_period\tperiodogram_ztf_g_peak2_value" \
                             "\tperiodogram_ztf_r_peak1_period\tperiodogram_ztf_r_peak1_value" \
                             "\tperiodogram_ztf_r_peak2_period\tperiodogram_ztf_r_peak2_value" \
                             "\tperiodogram_multiband_gaia_peak1_period\tperiodogram_multiband_gaia_peak1_value" \
                             "\tperiodogram_multiband_gaia_peak2_period\tperiodogram_multiband_gaia_peak2_value" \
                             "\tperiodogram_multiband_ztf_peak1_period\tperiodogram_multiband_ztf_peak1_value" \
                             "\tperiodogram_multiband_ztf_peak2_period\tperiodogram_multiband_ztf_peak2_value" \
                             "\tperiodogram_multiband_all_peak1_period\tperiodogram_multiband_all_peak1_value" \
                             "\tperiodogram_multiband_all_peak2_period\tperiodogram_multiband_all_peak2_value" \
                             "\tztf_g_ls_fap\tztf_r_ls_fap\tztf_g_median_ls_powers_old_psd\tztf_g_fit_nrmse" \
                             "\tztf_r_median_ls_powers_old_psd\tztf_r_fit_nrmse\tztf_i_median_ls_powers_old_psd\tztf_i_fit_nrmse" \
                             "\tperiodogram_multiband_gaia_median_ls_power" \
                             "\tperiodogram_multiband_ztf_median_ls_power\tperiodogram_multiband_all_median_ls_power" \
                             "\tztf_g_median_ls_powers_new_psd\tztf_r_median_ls_powers_new_psd\tztf_i_median_ls_powers_new_psd" \
                             "\tztf_g_ls_peak_old_psd\tztf_g_ls_peak_new_psd\tztf_r_ls_peak_old_psd\tztf_r_ls_peak_new_psd" \
                             "\tztf_i_ls_peak_old_psd\tztf_i_ls_peak_new_psd" \
                             "\tztf_g_median_ls_powers_old_normal\tztf_g_median_ls_powers_new_normal" \
                             "\tztf_g_ls_peak_old_normal\tztf_g_ls_peak_new_normal" \
                             "\tztf_r_median_ls_powers_old_normal\tztf_r_median_ls_powers_new_normal" \
                             "\tztf_r_ls_peak_old_normal\tztf_r_ls_peak_new_normal" \
                             "\tztf_i_median_ls_powers_old_normal\tztf_i_median_ls_powers_new_normal" \
                             "\tztf_i_ls_peak_old_normal\tztf_i_ls_peak_new_normal"


input_ztf_lightcurves_to_analyse = f"input_lightcurves/{file_category}/ztf/"

neowise_light_curve_file_path = f"input_lightcurves/{file_category}/neowise_xmatch.vot"

output_ztf_lightcurves_period = f"{textfiles_output_folder}{date_time_to_use}_{file_category}_periods_ztf.txt"
output_ztf_lightcurves_pictures = f"output_pictures/{file_category}/ztf_lightcurves/"

output_ztf_fit = f"output_pictures/{file_category}/ztf_lightcurves_fit/"
output_ztf_folded = f"output_pictures/{file_category}/ztf_lightcurves_folded/"

output_periodogram_ztf_g_png = f"output_pictures/{file_category}/periodogram_ztf_g_png/"
output_periodogram_ztf_g_pdf = f"output_pictures/{file_category}/periodogram_ztf_g_pdf/"

output_periodogram_ztf_r_png = f"output_pictures/{file_category}/periodogram_ztf_r_png/"
output_periodogram_ztf_r_pdf = f"output_pictures/{file_category}/periodogram_ztf_r_pdf/"

output_frequency_periodogram_ztf_g_png = f"output_pictures/{file_category}/frequency_periodogram_ztf_g_png/"
output_frequency_periodogram_ztf_g_pdf = f"output_pictures/{file_category}/frequency_periodogram_ztf_g_pdf/"

output_frequency_periodogram_ztf_r_png = f"output_pictures/{file_category}/frequency_periodogram_ztf_r_png/"
output_frequency_periodogram_ztf_r_pdf = f"output_pictures/{file_category}/frequency_periodogram_ztf_r_pdf/"

output_multiband_frequency_periodogram_png = f"output_pictures/{file_category}/frequency_periodogram_multiband/"

output_multiband_fits_png = f"output_pictures/{file_category}/multiband_fits/"

# Output files for SED line fit and BB fit:

output_sed_fit_png = f"output_pictures/{file_category}/sed_png/"
output_sed_fit_pdf = f"output_pictures/{file_category}/sed_pdf/"

output_sed_bar_png = f"output_pictures/{file_category}/sed_bar_png/"
output_sed_bar_pdf = f"output_pictures/{file_category}/sed_bar_pdf/"

output_sed_ir_slope_fit_png = f"output_pictures/{file_category}/sed_ir_slope_png/"
output_sed_ir_slope_fit_pdf = f"output_pictures/{file_category}/sed_ir_slope_pdf/"

output_sed_integrated_excess_figure_png = f"output_pictures/{file_category}/sed_int_excess_png/"
output_sed_integrated_excess_figure_pdf = f"output_pictures/{file_category}/sed_int_excess_pdf/"

output_sed_temp_fits_txt = f"{textfiles_output_folder}{date_time_to_use}__sed_temp.txt"

output_sed_temp_pic = f"output_pictures/{file_category}/sed_temps/"

foundation_txt = "#source_id126\tmir_exists"
all_ratio_txt = "\tlast_wv\tlast_ratio_err\tlast_ratio\tavg_diff\tavg_ratio\t" \
                        "biggest_ratio_err\tbiggest_ratio_wl\tbiggest_ratio"
mir_ratio_txt = "\tlast_wv_mir\tlast_ratio_err_mir\tlast_ratio_mir\tavg_diff_mir\tavg_ratio_mir\t" \
                    "biggest_ratio_err_mir\tbiggest_ratio_wl_mir\tbiggest_ratio_mir"
ir_slope_txt = "\tslope_25\tslope_error_lmfit_25\tslope_error_mine_25\tconstant_25\tslope_25_class"
ir_slope_txt2 = "\tslope_20\tslope_error_lmfit_20\tslope_error_mine_20\tconstant_20\tslope_20_class"
integrated_excess_txt = "\tsed_lin_int\tsed_lin_int_err\tray_lin_int\tray_lin_int_err\tsed_log_int\tsed_log_int_err\t" \
                        "ray_log_int\tray_log_int_err"
other_txts_for_sed = [all_ratio_txt, mir_ratio_txt, ir_slope_txt, ir_slope_txt2, integrated_excess_txt]

bb_fit_txt_to_save = "#source_id_12\tstar_temp\tdisk_temp\tscale_factor_disk\tradius"

# SEDFitter parameters:

sedfitter_pc_separation = 1    # Separation of stars to do fits on
sedfitter_upper_limit_confident = 0.3   # Expected 0 to 1. 1 is full confidence, 0 is treating upper limit as if not existant

sedfitter_file_data = f"output_textfiles/sed/{file_category}{date_time_to_use}_for_sedfitter"
sedfitter_output_parameters = f"sedfitter/parameters/{date_time_to_use}_{file_category}_"
sedfitter_output_parameters_folder = 'sedfitter/parameters/'
sedfitter_filters_total = ["GDR2_BP", "GEDR3_BP", "GEDR3_G", "GDR2_G", "GEDR3_RP", "GDR2_RP", "2J", "2H", "2K", "WISE1",
                           "I1", "I2", "WISE2", "I3", "I4", "WISE3", "WISE4", "M1", "PACS1", "M2", "PACS2"]
sedfitter_filters_total = ["GEDR3_BP", "GEDR3_G", "GEDR3_RP", "2J", "2H", "2K", "WISE1",
                           "I1", "I2", "WISE2", "I3", "I4", "WISE3", "WISE4", "M1", "PACS1", "M2", "PACS2"]
#sedfitter_filters_total = ["GEDR3_BP", "GEDR3_G", "GEDR3_RP", "2J", "2H", "2K", "WISE1", "I1"]
sedfitter_models_dir = 'models_kurucz'

#sedfitter_filters_total = ["2J", "2H", "2K", "WISE1", "I1", "I2", "WISE2", "I3", "I4", "WISE3", "WISE4", "M1", "PACS1", "M2", "PACS2"]
#sedfitter_filters_total = ['2J', '2H', '2K', 'I1', 'I2', 'I3', 'I4', 'M1', 'M2']
sedfitter_models_dir = 'models_r06'
#sedfitter_models_dir = 'spubhmi'
sedfitter_models_dir = 'sp--s-i'
#sedfitter_models_dir = 's-p-smi'
#sedfitter_models_dir = 's-p-hmi'
#sedfitter_models_dir = 'sp--smi'
#sedfitter_models_dir = 's---s-i'
#sedfitter_models_dir = 's---smi'

sedfitter_models_dir = 'spu-smi'

# SED Catalogues

xmatched_new_catalogues_directory = "xmatched_catalogues/"
temp_path = "temp/"

# Nebulae data

input_folder_nebulae = "mol_clouds/"
output_textfile_nebulae = f"output_textfiles/{date_time_to_use}_{file_category}_nebulae_data.txt"
output_textfile_nebulae_star_pm = f"output_textfiles/{date_time_to_use}_{file_category}_nebulae_with_star_pm.txt"
output_textfile_nebulae_only_pm = f"output_textfiles/{date_time_to_use}_{file_category}_nebulae_only_pm.txt"
input_pm_nebulae = "mol_clouds/pm_nebula/mol_clouds_gaianew_v2.txt"

# Wavelength in meters
irac1 = 3.55 * pow(10, -6)
irac2 = 4.493 * pow(10, -6)
irac3 = 5.731 * pow(10, -6)
irac4 = 7.872 * pow(10, -6)
mips1 = 23.67 * pow(10, -6)
mips2 = 71.42 * pow(10, -6)
mips3 = 155.9 * pow(10, -6)
iras12 = 11.59 * pow(10, -6)
irs16 = 16.0 * pow(10, -6)
irs22 = 22.0 * pow(10, -6)
iras25 = 23.88 * pow(10, -6)
pacs70 = 70.0 * pow(10, -6)
pacs100 = 100.0 * pow(10, -6)
pacs160 = 160.0 * pow(10, -6)
spire250 = 250.0 * pow(10, -6)
spire350 = 363.0 * pow(10, -6)
spire500 = 517.0 * pow(10, -6)
j_2mass = 1.235 * pow(10, -6)
h_2mass = 1.662 * pow(10, -6)
k_2mass = 2.159 * pow(10, -6)
W1 = 3.3526 * pow(10, -6)
W2 = 4.6028 * pow(10, -6)
W3 = 11.5608 * pow(10, -6)
W4 = 22.0883 * pow(10, -6)

bp_dr2 = 0.50515 * pow(10, -6)    # DR2 Gaia
g_dr2 = 0.62306 * pow(10, -6)
rp_dr2 = 0.77776 * pow(10, -6)

bp_edr3 = 0.51097 * pow(10, -6)  # EDR3 Gaia
g_edr3 = 0.62179 * pow(10, -6)
rp_edr3 = 0.77691 * pow(10, -6)

# F0 of the filters in Jy
W1_F0 = 306.682 / 1.0084    # assume rayleigh jeans regime for WISE/ALLWISE (correction coefficients assumption)
W2_F0 = 170.663 / 1.0066    # In reality, taking into account would change result by significantly less than 1%
W3_F0 = 29.045 / 1.0088     # Because of conversion to log log graph
W4_F0 = 8.284 / 1.0013
j_2mass_F0 = 1594.0
h_2mass_F0 = 1024.0
k_2mass_F0 = 666.7
irac1_F0 = 280.9
irac2_F0 = 179.7
irac3_F0 = 115.0
irac4_F0 = 64.13
mips1_F0 = 7.14
mips2_F0 = 0.775
mips3_F0 = 0.159
iras12_F0 = 28.30
iras25_F0 = 6.730

bp_dr2_F0 = 3535.0    # DR2 Gaia
g_dr2_F0 = 3296.0
rp_dr2_F0 = 2620.0

bp_edr3_F0 = 3552.0  # EDR3 Gaia
g_edr3_F0 = 3229.0
rp_edr3_F0 = 2555.0


# taken out of the extra catalogues:
# "IRS_2_arc", "SPIRE_250_17_6_arc", "SPIRE_350_23_9_arc", "SPIRE_500_35_2_arc", "PACS_160_10_7_arc"
extra_catalogues_dir = {
    "table_names": ["WISE_2_arc", "ALLWISE_2_arc", "SEIP_2_arc", "C2D_reliable_stars_2_arc", "C2D_cloud_2_arc",
                    "CSI_2264_2_arc",
                    "Cygnus_X_cat_2_arc", "GLIMPSE_I_2_arc", "GLIMPSE_II_2_arc", "GLIMPSE_II_Epoch_1_2_arc",
                    "GLIMPSE_II_Epoch_2_2_arc", "GLIMPSE_3D_2_arc", "GLIMPSE_3D_Epoch_1_2_arc",
                    "GLIMPSE_3D_Epoch_2_2_arc", "GLIMPSE_Vela_2_arc", "GLIMPSE_SMOG_2_arc",
                    "GLIMPSE_Cygnus_2_arc", "MIPSGAL_2_arc", "Taurus_2_arc", "YSO_GGD_2_arc",
                    "YSO_L1688_2_arc", "PACS_70_5_6_arc", "PACS_100_6_8_arc",
                    "twomass_2_arc", "gaia_dr2_edr3"],

    "WISE_2_arc": "../../Unige/2021-2022_Master_project/Data/x_matches/WISE_2_arc_.fits",
    "WISE_2_arc_flux": ["w1mpro", "w2mpro", "w3mpro", "w4mpro"],
    "WISE_2_arc_err": ["w1sigmpro", "w2sigmpro", "w3sigmpro", "w4sigmpro"],
    "WISE_2_arc_flag": ["ph_qual"],
    "WISE_2_arc_wave": [W1, W2, W3, W4],
    "WISE_2_arc_pref": 1,
    "WISE_2_arc_flux_unit": False,

    "ALLWISE_2_arc": "../../Unige/2021-2022_Master_project/Data/x_matches/ALLWISE_2_arc_.fits",
    "ALLWISE_2_arc_flux": ["w1mpro", "w2mpro", "w3mpro", "w4mpro"],
    "ALLWISE_2_arc_err": ["w1sigmpro", "w2sigmpro", "w3sigmpro", "w4sigmpro"],
    "ALLWISE_2_arc_flag": ["ph_qual"],
    "ALLWISE_2_arc_wave": [W1, W2, W3, W4],
    "ALLWISE_2_arc_pref": 1,
    "ALLWISE_2_arc_flux_unit": False,

    "SEIP_2_arc": "../../Unige/2021-2022_Master_project/Data/x_matches/SEIP_2_arc.fits",
    "SEIP_2_arc_flux": ["i1_f_ap1", "i2_f_ap1", "i3_f_ap1", "i4_f_ap1", "m1_f_ap"],
    "SEIP_2_arc_err": ["i1_df_ap1", "i2_df_ap1", "i3_df_ap1", "i4_df_ap1", "m1_df_ap"],
    "SEIP_2_arc_flag": ["i1_fluxtype", "i2_fluxtype", "i3_fluxtype", "i4_fluxtype", "m1_fluxtype"],
    "SEIP_2_arc_wave": [irac1, irac2, irac3, irac4, mips1],
    "SEIP_2_arc_pref": pow(10, -6),
    "SEIP_2_arc_flux_unit": True,

    "IRS_2_arc": "../../Unige/2021-2022_Master_project/Data/x_matches/IRS_2_arc.fits",
    "IRS_2_arc_flux": ["irac8", "iras12", "irs16", "irs22", "mips24", "iras25"],
    "IRS_2_arc_err": ["irac8u", "iras12u", "irs16u", "irs22u", "mips24u", "iras25u"],
    "IRS_2_arc_flag": [],
    "IRS_2_arc_wave": [irac4, iras12, irs16, irs22, mips1, iras25],
    "IRS_2_arc_pref": 1,
    "IRS_2_arc_flux_unit": True,

    "C2D_reliable_stars_2_arc": "../../Unige/2021-2022_Master_project/Data/x_matches/C2D_reliable_stars_2_arc.fits",
    "C2D_reliable_stars_2_arc_flux": ["ir1_flux_c", "ir2_flux_c", "ir3_flux_c", "ir4_flux_c", "mp1_flux_c",
                                      "mp2_flux_c"],
    "C2D_reliable_stars_2_arc_err": ["ir1_d_flux_c", "ir2_d_flux_c", "ir3_d_flux_c", "ir4_d_flux_c", "mp1_d_flux_c",
                                     "mp2_d_flux_c"],
    "C2D_reliable_stars_2_arc_flag": ["ir1_q_det_c", "ir2_q_det_c", "ir3_q_det_c", "ir4_q_det_c", "mp1_q_det_c",
                                      "mp2_q_det_c"],
    "C2D_reliable_stars_2_arc_wave": [irac1, irac2, irac3, irac4, mips1, mips2],
    "C2D_reliable_stars_2_arc_pref": pow(10, -3),
    "C2D_reliable_stars_2_arc_flux_unit": True,

    "C2D_cloud_2_arc": "../../Unige/2021-2022_Master_project/Data/x_matches/C2D_cloud_2_arc.fits",
    "C2D_cloud_2_arc_flux": ["ir1_flux_c", "ir2_flux_c", "ir3_flux_c", "ir4_flux_c", "mp1_flux_c", "mp2_flux_c"],
    "C2D_cloud_2_arc_err": ["ir1_d_flux_c", "ir2_d_flux_c", "ir3_d_flux_c", "ir4_d_flux_c", "mp1_d_flux_c",
                            "mp2_d_flux_c"],
    "C2D_cloud_2_arc_flag": ["ir1_q_det_c", "ir2_q_det_c", "ir3_q_det_c", "ir4_q_det_c", "mp1_q_det_c", "mp2_q_det_c"],
    "C2D_cloud_2_arc_wave": [irac1, irac2, irac3, irac4, mips1, mips2],
    "C2D_cloud_2_arc_pref": pow(10, -3),
    "C2D_cloud_2_arc_flux_unit": True,

    "CSI_2264_2_arc": "../../Unige/2021-2022_Master_project/Data/x_matches/CSI_2264_2_arc.fits",
    "CSI_2264_2_arc_flux": ["mag_36", "mag_45", "mag_58", "mag_80", "mag_24"],
    "CSI_2264_2_arc_err": ["err_36", "err_45", "err_58", "err_80", "err_24"],
    "CSI_2264_2_arc_flag": [],
    "CSI_2264_2_arc_wave": [irac1, irac2, irac3, irac4, mips1],
    "CSI_2264_2_arc_pref": 1,
    "CSI_2264_2_arc_flux_unit": False,

    "Cygnus_X_cat_2_arc": "../../Unige/2021-2022_Master_project/Data/x_matches/Cygnus_X_cat_2_arc.fits",
    "Cygnus_X_cat_2_arc_flux": ["i1flux", "i2flux", "i3flux", "i4flux", "m24flux"],
    "Cygnus_X_cat_2_arc_err": ["i1ferr", "i2ferr", "i3ferr", "i4ferr", "m24ferr"],
    "Cygnus_X_cat_2_arc_flag": [],
    "Cygnus_X_cat_2_arc_wave": [irac1, irac2, irac3, irac4, mips1],
    "Cygnus_X_cat_2_arc_pref": pow(10, -6),
    "Cygnus_X_cat_2_arc_flux_unit": True,

    "GLIMPSE_I_2_arc": "../../Unige/2021-2022_Master_project/Data/x_matches/GLIMPSE_I_2_arc.fits",
    "GLIMPSE_I_2_arc_flux": ["f3_6", "f4_5", "f5_8", "f8_0"],
    "GLIMPSE_I_2_arc_err": ["df3_6", "df4_5", "df5_8", "df8_0"],
    "GLIMPSE_I_2_arc_flag": [],
    "GLIMPSE_I_2_arc_wave": [irac1, irac2, irac3, irac4],
    "GLIMPSE_I_2_arc_pref": pow(10, -3),
    "GLIMPSE_I_2_arc_flux_unit": True,

    "GLIMPSE_II_2_arc": "../../Unige/2021-2022_Master_project/Data/x_matches/GLIMPSE_II_2_arc.fits",
    "GLIMPSE_II_2_arc_flux": ["f3_6", "f4_5", "f5_8", "f8_0"],
    "GLIMPSE_II_2_arc_err": ["df3_6", "df4_5", "df5_8", "df8_0"],
    "GLIMPSE_II_2_arc_flag": [],
    "GLIMPSE_II_2_arc_wave": [irac1, irac2, irac3, irac4],
    "GLIMPSE_II_2_arc_pref": pow(10, -3),
    "GLIMPSE_II_2_arc_flux_unit": True,

    "GLIMPSE_II_Epoch_1_2_arc": "../../Unige/2021-2022_Master_project/Data/x_matches/GLIMPSE_II_Epoch_1_2_arc.fits",
    "GLIMPSE_II_Epoch_1_2_arc_flux": ["f3_6", "f4_5", "f5_8", "f8_0"],
    "GLIMPSE_II_Epoch_1_2_arc_err": ["df3_6", "df4_5", "df5_8", "df8_0"],
    "GLIMPSE_II_Epoch_1_2_arc_flag": [],
    "GLIMPSE_II_Epoch_1_2_arc_wave": [irac1, irac2, irac3, irac4],
    "GLIMPSE_II_Epoch_1_2_arc_pref": pow(10, -3),
    "GLIMPSE_II_Epoch_1_2_arc_flux_unit": True,

    "GLIMPSE_II_Epoch_2_2_arc": "../../Unige/2021-2022_Master_project/Data/x_matches/GLIMPSE_II_Epoch_2_2_arc.fits",
    "GLIMPSE_II_Epoch_2_2_arc_flux": ["f3_6", "f4_5", "f5_8", "f8_0"],
    "GLIMPSE_II_Epoch_2_2_arc_err": ["df3_6", "df4_5", "df5_8", "df8_0"],
    "GLIMPSE_II_Epoch_2_2_arc_flag": [],
    "GLIMPSE_II_Epoch_2_2_arc_wave": [irac1, irac2, irac3, irac4],
    "GLIMPSE_II_Epoch_2_2_arc_pref": pow(10, -3),
    "GLIMPSE_II_Epoch_2_2_arc_flux_unit": True,

    "GLIMPSE_3D_2_arc": "../../Unige/2021-2022_Master_project/Data/x_matches/GLIMPSE_3D_2_arc.fits",
    "GLIMPSE_3D_2_arc_flux": ["f3_6", "f4_5", "f5_8", "f8_0"],
    "GLIMPSE_3D_2_arc_err": ["df3_6", "df4_5", "df5_8", "df8_0"],
    "GLIMPSE_3D_2_arc_flag": [],
    "GLIMPSE_3D_2_arc_wave": [irac1, irac2, irac3, irac4],
    "GLIMPSE_3D_2_arc_pref": pow(10, -3),
    "GLIMPSE_3D_2_arc_flux_unit": True,

    "GLIMPSE_3D_Epoch_1_2_arc": "../../Unige/2021-2022_Master_project/Data/x_matches/GLIMPSE_3D_Epoch_1_2_arc.fits",
    "GLIMPSE_3D_Epoch_1_2_arc_flux": ["f3_6", "f4_5", "f5_8", "f8_0"],
    "GLIMPSE_3D_Epoch_1_2_arc_err": ["df3_6", "df4_5", "df5_8", "df8_0"],
    "GLIMPSE_3D_Epoch_1_2_arc_flag": [],
    "GLIMPSE_3D_Epoch_1_2_arc_wave": [irac1, irac2, irac3, irac4],
    "GLIMPSE_3D_Epoch_1_2_arc_pref": pow(10, -3),
    "GLIMPSE_3D_Epoch_1_2_arc_flux_unit": True,

    "GLIMPSE_3D_Epoch_2_2_arc": "../../Unige/2021-2022_Master_project/Data/x_matches/GLIMPSE_3D_Epoch_2_2_arc.fits",
    "GLIMPSE_3D_Epoch_2_2_arc_flux": ["f3_6", "f4_5", "f5_8", "f8_0"],
    "GLIMPSE_3D_Epoch_2_2_arc_err": ["df3_6", "df4_5", "df5_8", "df8_0"],
    "GLIMPSE_3D_Epoch_2_2_arc_flag": [],
    "GLIMPSE_3D_Epoch_2_2_arc_wave": [irac1, irac2, irac3, irac4],
    "GLIMPSE_3D_Epoch_2_2_arc_pref": pow(10, -3),
    "GLIMPSE_3D_Epoch_2_2_arc_flux_unit": True,

    "GLIMPSE_Vela_2_arc": "../../Unige/2021-2022_Master_project/Data/x_matches/GLIMPSE_Vela_2_arc.fits",
    "GLIMPSE_Vela_2_arc_flux": ["f3_6", "f4_5", "f5_8", "f8_0"],
    "GLIMPSE_Vela_2_arc_err": ["df3_6", "df4_5", "df5_8", "df8_0"],
    "GLIMPSE_Vela_2_arc_flag": [],
    "GLIMPSE_Vela_2_arc_wave": [irac1, irac2, irac3, irac4],
    "GLIMPSE_Vela_2_arc_pref": pow(10, -3),
    "GLIMPSE_Vela_2_arc_flux_unit": True,

    "GLIMPSE_SMOG_2_arc": "../../Unige/2021-2022_Master_project/Data/x_matches/GLIMPSE_SMOG_2_arc.fits",
    "GLIMPSE_SMOG_2_arc_flux": ["f3_6", "f4_5", "f5_8", "f8_0"],
    "GLIMPSE_SMOG_2_arc_err": ["df3_6", "df4_5", "df5_8", "df8_0"],
    "GLIMPSE_SMOG_2_arc_flag": [],
    "GLIMPSE_SMOG_2_arc_wave": [irac1, irac2, irac3, irac4],
    "GLIMPSE_SMOG_2_arc_pref": pow(10, -3),
    "GLIMPSE_SMOG_2_arc_flux_unit": True,

    "GLIMPSE_Cygnus_2_arc": "../../Unige/2021-2022_Master_project/Data/x_matches/GLIMPSE_Cygnus_2_arc.fits",
    "GLIMPSE_Cygnus_2_arc_flux": ["f3_6", "f4_5", "f5_8", "f8_0"],
    "GLIMPSE_Cygnus_2_arc_err": ["df3_6", "df4_5", "df5_8", "df8_0"],
    "GLIMPSE_Cygnus_2_arc_flag": [],
    "GLIMPSE_Cygnus_2_arc_wave": [irac1, irac2, irac3, irac4],
    "GLIMPSE_Cygnus_2_arc_pref": pow(10, -3),
    "GLIMPSE_Cygnus_2_arc_flux_unit": True,

    "MIPSGAL_2_arc": "../../Unige/2021-2022_Master_project/Data/x_matches/MIPSGAL_2_arc.fits",
    "MIPSGAL_2_arc_flux": ["fnu_3_6", "fnu_4_5", "fnu_5_8", "fnu_8_0", "fnu_24"],
    "MIPSGAL_2_arc_err": ["sigma_fnu_3_6", "sigma_fnu_4_5", "sigma_fnu_5_8", "sigma_fnu_8_0", "sigma_fnu_24"],
    "MIPSGAL_2_arc_flag": [],
    "MIPSGAL_2_arc_wave": [irac1, irac2, irac3, irac4, mips1],
    "MIPSGAL_2_arc_pref": pow(10, -3),
    "MIPSGAL_2_arc_flux_unit": True,

    "Taurus_2_arc": "../../Unige/2021-2022_Master_project/Data/x_matches/Taurus_2_arc.fits",
    "Taurus_2_arc_flux": ["i1_02px_flx", "i2_02px_flx", "i3_02px_flx", "i4_02px_flx", "m1flux", "m2flux"],
    "Taurus_2_arc_err": ["i1_02px_err", "i2_02px_err", "i3_02px_err", "i4_02px_err", "m1fluxerr", "m2fluxerr"],
    "Taurus_2_arc_flag": [],
    "Taurus_2_arc_wave": [irac1, irac2, irac3, irac4, mips1, mips2],
    "Taurus_2_arc_pref": pow(10, -6),
    "Taurus_2_arc_flux_unit": True,

    "YSO_GGD_2_arc": "../../Unige/2021-2022_Master_project/Data/x_matches/YSO_GGD_2_arc.fits",
    "YSO_GGD_2_arc_flux": ["c3_6mag", "c4_5mag", "c5_8mag", "c8_0mag", "c24mag"],
    "YSO_GGD_2_arc_err": ["c3_6err", "c4_5err", "c5_8err", "c8_0err", "c24err"],
    "YSO_GGD_2_arc_flag": ["c3_6lim", "c4_5lim", "c5_8lim", "c8_0lim", "c24lim"],
    "YSO_GGD_2_arc_wave": [irac1, irac2, irac3, irac4, mips1],
    "YSO_GGD_2_arc_pref": 1,
    "YSO_GGD_2_arc_flux_unit": False,

    "YSO_L1688_2_arc": "../../Unige/2021-2022_Master_project/Data/x_matches/YSO_L1688_2_arc.fits",
    "YSO_L1688_2_arc_flux": ["c3_6mag", "c4_5mag", "c5_8mag", "c8_0mag", "c24mag", "c70mag"],
    "YSO_L1688_2_arc_err": ["c3_6err", "c4_5err", "c5_8err", "c8_0err", "c24err", "c70err"],
    "YSO_L1688_2_arc_flag": ["c3_6lim", "c4_5lim", "c5_8lim", "c8_0lim", "c24lim", "c70lim"],
    "YSO_L1688_2_arc_wave": [irac1, irac2, irac3, irac4, mips1, mips2],
    "YSO_L1688_2_arc_pref": 1,
    "YSO_L1688_2_arc_flux_unit": False,

    "PACS_70_5_6_arc": "../../Unige/2021-2022_Master_project/Data/x_matches/PACS_70_5_6_arc.fits",
    "PACS_70_5_6_arc_flux": ["flux"],
    "PACS_70_5_6_arc_err": ["snrnoise"],
    "PACS_70_5_6_arc_flag": [],
    "PACS_70_5_6_arc_wave": [pacs70],
    "PACS_70_5_6_arc_pref": pow(10, -3),
    "PACS_70_5_6_arc_flux_unit": True,

    "PACS_100_6_8_arc": "../../Unige/2021-2022_Master_project/Data/x_matches/PACS_100_6_8_arc.fits",
    "PACS_100_6_8_arc_flux": ["flux"],
    "PACS_100_6_8_arc_err": ["snrnoise"],
    "PACS_100_6_8_arc_flag": [],
    "PACS_100_6_8_arc_wave": [pacs100],
    "PACS_100_6_8_arc_pref": pow(10, -3),
    "PACS_100_6_8_arc_flux_unit": True,

    "PACS_160_10_7_arc": "../../Unige/2021-2022_Master_project/Data/x_matches/PACS_160_10_7_arc.fits",
    "PACS_160_10_7_arc_flux": ["flux"],
    "PACS_160_10_7_arc_err": ["snrnoise"],
    "PACS_160_10_7_arc_flag": [],
    "PACS_160_10_7_arc_wave": [pacs160],
    "PACS_160_10_7_arc_pref": pow(10, -3),
    "PACS_160_10_7_arc_flux_unit": True,

    "SPIRE_250_17_6_arc": "../../Unige/2021-2022_Master_project/Data/x_matches/SPIRE_250_17_6_arc.fits",
    "SPIRE_250_17_6_arc_flux": ["flux"],
    "SPIRE_250_17_6_arc_err": ["flux_err"],
    "SPIRE_250_17_6_arc_flag": [],
    "SPIRE_250_17_6_arc_wave": [spire250],
    "SPIRE_250_17_6_arc_pref": pow(10, -3),
    "SPIRE_250_17_6_arc_flux_unit": True,

    "SPIRE_350_23_9_arc": "../../Unige/2021-2022_Master_project/Data/x_matches/SPIRE_350_23_9_arc.fits",
    "SPIRE_350_23_9_arc_flux": ["flux"],
    "SPIRE_350_23_9_arc_err": ["flux_err"],
    "SPIRE_350_23_9_arc_flag": [],
    "SPIRE_350_23_9_arc_wave": [spire350],
    "SPIRE_350_23_9_arc_pref": pow(10, -3),
    "SPIRE_350_23_9_arc_flux_unit": True,

    "SPIRE_500_35_2_arc": "../../Unige/2021-2022_Master_project/Data/x_matches/SPIRE_500_35_2_arc.fits",
    "SPIRE_500_35_2_arc_flux": ["flux"],
    "SPIRE_500_35_2_arc_err": ["flux_err"],
    "SPIRE_500_35_2_arc_flag": [],
    "SPIRE_500_35_2_arc_wave": [spire500],
    "SPIRE_500_35_2_arc_pref": pow(10, -3),
    "SPIRE_500_35_2_arc_flux_unit": True,

    "twomass_2_arc": "../../Unige/2021-2022_Master_project/Data/x_matches/twomass_2_arc.fits",
    "twomass_2_arc_flux": ["j_m", "h_m", "k_m"],
    "twomass_2_arc_err": ["j_msigcom", "h_msigcom", "k_msigcom"],
    "twomass_2_arc_flag": ["ph_qual"],
    "twomass_2_arc_wave": [j_2mass, h_2mass, k_2mass],
    "twomass_2_arc_pref": 1,
    "twomass_2_arc_flux_unit": False,

    "gaia_dr2_edr3": "../../Unige/2021-2022_Master_project/Data/x_matches/gaia_dr2_edr3.fits",
    "gaia_dr2_edr3_flux": ["phot_g_mean_mag_g2", "phot_bp_mean_mag_g2", "phot_rp_mean_mag_g2",
                           "phot_g_mean_mag_ge3", "phot_bp_mean_mag_ge3", "phot_rp_mean_mag_ge3"],
    "gaia_dr2_edr3_err": ["g_mean_mag_err_g2", "bp_mean_mag_err_g2", "rp_mean_mag_err_g2",
                          "g_mean_mag_err_ge3", "bp_mean_mag_err_ge3", "rp_mean_mag_err_ge3"],
    "gaia_dr2_edr3_flag": [],
    "gaia_dr2_edr3_wave": [g_dr2, bp_dr2, rp_dr2,
                           g_edr3, bp_edr3, rp_edr3],
    "gaia_dr2_edr3_pref": 1,
    "gaia_dr2_edr3_flux_unit": False
    }

extra_catalogues_dir_lav = {
    "table_names": ["WISE_2_arc", "ALLWISE_2_arc", "SEIP_2_arc", "C2D_reliable_stars_2_arc", "C2D_cloud_2_arc",
                    "CSI_2264_2_arc",
                    "Cygnus_X_cat_2_arc", "GLIMPSE_I_2_arc", "GLIMPSE_II_2_arc", "GLIMPSE_II_Epoch_1_2_arc",
                    "GLIMPSE_II_Epoch_2_2_arc", "GLIMPSE_3D_2_arc", "GLIMPSE_3D_Epoch_1_2_arc",
                    "GLIMPSE_3D_Epoch_2_2_arc", "GLIMPSE_Vela_2_arc", "GLIMPSE_SMOG_2_arc",
                    "GLIMPSE_Cygnus_2_arc", "MIPSGAL_2_arc", "Taurus_2_arc", "YSO_GGD_2_arc",
                    "YSO_L1688_2_arc", "PACS_70_5_6_arc", "PACS_100_6_8_arc",
                    "twomass_2_arc", "gaia_dr2_edr3"],

    "WISE_2_arc": "../../Unige/2021-2022_Master_project/Data/x_matches_lav/WISE_2_arc_.fits",
    "WISE_2_arc_flux": ["w1mpro", "w2mpro", "w3mpro", "w4mpro"],
    "WISE_2_arc_err": ["w1sigmpro", "w2sigmpro", "w3sigmpro", "w4sigmpro"],
    "WISE_2_arc_flag": ["ph_qual"],
    "WISE_2_arc_wave": [W1, W2, W3, W4],
    "WISE_2_arc_pref": 1,
    "WISE_2_arc_flux_unit": False,

    "ALLWISE_2_arc": "../../Unige/2021-2022_Master_project/Data/x_matches_lav/ALLWISE_2_arc_.fits",
    "ALLWISE_2_arc_flux": ["w1mpro", "w2mpro", "w3mpro", "w4mpro"],
    "ALLWISE_2_arc_err": ["w1sigmpro", "w2sigmpro", "w3sigmpro", "w4sigmpro"],
    "ALLWISE_2_arc_flag": ["ph_qual"],
    "ALLWISE_2_arc_wave": [W1, W2, W3, W4],
    "ALLWISE_2_arc_pref": 1,
    "ALLWISE_2_arc_flux_unit": False,

    "SEIP_2_arc": "../../Unige/2021-2022_Master_project/Data/x_matches_lav/SEIP_2_arc.fits",
    "SEIP_2_arc_flux": ["i1_f_ap1", "i2_f_ap1", "i3_f_ap1", "i4_f_ap1", "m1_f_ap"],
    "SEIP_2_arc_err": ["i1_df_ap1", "i2_df_ap1", "i3_df_ap1", "i4_df_ap1", "m1_df_ap"],
    "SEIP_2_arc_flag": ["i1_fluxtype", "i2_fluxtype", "i3_fluxtype", "i4_fluxtype", "m1_fluxtype"],
    "SEIP_2_arc_wave": [irac1, irac2, irac3, irac4, mips1],
    "SEIP_2_arc_pref": pow(10, -6),
    "SEIP_2_arc_flux_unit": True,

    "C2D_reliable_stars_2_arc": "../../Unige/2021-2022_Master_project/Data/x_matches_lav/C2D_reliable_stars_2_arc.fits",
    "C2D_reliable_stars_2_arc_flux": ["ir1_flux_c", "ir2_flux_c", "ir3_flux_c", "ir4_flux_c", "mp1_flux_c",
                                      "mp2_flux_c"],
    "C2D_reliable_stars_2_arc_err": ["ir1_d_flux_c", "ir2_d_flux_c", "ir3_d_flux_c", "ir4_d_flux_c", "mp1_d_flux_c",
                                     "mp2_d_flux_c"],
    "C2D_reliable_stars_2_arc_flag": ["ir1_q_det_c", "ir2_q_det_c", "ir3_q_det_c", "ir4_q_det_c", "mp1_q_det_c",
                                      "mp2_q_det_c"],
    "C2D_reliable_stars_2_arc_wave": [irac1, irac2, irac3, irac4, mips1, mips2],
    "C2D_reliable_stars_2_arc_pref": pow(10, -3),
    "C2D_reliable_stars_2_arc_flux_unit": True,

    "C2D_cloud_2_arc": "../../Unige/2021-2022_Master_project/Data/x_matches_lav/C2D_cloud_2_arc.fits",
    "C2D_cloud_2_arc_flux": ["ir1_flux_c", "ir2_flux_c", "ir3_flux_c", "ir4_flux_c", "mp1_flux_c", "mp2_flux_c"],
    "C2D_cloud_2_arc_err": ["ir1_d_flux_c", "ir2_d_flux_c", "ir3_d_flux_c", "ir4_d_flux_c", "mp1_d_flux_c",
                            "mp2_d_flux_c"],
    "C2D_cloud_2_arc_flag": ["ir1_q_det_c", "ir2_q_det_c", "ir3_q_det_c", "ir4_q_det_c", "mp1_q_det_c", "mp2_q_det_c"],
    "C2D_cloud_2_arc_wave": [irac1, irac2, irac3, irac4, mips1, mips2],
    "C2D_cloud_2_arc_pref": pow(10, -3),
    "C2D_cloud_2_arc_flux_unit": True,

    "CSI_2264_2_arc": "../../Unige/2021-2022_Master_project/Data/x_matches_lav/CSI_2264_2_arc.fits",
    "CSI_2264_2_arc_flux": ["mag_36", "mag_45", "mag_58", "mag_80", "mag_24"],
    "CSI_2264_2_arc_err": ["err_36", "err_45", "err_58", "err_80", "err_24"],
    "CSI_2264_2_arc_flag": [],
    "CSI_2264_2_arc_wave": [irac1, irac2, irac3, irac4, mips1],
    "CSI_2264_2_arc_pref": 1,
    "CSI_2264_2_arc_flux_unit": False,

    "Cygnus_X_cat_2_arc": "../../Unige/2021-2022_Master_project/Data/x_matches_lav/Cygnus_X_cat_2_arc.fits",
    "Cygnus_X_cat_2_arc_flux": ["i1flux", "i2flux", "i3flux", "i4flux", "m24flux"],
    "Cygnus_X_cat_2_arc_err": ["i1ferr", "i2ferr", "i3ferr", "i4ferr", "m24ferr"],
    "Cygnus_X_cat_2_arc_flag": [],
    "Cygnus_X_cat_2_arc_wave": [irac1, irac2, irac3, irac4, mips1],
    "Cygnus_X_cat_2_arc_pref": pow(10, -6),
    "Cygnus_X_cat_2_arc_flux_unit": True,

    "GLIMPSE_I_2_arc": "../../Unige/2021-2022_Master_project/Data/x_matches_lav/GLIMPSE_I_2_arc.fits",
    "GLIMPSE_I_2_arc_flux": ["f3_6", "f4_5", "f5_8", "f8_0"],
    "GLIMPSE_I_2_arc_err": ["df3_6", "df4_5", "df5_8", "df8_0"],
    "GLIMPSE_I_2_arc_flag": [],
    "GLIMPSE_I_2_arc_wave": [irac1, irac2, irac3, irac4],
    "GLIMPSE_I_2_arc_pref": pow(10, -3),
    "GLIMPSE_I_2_arc_flux_unit": True,

    "GLIMPSE_II_2_arc": "../../Unige/2021-2022_Master_project/Data/x_matches_lav/GLIMPSE_II_2_arc.fits",
    "GLIMPSE_II_2_arc_flux": ["f3_6", "f4_5", "f5_8", "f8_0"],
    "GLIMPSE_II_2_arc_err": ["df3_6", "df4_5", "df5_8", "df8_0"],
    "GLIMPSE_II_2_arc_flag": [],
    "GLIMPSE_II_2_arc_wave": [irac1, irac2, irac3, irac4],
    "GLIMPSE_II_2_arc_pref": pow(10, -3),
    "GLIMPSE_II_2_arc_flux_unit": True,

    "GLIMPSE_II_Epoch_1_2_arc": "../../Unige/2021-2022_Master_project/Data/x_matches_lav/GLIMPSE_II_Epoch_1_2_arc.fits",
    "GLIMPSE_II_Epoch_1_2_arc_flux": ["f3_6", "f4_5", "f5_8", "f8_0"],
    "GLIMPSE_II_Epoch_1_2_arc_err": ["df3_6", "df4_5", "df5_8", "df8_0"],
    "GLIMPSE_II_Epoch_1_2_arc_flag": [],
    "GLIMPSE_II_Epoch_1_2_arc_wave": [irac1, irac2, irac3, irac4],
    "GLIMPSE_II_Epoch_1_2_arc_pref": pow(10, -3),
    "GLIMPSE_II_Epoch_1_2_arc_flux_unit": True,

    "GLIMPSE_II_Epoch_2_2_arc": "../../Unige/2021-2022_Master_project/Data/x_matches_lav/GLIMPSE_II_Epoch_2_2_arc.fits",
    "GLIMPSE_II_Epoch_2_2_arc_flux": ["f3_6", "f4_5", "f5_8", "f8_0"],
    "GLIMPSE_II_Epoch_2_2_arc_err": ["df3_6", "df4_5", "df5_8", "df8_0"],
    "GLIMPSE_II_Epoch_2_2_arc_flag": [],
    "GLIMPSE_II_Epoch_2_2_arc_wave": [irac1, irac2, irac3, irac4],
    "GLIMPSE_II_Epoch_2_2_arc_pref": pow(10, -3),
    "GLIMPSE_II_Epoch_2_2_arc_flux_unit": True,

    "GLIMPSE_3D_2_arc": "../../Unige/2021-2022_Master_project/Data/x_matches_lav/GLIMPSE_3D_2_arc.fits",
    "GLIMPSE_3D_2_arc_flux": ["f3_6", "f4_5", "f5_8", "f8_0"],
    "GLIMPSE_3D_2_arc_err": ["df3_6", "df4_5", "df5_8", "df8_0"],
    "GLIMPSE_3D_2_arc_flag": [],
    "GLIMPSE_3D_2_arc_wave": [irac1, irac2, irac3, irac4],
    "GLIMPSE_3D_2_arc_pref": pow(10, -3),
    "GLIMPSE_3D_2_arc_flux_unit": True,

    "GLIMPSE_3D_Epoch_1_2_arc": "../../Unige/2021-2022_Master_project/Data/x_matches_lav/GLIMPSE_3D_Epoch_1_2_arc.fits",
    "GLIMPSE_3D_Epoch_1_2_arc_flux": ["f3_6", "f4_5", "f5_8", "f8_0"],
    "GLIMPSE_3D_Epoch_1_2_arc_err": ["df3_6", "df4_5", "df5_8", "df8_0"],
    "GLIMPSE_3D_Epoch_1_2_arc_flag": [],
    "GLIMPSE_3D_Epoch_1_2_arc_wave": [irac1, irac2, irac3, irac4],
    "GLIMPSE_3D_Epoch_1_2_arc_pref": pow(10, -3),
    "GLIMPSE_3D_Epoch_1_2_arc_flux_unit": True,

    "GLIMPSE_3D_Epoch_2_2_arc": "../../Unige/2021-2022_Master_project/Data/x_matches_lav/GLIMPSE_3D_Epoch_2_2_arc.fits",
    "GLIMPSE_3D_Epoch_2_2_arc_flux": ["f3_6", "f4_5", "f5_8", "f8_0"],
    "GLIMPSE_3D_Epoch_2_2_arc_err": ["df3_6", "df4_5", "df5_8", "df8_0"],
    "GLIMPSE_3D_Epoch_2_2_arc_flag": [],
    "GLIMPSE_3D_Epoch_2_2_arc_wave": [irac1, irac2, irac3, irac4],
    "GLIMPSE_3D_Epoch_2_2_arc_pref": pow(10, -3),
    "GLIMPSE_3D_Epoch_2_2_arc_flux_unit": True,

    "GLIMPSE_Vela_2_arc": "../../Unige/2021-2022_Master_project/Data/x_matches_lav/GLIMPSE_Vela_2_arc.fits",
    "GLIMPSE_Vela_2_arc_flux": ["f3_6", "f4_5", "f5_8", "f8_0"],
    "GLIMPSE_Vela_2_arc_err": ["df3_6", "df4_5", "df5_8", "df8_0"],
    "GLIMPSE_Vela_2_arc_flag": [],
    "GLIMPSE_Vela_2_arc_wave": [irac1, irac2, irac3, irac4],
    "GLIMPSE_Vela_2_arc_pref": pow(10, -3),
    "GLIMPSE_Vela_2_arc_flux_unit": True,

    "GLIMPSE_SMOG_2_arc": "../../Unige/2021-2022_Master_project/Data/x_matches_lav/GLIMPSE_SMOG_2_arc.fits",
    "GLIMPSE_SMOG_2_arc_flux": ["f3_6", "f4_5", "f5_8", "f8_0"],
    "GLIMPSE_SMOG_2_arc_err": ["df3_6", "df4_5", "df5_8", "df8_0"],
    "GLIMPSE_SMOG_2_arc_flag": [],
    "GLIMPSE_SMOG_2_arc_wave": [irac1, irac2, irac3, irac4],
    "GLIMPSE_SMOG_2_arc_pref": pow(10, -3),
    "GLIMPSE_SMOG_2_arc_flux_unit": True,

    "GLIMPSE_Cygnus_2_arc": "../../Unige/2021-2022_Master_project/Data/x_matches_lav/GLIMPSE_Cygnus_2_arc.fits",
    "GLIMPSE_Cygnus_2_arc_flux": ["f3_6", "f4_5", "f5_8", "f8_0"],
    "GLIMPSE_Cygnus_2_arc_err": ["df3_6", "df4_5", "df5_8", "df8_0"],
    "GLIMPSE_Cygnus_2_arc_flag": [],
    "GLIMPSE_Cygnus_2_arc_wave": [irac1, irac2, irac3, irac4],
    "GLIMPSE_Cygnus_2_arc_pref": pow(10, -3),
    "GLIMPSE_Cygnus_2_arc_flux_unit": True,

    "MIPSGAL_2_arc": "../../Unige/2021-2022_Master_project/Data/x_matches_lav/MIPSGAL_2_arc.fits",
    "MIPSGAL_2_arc_flux": ["fnu_3_6", "fnu_4_5", "fnu_5_8", "fnu_8_0", "fnu_24"],
    "MIPSGAL_2_arc_err": ["sigma_fnu_3_6", "sigma_fnu_4_5", "sigma_fnu_5_8", "sigma_fnu_8_0", "sigma_fnu_24"],
    "MIPSGAL_2_arc_flag": [],
    "MIPSGAL_2_arc_wave": [irac1, irac2, irac3, irac4, mips1],
    "MIPSGAL_2_arc_pref": pow(10, -3),
    "MIPSGAL_2_arc_flux_unit": True,

    "Taurus_2_arc": "../../Unige/2021-2022_Master_project/Data/x_matches_lav/Taurus_2_arc.fits",
    "Taurus_2_arc_flux": ["i1_02px_flx", "i2_02px_flx", "i3_02px_flx", "i4_02px_flx", "m1flux", "m2flux"],
    "Taurus_2_arc_err": ["i1_02px_err", "i2_02px_err", "i3_02px_err", "i4_02px_err", "m1fluxerr", "m2fluxerr"],
    "Taurus_2_arc_flag": [],
    "Taurus_2_arc_wave": [irac1, irac2, irac3, irac4, mips1, mips2],
    "Taurus_2_arc_pref": pow(10, -6),
    "Taurus_2_arc_flux_unit": True,

    "YSO_GGD_2_arc": "../../Unige/2021-2022_Master_project/Data/x_matches_lav/YSO_GGD_2_arc.fits",
    "YSO_GGD_2_arc_flux": ["c3_6mag", "c4_5mag", "c5_8mag", "c8_0mag", "c24mag"],
    "YSO_GGD_2_arc_err": ["c3_6err", "c4_5err", "c5_8err", "c8_0err", "c24err"],
    "YSO_GGD_2_arc_flag": ["c3_6lim", "c4_5lim", "c5_8lim", "c8_0lim", "c24lim"],
    "YSO_GGD_2_arc_wave": [irac1, irac2, irac3, irac4, mips1],
    "YSO_GGD_2_arc_pref": 1,
    "YSO_GGD_2_arc_flux_unit": False,

    "YSO_L1688_2_arc": "../../Unige/2021-2022_Master_project/Data/x_matches_lav/YSO_L1688_2_arc.fits",
    "YSO_L1688_2_arc_flux": ["c3_6mag", "c4_5mag", "c5_8mag", "c8_0mag", "c24mag", "c70mag"],
    "YSO_L1688_2_arc_err": ["c3_6err", "c4_5err", "c5_8err", "c8_0err", "c24err", "c70err"],
    "YSO_L1688_2_arc_flag": ["c3_6lim", "c4_5lim", "c5_8lim", "c8_0lim", "c24lim", "c70lim"],
    "YSO_L1688_2_arc_wave": [irac1, irac2, irac3, irac4, mips1, mips2],
    "YSO_L1688_2_arc_pref": 1,
    "YSO_L1688_2_arc_flux_unit": False,

    "PACS_70_5_6_arc": "../../Unige/2021-2022_Master_project/Data/x_matches_lav/PACS_70_5_6_arc.fits",
    "PACS_70_5_6_arc_flux": ["flux"],
    "PACS_70_5_6_arc_err": ["snrnoise"],
    "PACS_70_5_6_arc_flag": [],
    "PACS_70_5_6_arc_wave": [pacs70],
    "PACS_70_5_6_arc_pref": pow(10, -3),
    "PACS_70_5_6_arc_flux_unit": True,

    "PACS_100_6_8_arc": "../../Unige/2021-2022_Master_project/Data/x_matches_lav/PACS_100_6_8_arc.fits",
    "PACS_100_6_8_arc_flux": ["flux"],
    "PACS_100_6_8_arc_err": ["snrnoise"],
    "PACS_100_6_8_arc_flag": [],
    "PACS_100_6_8_arc_wave": [pacs100],
    "PACS_100_6_8_arc_pref": pow(10, -3),
    "PACS_100_6_8_arc_flux_unit": True,

    "twomass_2_arc": "../../Unige/2021-2022_Master_project/Data/x_matches_lav/twomass_2_arc.fits",
    "twomass_2_arc_flux": ["j_m", "h_m", "k_m"],
    "twomass_2_arc_err": ["j_msigcom", "h_msigcom", "k_msigcom"],
    "twomass_2_arc_flag": ["ph_qual"],
    "twomass_2_arc_wave": [j_2mass, h_2mass, k_2mass],
    "twomass_2_arc_pref": 1,
    "twomass_2_arc_flux_unit": False,

    "gaia_dr2_edr3": "../../Unige/2021-2022_Master_project/Data/x_matches_lav/gaia_dr2_edr3.fits",
    "gaia_dr2_edr3_flux": ["phot_g_mean_mag_g2", "phot_bp_mean_mag_g2", "phot_rp_mean_mag_g2",
                           "phot_g_mean_mag_ge3", "phot_bp_mean_mag_ge3", "phot_rp_mean_mag_ge3"],
    "gaia_dr2_edr3_err": ["g_mean_mag_err_g2", "bp_mean_mag_err_g2", "rp_mean_mag_err_g2",
                          "g_mean_mag_err_ge3", "bp_mean_mag_err_ge3", "rp_mean_mag_err_ge3"],
    "gaia_dr2_edr3_flag": [],
    "gaia_dr2_edr3_wave": [g_dr2, bp_dr2, rp_dr2,
                           g_edr3, bp_edr3, rp_edr3],
    "gaia_dr2_edr3_pref": 1,
    "gaia_dr2_edr3_flux_unit": False
    }

irsa_catalogues = ["neowiser_p1bs_psd", "allwise_p3as_psd", "allsky_4band_p3as_psd", "fp_psc", "slphotdr4", "dr4_clouds_hrel",
                   "dr4_off_cloud_hrel", "dr4_cores_hrel", "dr4_stars_hrel", "csi2264t1", "cygx_cat", "glimpse_s07",
                   "glimpse2_v2cat", "glimpse2ep1c08", "glimpse2ep2mra09", "glimpse3d_v1cat_tbl", "glimpse3dep1c",
                   "glimpse3dep2mra", "glimpse360c", "velcarc", "glimpsesmogc", "glimpsecygxc", "mipsgalc",
                   "taurus_2008_2_1", "ysoggd1215obj", "ysoi20050obj", "ysol1688obj", "yson1333obj", "ppsc_70",
                   "ppsc_100"]

xmm_catalogues = ['XMM-EPIC-STACK', 'XMM-EPIC', 'XMM-SLEW', 'CHANDRA-SC2'] # XMM-OM - 180-600 nm
xmm_catalogues_xmatch_radius = [6, 6, 4.2, 1]

catalogue_properties = {"allwise_p3as_psd_flux": ["W1mag_wise", "W2mag_wise", "W3mag_wise", "W4mag_wise"],
                        "allwise_p3as_psd_err": ["e_W1mag_wise", "e_W2mag_wise", "e_W3mag_wise", "e_W4mag_wise"],
                        "allwise_p3as_psd_flag": ["ph_qual_wise"],
                        "allwise_p3as_psd_wave": [W1, W2, W3, W4],
                        "allwise_p3as_psd_pref": 1,
                        "allwise_p3as_psd_flux_unit": False,
                        }


# Constants:

distance_to_vega_pc = 7.68  # pc
wiens_displ_const = 2.897771955 * np.power(10.0, -3)        # SI units
light_speed = 299792458.0   # m/s
h = 6.62607004e-34  # Planck's constant, SI units
kb = 1.380649e-23   # Boltzman constant, SI units

dflt_no_vle = -9999  # default value if basically none is expected


# Gaia Table Field names

field_name_distance = 'distance'
field_name_parallax_g3 = 'parallax_ge3'    # GE3 or G3
field_name_parallax_error_g3 = 'parallax_error_ge3'
field_name_ra_g3 = 'ra_ge3'
field_name_dec_g3 = 'dec_ge3'
field_name_l_g3 = 'l_ge3'
field_name_b_g3 = 'b_ge3'
field_name_pmra_g3 = 'pmra_ge3'
field_name_pmdec_g3 = 'pmdec_ge3'
field_name_g_mag_g2 = 'phot_g_mean_mag_g2'
field_name_bp_mag_g2 = 'phot_bp_mean_mag_g2'
field_name_rp_mag_g2 = 'phot_rp_mean_mag_g2'
field_name_rad_vel_g2 = 'radial_velocity_g2'
field_name_period_g2 = 'period_g2'
field_name_period_error_g2 = 'period_error_g2'
field_name_a_g_val_g2 = 'a_g_val_g2'
field_name_rv_template_teff_g2 = 'rv_template_teff_g2'
field_name_teff_val_g2 = 'teff_val_g2'
field_name_radius_g2 = 'radius_val_g2'
field_name_radius_lower_g2 = 'radius_percentile_lower_g2'
field_name_radius_upper_g2 = 'radius_percentile_upper_g2'
field_name_main_id_simbad = 'main_id_simbad'
field_name_main_type_simbad = 'main_type_simbad'
field_name_other_types_simbad = 'other_types_simbad'
field_name_g_mag_g3 = 'phot_g_mean_mag_ge3'
field_name_bp_mag_g3 = 'phot_bp_mean_mag_ge3'
field_name_rp_mag_g3 = 'phot_rp_mean_mag_ge3'

