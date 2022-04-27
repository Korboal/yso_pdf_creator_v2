#import get_astroquery
from star_class import Star
import numpy as np
import get_gaia_table
import tools
import config_file as cf
import matplotlib
from xmatched_tables_class import XMatchTables
import os
import pandas as pd
import time
from do_astrometry_analysis import do_nebula_pm_analysis
from astropy.io.votable import parse_single_table

if not cf.debug_mode:   # If doing lots of images, without this function need insane amount of RAM
    matplotlib.use('Agg')

# TODO:
# - Double check default function parameter: is made at initialisation of script, not each function's call
# - Check == for True/False/None
# - Convert loops to "for i,v in in enumerate(a):" instead of range
# - "for av, bv in zip(a,b):" "for i, (av, bv) in enumerate(zip(a,b)):"
# - Dictionary: when looping over it instead use "for key in dict:"
# - Looping over values and dictionary: "for key, val in d.items():"
# - logging instead of print: "import logging // logging.debug() logging.info() logging.error("error_msg")" etc etc
# https://www.youtube.com/watch?v=qUeud6DvOWIwdw


def main():
    """
    The main function of the program. It takes in Gaia information table and depending on requested information runs
    appropriate functions.
    """

    gaia_lpv_table = get_gaia_table.get_data_from_table(cf.input_gaia_table_path)     # Get main Gaia info (GDR2 + GEDR3)
    ids = tools.get_specific_column_from_data_str(gaia_lpv_table, "source_id").astype(str)
    if np.size(cf.ids_to_calculate) != 0:  # If want to only run program for specific IDs, then it takes those IDs here
        ids = tools.get_ids_to_analyse(ids, cf.ids_to_calculate)
    ids = np.asarray(ids)

    tools.create_all_directories()

    sedfitter_distances_that_exist = []     # SEDFitter will take stars at these distances. I.e. name of files for it

    time_before = time.perf_counter()   # This is for estimate of the "how much time left to run the program"
    recent_time_length = []

    sed_txt_to_save = cf.foundation_txt     # Text to save frm sed_line_fit functions
    for i in range(0, len(cf.other_txts_for_sed)):
        if cf.sed_fits_to_do[i + 1] == "1":
            sed_txt_to_save += cf.other_txts_for_sed[i]

    x_match_obj = XMatchTables(cf.extra_catalogues_dir)     # Get the data from other x-matched catalogues
    x_match_obj.__setstate__()

    if cf.lightcurve_fit_save_variables and cf.lightcurve_fit:  # To prepare saving variables for all fits
        tools.save_in_txt([cf.lightcurve_gaia_txt_to_save], cf.output_fitted_period_text)
    if cf.sed_line_fit_save_variables and cf.sed_line_fit:
        tools.save_in_txt([sed_txt_to_save], cf.output_fitted_param)
    if cf.save_bb_variables and cf.do_bb_fit:
        tools.save_in_txt([cf.bb_fit_txt_to_save], cf.output_sed_temp_fits_txt)
    if cf.ztf_save_variables and cf.ztf_lightcurve_fit:
        tools.save_in_txt([cf.ztf_lightcurve_txt_to_save], cf.output_ztf_lightcurves_period)

    if cf.find_nebula:
        nebulae_files = tools.get_all_file_names_in_a_folder(cf.input_folder_nebulae)
        tools.save_in_txt_topcat(["# source_id_\tra\tdec\tpmra\tpmdec\tclosest_nebula\tnebula\tdist_to_closest_nebula\tconfidence"], cf.output_textfile_nebulae)
        tools.save_in_txt_topcat(["# nebula_name\tra_neb\tdec_neb\tdist_to_neb_pc\tsize_neb_pc\tpmra_neb\tpmdec_neb\tnumber_of_stars"], cf.output_textfile_nebulae_only_pm)
        tools.save_in_txt_topcat(["# source_id_\tra\tdec\tpmra\tpmdec\tclosest_nebula\tnebula\tdist_to_closest_nebula\tconfidence\tis_part_of_nebula"], cf.output_textfile_nebulae_star_pm)
        nebulae_data = None
        for file in nebulae_files:
            nebula_data = tools.load_data(cf.input_folder_nebulae + file)
            if nebulae_data is None:
                nebulae_data = nebula_data
            else:
                nebulae_data = np.append(nebulae_data, nebula_data, axis=0)

        #nebula_data_visual = tools.load_data("mol_clouds/og_or_old/mol_clouds_full_visual_sizes.txt")

    if cf.do_vvvdr4_astroquery:
        for catalogue in ["VVVDR4"]:
            output_path = cf.xmatched_new_catalogues_directory + catalogue + "_" + cf.file_category + ".csv"
            output_empty_path = cf.xmatched_new_catalogues_directory + catalogue + "_" + cf.file_category + "_no_xmatch_ids.txt"

            if os.path.isfile(output_path):
                tb = pd.read_csv(output_path)
                all_source_ids = list(tb["source_id_gdr2"].values.astype(str))
            else:
                all_source_ids = []

            if os.path.isfile(output_empty_path):
                all_source_ids_2 = list(tools.load_data(output_empty_path))
            else:
                all_source_ids_2 = []

            all_source_ids = all_source_ids + all_source_ids_2

            for i in range(np.size(ids)):
                star_object = Star(ids[i], x_match_obj)
                star_object.__setstate__(
                    get_gaia_table.get_star_info(gaia_lpv_table, ids[i]))  # Setting Gaia data here

                try:
                    get_astroquery.add_vvv_missing_entry(all_source_ids, output_path, output_empty_path,
                                                         star_object.source_id, star_object.ra,
                                                         star_object.dec, catalogue, 1)
                    #get_astroquery.add_vvv_missing_entry(all_source_ids, output_path, output_empty_path, star_object.source_id, star_object.ra,
                    #                                    star_object.dec, catalogue, 1)
                except:
                    print(f"ERROR, but keep going! {star_object.source_id}")

                print(int(i / np.size(ids) * 100))

                del star_object

    if cf.do_irsa_astroquery:
        for irsa_catalogue in cf.irsa_catalogues:
            output_path = cf.xmatched_new_catalogues_directory + irsa_catalogue + "_" + cf.file_category + ".csv"
            output_empty_path = cf.xmatched_new_catalogues_directory + irsa_catalogue + "_" + cf.file_category + "_no_xmatch_ids.txt"

            if os.path.isfile(output_path):
                tb = pd.read_csv(output_path)
                all_source_ids = list(tb["source_id_gdr2"].values.astype(str))
            else:
                all_source_ids = []

            if os.path.isfile(output_empty_path):
                all_source_ids_2 = list(tools.load_data(output_empty_path))
            else:
                all_source_ids_2 = []

            all_source_ids = all_source_ids + all_source_ids_2

            for i in range(np.size(ids)):
                star_object = Star(ids[i], x_match_obj)
                star_object.__setstate__(
                    get_gaia_table.get_star_info(gaia_lpv_table, ids[i]))  # Setting Gaia data here

                try:
                    get_astroquery.add_irsa_missing_entry(all_source_ids, output_path, output_empty_path, star_object.source_id, star_object.ra,
                                                        star_object.dec, irsa_catalogue, 2)
                except:
                    print(f"ERROR, but keep going! {star_object.source_id}")

                print(int(i / np.size(ids) * 100))

                del star_object

    if cf.neowise_lightcurve_fit:
        if os.path.isfile(cf.neowise_light_curve_file_path):
            neowise_votable_data = parse_single_table(cf.neowise_light_curve_file_path)
        else:
            print("Cannot find NEOWISE light curve")
            neowise_votable_data = None

    if cf.do_xmm_astroquery:
        for j in range(len(cf.xmm_catalogues)):
            xmm_catalogue = cf.xmm_catalogues[j]
            output_path = cf.xmatched_new_catalogues_directory + xmm_catalogue + "_" + cf.file_category + ".csv"
            output_empty_path = cf.xmatched_new_catalogues_directory + xmm_catalogue + "_" + cf.file_category + "_no_xmatch_ids.txt"

            if os.path.isfile(output_path):
                tb = pd.read_csv(output_path)
                all_source_ids = list(tb["source_id_gdr2"].values.astype(str))
            else:
                all_source_ids = []

            if os.path.isfile(output_empty_path):
                all_source_ids_2 = list(tools.load_data(output_empty_path))
            else:
                all_source_ids_2 = []

            all_source_ids = all_source_ids + all_source_ids_2

            for i in range(np.size(ids)):
                star_object = Star(ids[i], x_match_obj)
                star_object.__setstate__(get_gaia_table.get_star_info(gaia_lpv_table, ids[i]))  # Setting Gaia data here

                try:
                    get_astroquery.add_xmm_missing_entry(all_source_ids, output_path, output_empty_path, star_object.source_id, star_object.ra, star_object.dec, xmm_catalogue, cf.xmm_catalogues_xmatch_radius[j])
                except:
                    print(f"ERROR, but keep going! {star_object.source_id}")

                print(int(i / np.size(ids) * 100))

                del star_object


    for i in range(np.size(ids)):   # Go through each ID, at the end deleting each one to save memory
        star_object = Star(ids[i], x_match_obj)
        star_object.__setstate__(get_gaia_table.get_star_info(gaia_lpv_table, ids[i]))  # Setting Gaia data here

        if not tools.isfile(star_object.pdf_dir) or True:
            #if ids[i] in ids_to_do and not tools.isfile(star_object.pdf_dir):
            if cf.lightcurve_fit:   # Gaia light curve fit      # cf.lightcurve_fit_save_variables
                star_object.lightcurve_fit_and_plot(save_variables=cf.lightcurve_fit_save_variables,
                                                    save_images=cf.lightcurve_fit_save_images,
                                                    show_images=cf.lightcurve_fit_show_images,
                                                    print_variables=cf.lightcurve_fit_print_variables, manual_period_guess=cf.period_manual)

            if cf.ztf_lightcurve_fit:   # ZTF light curve fits
                star_object.analyse_ztf_lightcurves(cf.ztf_save_variables, cf.ztf_show_variables, cf.ztf_save_pictures,
                                                    cf.ztf_show_pictures, cf.do_multiband_periodogram, manual_period_guess=cf.period_manual)

            if cf.neowise_lightcurve_fit:
                star_object.analyse_neowise_lightcurves(neowise_votable_data, False, True, False, True)

            if cf.sed_line_fit:     # Fits for SED (not star models, but IR slopes)
                star_object.sed_line_fit_and_plot(cf.sed_fits_to_do,
                                                  save_variables=cf.sed_line_fit_save_variables,
                                                  save_images=cf.sed_line_fit_save_images,
                                                  show_images=cf.sed_line_fit_show_images,
                                                  print_variables=cf.sed_line_fit_print_variables)

            if cf.do_pdf_creation:      # PDF creation of stars with main info + plots
                star_object.create_pdf_canvas_for_a_star(gaia_lpv_table)

            if cf.do_bb_fit:    # My attempt at fitting stars with pure BB (doesn't work well obviously)
                star_object.sed_bb_fit(cf.bb_fit_save_images, cf.bb_fit_show_images, cf.save_bb_variables)

            if cf.do_sedfitter:     # Use SEDFitter library to fit actual star models
                if star_object.ir_slope25 is None:  # If want to only fit "YSO" based on slope
                    star_object.sed_line_fit_and_plot("0001000")
                if star_object.ir_slope25 > -0.5 and 0 <= star_object.distance_ge3 <= 10000:     # Here we choose which slope is considered to be "YSO". -99 would mean all stars usually
                    distance = star_object.distance_ge3     # Because SEDFitter fits with min/max dist, need to separate stars according to their distance ranges
                    dist_range = int((distance // cf.sedfitter_pc_separation) * cf.sedfitter_pc_separation)

                    if dist_range not in sedfitter_distances_that_exist:
                        sedfitter_distances_that_exist.append(dist_range)

                    sedfitter_file_input = cf.sedfitter_file_data + str(dist_range) + ".txt"
                    star_object.sed_fitter_fit(sedfitter_file_input, cf.sedfitter_filters_total)   # SEDFitter for plotting

            # star_object.print_amount_of_ir_points()

            if cf.find_nebula:
                star_object.get_closest_nebula(nebulae_data, nebula_data_visual)

        output_time_to_print, time_before, recent_time_length = tools.time_counter(i, time_before, np.size(ids),
                                                                                   recent_time_length)
        print(f"{star_object.source_id} {int(i / np.size(ids) * 100)}% {output_time_to_print}")  # Time count here

        del star_object     # Deleting each star class object to potentially save memory

    if cf.do_sedfitter:  # Do SEDFitter here
        from sed_fit_v1 import bb_model_fit_with_image
        print("Starting SEDFitter fitting")

        sedfitter_distances_that_exist.sort()

        all_fits = None
        output_joined = cf.sedfitter_output_parameters + "_joined.txt"

        all_fits_ranges = None
        output_ranges_joined = cf.sedfitter_output_parameters + "_joined_param_range.txt"

        if len(sedfitter_distances_that_exist) != 0:
            for current_distance in sedfitter_distances_that_exist:     # Goes through each distance range to fit correctly
                sedfitter_file_input = cf.sedfitter_file_data + str(current_distance) + ".txt"
                print("Fitting", current_distance, current_distance + cf.sedfitter_pc_separation)

                bb_model_fit_with_image(sedfitter_file_input, cf.sedfitter_filters_total, cf.sedfitter_models_dir,
                                        cf.sedfitter_output_parameters, current_distance,
                                        current_distance + cf.sedfitter_pc_separation)

                # save all parameters together
                new_fits = np.loadtxt(cf.sedfitter_output_parameters + "parameters.txt", dtype=str)

                if all_fits is None:
                    if new_fits.ndim == 1:
                        all_fits = [new_fits]
                    else:
                        all_fits = new_fits
                elif new_fits.ndim == 1:
                    all_fits = np.append(all_fits, [new_fits], axis=0)
                else:
                    all_fits = np.append(all_fits, new_fits, axis=0)

                if new_fits.ndim == 1:
                    tools.save_in_txt_topcat(new_fits, output_joined)
                else:
                    for row in list(new_fits):
                        tools.save_in_txt_topcat(row, output_joined)

                # save all parameter ranges together
                new_fits_ranges = np.loadtxt(cf.sedfitter_output_parameters + "parameters_ranges.txt", dtype=str)

                if all_fits_ranges is None:
                    if new_fits_ranges.ndim == 1:
                        all_fits_ranges = [new_fits_ranges]
                    else:
                        all_fits_ranges = new_fits_ranges
                elif new_fits_ranges.ndim == 1:
                    all_fits_ranges = np.append(all_fits_ranges, [new_fits_ranges], axis=0)
                else:
                    all_fits_ranges = np.append(all_fits_ranges, new_fits_ranges, axis=0)

                if new_fits_ranges.ndim == 1:
                    tools.save_in_txt_topcat(new_fits_ranges, output_ranges_joined)
                else:
                    for row in list(new_fits_ranges):
                        tools.save_in_txt_topcat(row, output_ranges_joined)

            output = cf.sedfitter_output_parameters + "_joined_2.txt"
            for row in list(all_fits):
                tools.save_in_txt_topcat(row, output)

            output = cf.sedfitter_output_parameters + "_joined_param_ranges_2.txt"
            for row in list(all_fits_ranges):
                tools.save_in_txt_topcat(row, output)

    if cf.find_nebula:
        do_nebula_pm_analysis(cf.output_textfile_nebulae, nebulae_data)

    if cf.recalculate_sedfitter_param_ranges:
        from sed_fit_v1 import write_parameters_for_calculated_models

        all_fits_ranges = None
        output_ranges_joined = cf.sedfitter_output_parameters + "_joined_param_range.txt"

        files_to_analyse = tools.get_all_file_names_in_a_folder(cf.recalculate_folder)

        for output_fitinfo in files_to_analyse:
            write_parameters_for_calculated_models(cf.recalculate_folder + output_fitinfo, cf.sedfitter_output_parameters)

            # save all parameter ranges together
            new_fits_ranges = np.loadtxt(cf.sedfitter_output_parameters + "parameters_ranges.txt", dtype=str)

            if all_fits_ranges is None:
                if new_fits_ranges.ndim == 1:
                    all_fits_ranges = [new_fits_ranges]
                else:
                    all_fits_ranges = new_fits_ranges
            elif new_fits_ranges.ndim == 1:
                all_fits_ranges = np.append(all_fits_ranges, [new_fits_ranges], axis=0)
            else:
                all_fits_ranges = np.append(all_fits_ranges, new_fits_ranges, axis=0)

            if new_fits_ranges.ndim == 1:
                tools.save_in_txt_topcat(new_fits_ranges, output_ranges_joined)
            else:
                for row in list(new_fits_ranges):
                    tools.save_in_txt_topcat(row, output_ranges_joined)

        output = cf.sedfitter_output_parameters + "_joined_param_ranges_2.txt"
        for row in list(all_fits_ranges):
            tools.save_in_txt_topcat(row, output)

    print("Done, goodbye")


if __name__ == '__main__':
    main()
