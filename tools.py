import numpy as np
from astropy.io import fits
from os import listdir
from os.path import isfile, join
import time
import config_file as cf
from typing import Union, Tuple
from star_class import Star
import os


def check_if_data_contains_id(data, source_id: str) -> bool:
    """
    Find whether id is located in the FITS table

    :param data: Gaia FITS table from .vots
    :param source_id: ID of the object
    :return: True/False whether object is in the table
    """
    return source_id in (data.field('source_id').astype(str))


def check_if_not_nan_or_zero(value: float) -> bool:
    """
    Return True if not nan or zero

    :param value: Value to check
    :return: If not zero/nan then returns True
    """
    return value != 0 and value != "" and not np.isnan(value) and value != 'nan'


def ang_sep(obj1: list, obj2: list) -> float:
    """
    Takes [ra, dec] object positions and returns angular separation between the objects in degrees

    :param obj1: 1D array [ra, dec] coordinates of first object
    :param obj2: 1D array [ra, dec] coordinates of second object
    :return: Angular separation between two objects in degrees
    """

    ra1 = obj1[0] / 180 * np.pi
    dec1 = obj1[1] / 180 * np.pi
    ra2 = obj2[0] / 180 * np.pi
    dec2 = obj2[1] / 180 * np.pi
    return np.arccos((np.sin(dec1) * np.sin(dec2) + np.cos(dec1) * np.cos(dec2) * np.cos(ra1 - ra2))) / np.pi * 180


def load_data(path_textfile: str) -> np.ndarray:
    """
    Loads data from the path given as input and returns as string 2D array

    :param path_textfile: Path to the textfile
    :return: string based 2D array. E.g. if you want a specific column, use data[:, i]. Specific row: data[i]
    """
    data = np.loadtxt(path_textfile, dtype=str)
    return data


def str_tab(value: str) -> str:
    """
    Converts the value to string, replaces dots by commas and adds tab at the end. Used for saving into txt file

    :param value: Value to be converted
    :return: Cleaned string with tab at the end
    """
    return f"{str(value).replace('.', ',')}\t"


def save_in_txt(text: list, filename: str):
    """
    Saves text in file, separating each element in text by tab and adding a new line below it

    :param text: 1D array with words to write
    :param filename: Path and filename where to save
    :return: Returns nothing, but appends the text file
    """
    with open(filename, 'a+') as f:
        for word in text:
            f.write(str_tab(word))
        f.write('\n')


def str_tab_topcat(value: str) -> str:
    """
    Converts the value to string, replaces commas by dots and adds tab at the end. Used for saving into txt file. Made to be read by TOPCAT (because of dots)

    :param value: Value to be converted
    :return: Cleaned string with tab at the end
    """
    return f"{str(value).replace(',', '.')}\t"


def save_in_txt_topcat(text: list, filename: str):
    """
    Saves text in file, separating each element in text by tab and adding a new line below it. To be read by TOPCAT because saves with dots instead of commas.

    :param text: 1D array with words to write
    :param filename: Path and filename where to save
    """
    with open(filename, 'a+') as f:
        for word in text:
            f.write(str_tab_topcat(word))
        f.write('\n')


def new_txt_topcat(text: list, filename: str):
    """
    Saves text in file, separating each element in text by tab and adding a new line below it. To be read by TOPCAT because saves with dots instead of commas.

    :param text: 1D array with words to write
    :param filename: Path and filename where to save
    """
    with open(filename, 'w') as f:
        for word in text:
            f.write(str_tab_topcat(word))
        f.write('\n')


def load_fits_table(directory: str):
    """
    Loads FITS table.

    :param directory: Path to the file
    :return: data and columns
    """
    hdul = fits.open(directory, memmap=True)
    data = hdul[1].data
    cols = hdul[1].columns
    return data, cols


def get_all_file_names_in_a_folder(path_to_get_files_from: str) -> list:
    """
    Gets a list of all files in a folder

    :param path_to_get_files_from: Folder where to find files
    :return: List of names of files (not paths to them, only names)
    """

    file_names = [f for f in listdir(path_to_get_files_from) if isfile(join(path_to_get_files_from, f))]
    if '.DS_Store' in file_names:
        file_names.remove('.DS_Store')  # Sometimes these get in the way, so try to remove this file
    return file_names


def find_nearest(array: np.ndarray, value: float) -> float:
    """
    Returns value in the array closest to the the requested value

    :param array: Array where to find the value
    :param value: Value to find
    :return: Value that is closest to the requested value
    """
    idx = find_nearest_index(array, value)
    return array[idx]


def find_nearest_index(array: np.ndarray, value: float) -> int:
    """
    Returns location/index of a value in the array closest to the the requested value

    :param array: Array where to find the value
    :param value: Value to find
    :return: Location/index of a value that is closest to the requested value
    """
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx


def time_counter(i: int, time_before: float, total_file_amount: int, recent_time_lengths: list) -> Tuple[str, float, list]:
    """
    Time counter for "expected time left to finish the job"

    :param i: Iteration within a loop
    :param time_before: Time before the job was started (float using time.perf_counter())
    :param total_file_amount: Total amount of files to calculate
    :param recent_time_lengths: The array with seconds in how long it took to do calculations for the last few operations
    :return: Text to print out about time left, time after, list with times taken in seconds
    """
    amount_of_averaged_times = 50

    time_after = time.perf_counter()
    time_difference = abs(time_before - time_after)

    if len(recent_time_lengths) >= amount_of_averaged_times:
        recent_time_lengths.pop(0)
    recent_time_lengths.append(time_difference)

    time_remaining = sum(recent_time_lengths) / len(recent_time_lengths) * (total_file_amount - i)
    time_unit = "sec"
    if time_remaining > 60:
        time_remaining = time_remaining / 60
        time_unit = "min"
    output_time_to_print = f"{int(time_remaining)} {time_unit} remaining"
    time_before = time.perf_counter()
    return output_time_to_print, time_before, recent_time_lengths


def prepare_error_for_weights(error_array: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
    Takes in error array and fixes it so there are no zeroes and converts it to relative error^2.

    :param error_array: The errors values
    :param y: The flux values
    :return: Relative errors without any zeroes
    """
    error_array = np.asarray(error_array)
    mean_error = np.mean(error_array)
    if mean_error == 0:
        return np.ones(np.size(error_array))
    else:
        for i in range(np.size(error_array)):
            if error_array[i] < 10e-30:
                error_array[i] = mean_error

    error_array = error_array / y  # convert to relative error

    return np.square(error_array)


def get_ticks_for_plot(max_x: float) -> list:
    """
    Gets array of ticks for the matplotlib for um scale. Starts with 0.5, 1, 3, 5 and appends in geometric progression until the end

    :param max_x: Maximum value of x to plot
    :return: List of tick arrays
    """
    ticks_array = [0.5, 1, 3, 5]
    new_tick = 10 * 0.000001
    while new_tick < max_x:
        ticks_array.append(int(new_tick * np.power(10, 6)))
        new_tick = new_tick * 2
    return ticks_array


def get_ids_to_analyse(all_ids: np.ndarray, ids_to_find: np.ndarray) -> np.ndarray:
    """
    Finds ids that need to be analysed within all_ids and check if they exist. Returns those ids that do exist

    :param all_ids: list of all ids string
    :param ids_to_find: list if ids to find
    :return: Array that contains all ids that exist in the first one and were needed to be found
    """
    ids_new = np.array([]).astype(str)

    for i in range(np.size(ids_to_find)):
        index_to_use = np.where(all_ids == ids_to_find[i])[0]
        if len(index_to_use) != 0:
            j = index_to_use[0]
            ids_new = np.append(ids_new, str(all_ids[j])).astype(str)

    return ids_new


def get_specific_column_from_data_str(data, field_name: str) -> np.ndarray:
    """
    Finds a specific column from the data and returns it

    :param data: FITS .vot table
    :param field_name: the field to be found
    :return: The whole column from the table
    """
    column = data.field(field_name)
    return column


def find_min_max_x_for_plot(star_obj: Star) -> Tuple[float, float]:
    """
    Finds the min/max value from Star object for plotting. Checks x_good, x_upper and x_viz.

    :param star_obj: Star object
    :return: minimum and maximum x for plotting
    """

    if np.size(star_obj.x_good) != 0:
        x_good_min = np.min(star_obj.x_good)
        x_good_max = np.max(star_obj.x_good)
    else:
        x_good_min = 10000
        x_good_max = 0
    if np.size(star_obj.x_viz) != 0:
        x_viz_min = np.min(star_obj.x_viz)
        x_viz_max = np.max(star_obj.x_viz)
    else:
        x_viz_min = 10000
        x_viz_max = 0
    if np.size(star_obj.x_upper) != 0:
        x_upper_min = np.min(star_obj.x_upper)
        x_upper_max = np.max(star_obj.x_upper)
    else:
        x_upper_min = 10000
        x_upper_max = 0
    min_x = min(x_viz_min, x_good_min, x_upper_min)
    max_x = max(x_viz_max, x_good_max, x_upper_max)
    return min_x, max_x


def conv_wavelength_to_freq(wavelength: float) -> float:
    """
    Convert wavelength to frequency

    :param wavelength: Wavelength in meters
    :return: Frequency in Hz
    """
    return cf.light_speed / wavelength


def conv_freq_to_wavelength(freq: float) -> float:
    """
    Convert frequency to wavelength

    :param freq: Frequency in Hz
    :return: Wavelength in meters
    """
    return cf.light_speed / freq


def compare_two_floats(num1: float, num2: float, relative_err: float) -> bool:
    """
    Compares two floats. If they are close within percentage error returns True. Otherwise return False.

    :param num1: First float
    :param num2: Second float
    :param relative_err: Relative error. I.e. 0.01 means 1% error
    :return: True/False if numbers are almost same/not same
    """
    if abs(1 - num1 / num2) < relative_err:
        return True
    else:
        return False


def take_within_x_boundaries(x: np.ndarray, y: np.ndarray, error: np.ndarray, x_min: float, x_max: float,
                             minimum_points: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Takes x values from within boundaries. If it has less than minimum points, returns original arrays

    :param x: x values
    :param y: y values
    :param error: error values
    :param x_min: Minimum x
    :param x_max: Maximum x
    :param minimum_points: Minimum points to not return original arrays
    :return: New arrays
    """
    if x_min > x_max:
        x_min, x_max = x_max, x_min

    x_new, y_new, error_new = [], [], []
    for i in range(np.size(x)):
        if x_min <= x[i] <= x_max:
            x_new.append(x[i])
            y_new.append(y[i])
            error_new.append(error[i])
    x_new, y_new, error_new = np.asarray(x_new), np.asarray(y_new), np.asarray(error_new)
    if np.size(x_new) < minimum_points:
        x_new = np.copy(x)
        y_new = np.copy(y)
        error_new = np.copy(error)
    return x_new, y_new, error_new


def find_filter(wavelength: float) -> str:
    """
    Finds the filter's name based on the wavelength. Supports GDR2, GEDR3, WISE/ALLWISE, IRAC, MIPS, PACS70-100

    :param wavelength: Wavelength of the filter in meters
    :return: Name of the filter
    """
    percentage_error = 0.00001

    wavelengths_array = np.array([cf.g_dr2, cf.bp_dr2, cf.rp_dr2, cf.g_edr3, cf.bp_edr3, cf.rp_edr3,
                                  cf.W1, cf.W2, cf.W3, cf.W4, cf.j_2mass, cf.h_2mass, cf.k_2mass,
                                  cf.irac1, cf.irac2, cf.irac3, cf.irac4, cf.mips1, cf.mips2, cf.mips3,
                                  cf.iras12, cf.iras25, cf.pacs70, cf.pacs100])

    filter_names = np.array(
        ["GDR2_G", "GDR2_BP", "GDR2_RP", "GEDR3_G", "GEDR3_BP", "GEDR3_RP", 'WISE1', 'WISE2', 'WISE3', 'WISE4',
         "2J", "2H", "2K", "I1", "I2", "I3", "I4", "M1", "M2", "M3", "I12", "I25", "PACS1", "PACS2"])

    index_with_wavelength = find_nearest_index(wavelengths_array, wavelength)
    if compare_two_floats(wavelength, wavelengths_array[index_with_wavelength], percentage_error):
        return filter_names[index_with_wavelength]
    else:
        raise ValueError('Unknown wavelength input')


def get_sedfitter_star_data(star_obj: Star, filters_total: list) -> np.ndarray:
    """
    Sorts and converts star's data to SEDFitter ready to use data. Filters to find are given in filters_total according to names and order. 0-s are given if no filter is found.

    :param star_obj: Star class object
    :param filters_total: List of filter names to find. Order of output will be the same as order of this parameter
    :return: SEDFitter ready to use data: ID Ra Dec FILTER_FLAG Values (mJy) Error (mJy)
    """
    from sed_linefit_v4_0 import prepare_data_for_interpolate

    star_info = np.array([star_obj.source_id, star_obj.ra, star_obj.dec])
    star_x, star_y, star_error = prepare_data_for_interpolate(star_obj.x_good, star_obj.y_good, star_obj.error_good)

    star_x_upper, star_y_upper, star_error_upper = prepare_data_for_interpolate(star_obj.x_upper, star_obj.y_upper,
                                                                                star_obj.error_upper)

    # Define filters
    filter_flags = []
    flux_with_error = []

    star_x_filter_names = []

    for j in range(np.size(star_x)):  # Convert x-parameters to filter names
        star_x_filter_names.append(find_filter(star_x[j]))

    star_x_upper_filter_names = []
    for j in range(np.size(star_x_upper)):  # Convert x-parameters for upper limits to filter names
        star_x_upper_filter_names.append(find_filter(star_x_upper[j]))

    for i in range(len(filters_total)):
        if filters_total[i] in star_x_filter_names:  # If filter exists
            index = star_x_filter_names.index(filters_total[i])
            filter_flags.append(1)  # 1 means good data. Convert data from W/m^2 to mJy
            flux_with_error.append(
                star_y[index] / np.power(10.0, -26) / conv_wavelength_to_freq(star_x[index]) * np.power(10.0, 3))
            flux_with_error.append(
                star_error[index] / np.power(10.0, -26) / conv_wavelength_to_freq(star_x[index]) * np.power(10.0, 3))
        elif filters_total[i] in star_x_upper_filter_names:
            index = star_x_upper_filter_names.index(filters_total[i])
            filter_flags.append(3)  # 3 means upper data. Convert data from W/m^2 to mJy
            flux_with_error.append(
                star_y_upper[index] / np.power(10.0, -26) / conv_wavelength_to_freq(star_x_upper[index]) * np.power(
                    10.0, 3))
            flux_with_error.append(cf.sedfitter_upper_limit_confident)
        else:
            filter_flags.append(0)  # No filter: use 0 as "no such data"
            flux_with_error.append(0)
            flux_with_error.append(0)

    star_info = np.append(star_info, filter_flags)
    star_info = np.append(star_info, flux_with_error)
    star_info = star_info.flatten()
    return star_info


def create_directory(directory_path: str):
    """
    Creates a directory if it does not exist

    :param directory_path: The path to directory, ending with a "/"
    """
    if not os.path.exists(directory_path):
        os.mkdir(directory_path)


def create_all_directories():
    """
    Creates all directories (if needed) used in this project. Requires to manually input directiors into this function.
    """
    directories = [cf.output_periodogram_gaia_g_png, cf.output_periodogram_gaia_g_pdf, cf.output_light_curve_g_band_png,
                   cf.output_light_curve_g_band_pdf, cf.output_raw_data_g_band_png, cf.output_raw_data_g_band_pdf,
                   cf.output_folded_light_curve_png, cf.output_folded_light_curve_pdf,
                   cf.output_rp_raw_data_png, cf.output_rp_raw_data_pdf,
                   cf.output_bp_raw_data_png, cf.output_bp_raw_data_pdf, cf.output_ztf_lightcurves_pictures,
                   cf.output_sed_ir_slope_fit_png, cf.output_sed_ir_slope_fit_pdf,
                   cf.output_sed_integrated_excess_figure_png, cf.output_sed_integrated_excess_figure_pdf,
                   cf.output_sed_fit_png, cf.output_sed_fit_pdf, cf.output_sed_bar_png, cf.output_sed_bar_pdf,
                   cf.output_sed_temp_pic, cf.output_pdf_files, cf.textfiles_output_folder, cf.temp_path,
                   cf.output_gaia_all_bands_raw_data_png, cf.output_ztf_fit, cf.output_ztf_folded,
                   cf.output_periodogram_ztf_g_png, cf.output_periodogram_ztf_g_pdf,
                   cf.output_periodogram_ztf_r_png, cf.output_periodogram_ztf_r_pdf,
                   cf.xmatched_new_catalogues_directory, cf.output_frequency_periodogram_gaia_g_png,
                   cf.output_frequency_periodogram_gaia_g_pdf, cf.output_frequency_periodogram_ztf_g_png,
                   cf.output_frequency_periodogram_ztf_g_pdf, cf.output_frequency_periodogram_ztf_r_png,
                   cf.output_frequency_periodogram_ztf_r_pdf, cf.output_multiband_frequency_periodogram_png,
                   cf.output_multiband_fits_png]

    for directory in directories:
        create_directory(directory)


def identify_yso_class(star_slope: float) -> str:
    """
    Classify the class of the object based on slope value. All classes can be changed in config_file if needed

    :param star_slope: float value of a slope
    :return: string name of the slope
    """
    #2D array. Each component: first name, then at what slope the classification starts (it ends wherever next class starts)
    yso_classes = cf.yso_classes

    yso_classes = np.array(yso_classes)
    yso_class_names = yso_classes[:, 0]
    yso_class_slope_start = yso_classes[:, 1].astype(float)

    sorted_indices = yso_class_slope_start.argsort()  # Sorts indices to be in ascending order
    yso_class_names = yso_class_names[sorted_indices]
    yso_class_slope_start = yso_class_slope_start[sorted_indices]

    closest_index = find_nearest_index(yso_class_slope_start, star_slope)
    if star_slope < yso_class_slope_start[closest_index] and closest_index != 0:
        closest_index = closest_index - 1

    return yso_class_names[closest_index]


def linear_best_fit(x: list, y: list) -> Tuple[float, float]:
    """
    Find linear best fit line using least-squared pure python implementation

    :param x: x-values
    :param y: y-values
    :return: c, m float values, where best-fit line is y=mx+c
    """
    xbar = sum(x) / len(x)
    ybar = sum(y) / len(y)
    n = len(x)

    numer = sum([xi * yi for xi, yi in zip(x, y)]) - n * xbar * ybar
    denum = sum([xi ** 2 for xi in x]) - n * xbar ** 2

    m = numer / denum
    c = ybar - m * xbar

    return c, m
