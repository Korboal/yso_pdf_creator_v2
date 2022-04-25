import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from lmfit import Model, Parameters, Parameter, minimize
from astropy.io.votable import parse_single_table
import tools
from tools import conv_wavelength_to_freq, compare_two_floats
from get_gaia_table import find_index_with_id
import config_file as cf
import scipy
from scipy import interpolate
from scipy import integrate
from collections import defaultdict
from star_class import Star
from typing import Union, Tuple
from xmatched_tables_class import XMatchTables
import os


def model_minimize_ir_slope_log_space(pars: Parameters, x: np.ndarray, y: np.ndarray, error: np.ndarray) -> float:
    """
    Linear fit y=m*x+c Model, where it is attempted to minimize the error, between y±error and the slope

    :param pars: Parameters of the linear fit (should contain m, c parameters + guess + boundaries)
    :param x: Wavelength in log(meters)
    :param y: Flux in log(W/m^2)
    :param error: Error of flux in log(W/m^2)
    :return: Error of each individual point
    """
    vals = pars.valuesdict()
    m = vals['m']
    c = vals['c']

    y_slope = x * m + c  # Model y fit

    y_pos_err = y + error   # Take into account errors for y
    y_neg_err = y - error

    diff1 = y_slope - y_pos_err     # See difference between the fit and y±error
    diff2 = y_slope - y_neg_err

    diff_multiplied = diff1 * diff2     # Multiply the difference. -1 means that slope within error bars
    indices_slope_within_error = np.where(diff_multiplied < 0)

    minimum_diff = np.amin(np.array([np.abs(diff1), np.abs(diff2)]), axis=0)    # Finds minimum distance to the slope
    if np.size(indices_slope_within_error) > 0:     # If error bar within slope, then error = 0 for that point
        minimum_diff[indices_slope_within_error] = np.zeros(np.size(minimum_diff[indices_slope_within_error]))

    return minimum_diff


def minimize_test(x: np.ndarray, y: np.ndarray, error: np.ndarray) -> minimize:
    """
    Slope of SED in log vF(v) vs lambda space: y = m * x + c

    :param x: Wavelength in meters
    :param y: Flux vF(v)
    :param error: flux error
    :return: Output of minimizing
    """
    error = 1 / np.log(10) * error / y
    x = np.log10(x)
    y = np.log10(y)
    params = get_param_for_sed_linear_slope(x, y)
    out = minimize(model_minimize_ir_slope_log_space, params, args=(x,), kws={'y': y, 'error': error})
    return out


def model_function_rayleigh_jeans_in_linear_space(x: Union[float, np.ndarray], c: float) -> Union[float, np.ndarray]:
    """
    Rayleigh Jeans approximation in linear vF(v) vs lambda space: y = x^(-3) * 10^(-c)

    :param x: Wavelength in meters
    :param c: Constant for the best fit, expected to be positive
    :return: Depending on the type of x, returns the y-values of the function
    """
    values = np.power(x, cf.line_fit_power_rayleigh_jeans) * np.power(10.0, -c)
    return values


def model_function_rayleigh_jeans_in_log_space(x: Union[float, np.ndarray], c: float) -> Union[float, np.ndarray]:
    """
    Rayleigh Jeans approximation in log vF(v) vs lambda space: y = -3 * x - c

    :param x: Wavelength in log(meters)
    :param c: Constant for the best fit, expected to be positive
    :return: Depending on the type of x, returns the y-values of the function
    """
    values = cf.line_fit_power_rayleigh_jeans * x - c
    return values


def model_function_sed_slope_in_linear_space(x: Union[float, np.ndarray], m: float, c: float) -> Union[
    float, np.ndarray]:
    """
    Slope of SED in linear vF(v) vs lambda space: y = x^(-m) * 10^(-c)

    :param x: Wavelength in log(meters)
    :param m: Slope of the SED fit; positive or negative
    :param c: Constant for the best fit; positive or negative
    :return: Depending on the type of x, returns the y-values of the function
    """
    values = np.power(x, m) * np.power(10.0, c)
    return values


def model_function_sed_slope_in_log_space(x: Union[float, np.ndarray], m: float, c: float) -> Union[float, np.ndarray]:
    """
    Slope of SED in log vF(v) vs lambda space: y = m * x + c

    :param x: Wavelength in log(meters)
    :param m: Slope of the SED fit; positive or negative
    :param c: Constant for the best fit; positive or negative
    :return: Depending on the type of x, returns the y-values of the function
    """
    values = m * x + c
    return values


def get_model_linear_sed_fit_rayleigh_jeans() -> Model:
    """
    Returns model of the Rayleigh Jeans approximation in log vF(v) vs lambda space: y = -3 * x - c

    :return: Rayleigh Jeans approximation Model from lmfit package
    """
    return Model(model_function_rayleigh_jeans_in_log_space, independent_vars=['x'])


def get_model_linear() -> Model:
    """
    Returns model of the slope of SED in log vF(v) vs lambda space: y = m * x + c

    :return: Linear slope of SED in log vF(v) Model from lmfit package
    """
    return Model(model_function_sed_slope_in_log_space, independent_vars=['x'])


def get_param_for_rayleigh_jeans(x: np.ndarray, y: np.ndarray) -> Parameters:
    """
    Gives a guess for the parameter for lmfit of the Rayleigh Jeans approximation function

    :param x: Wavelengths in meters
    :param y: vF(v) in W/m^2
    :return: lmfit Parameter with guess and bounds
    """
    guess_bounds = 20  # How much the parameter is expected to vary by. 20 is quite generous
    if np.size(x) == 1:
        guess = -(np.log10(y[0]) - np.log10(x[0]) * cf.line_fit_power_rayleigh_jeans)  # c value is expected to be negative
    else:
        x, y = np.log10(x), np.log10(y)
        index = tools.find_nearest_index(x, cf.sed_linear_line_wavelength_start)  # Guess based on the first point that is fitted
        guess = -(y[index] - x[index] * cf.line_fit_power_rayleigh_jeans)  # c value is expected to be negative
    pars = Parameters()
    c = Parameter('c', value=guess, min=guess - guess_bounds, max=guess + guess_bounds)
    pars.add_many(c)
    return pars


def get_param_for_sed_linear_slope(x: np.ndarray, y: np.ndarray, ir_slope_end=cf.ir_slope_end) -> Parameters:
    """
    Gives a guess for the parameter for lmfit of the SED linear fit approximation function

    :param x: log of wavelengths in meters
    :param y: log of vF(v) in W/m^2
    :param ir_slope_end: where to end the slope
    :return: lmfit Parameter with guess and bounds
    """
    guess_bounds = 40  # 40 is quite a generous bound
    pars = Parameters()

    index_start = tools.find_nearest_index(x, np.log10(cf.ir_slope_start))  # np.log10(0.35 * pow(10, -6)))
    index_end = tools.find_nearest_index(x, np.log10(ir_slope_end))  # np.log10(0.8 * pow(10, -6))) #

    y1 = y[index_start]  # Guesses based on the first and last points using simple linear fit through 2 points
    y2 = y[index_end]
    x1 = x[index_start]
    x2 = x[index_end]

    guess_slope = (y1 - y2) / (x1 - x2)
    guess_constant = (x1 * y2 - x2 * y1) / (x1 - x2)

    #m = Parameter('m', value=guess_slope, min=guess_slope - guess_bounds, max=guess_slope + guess_bounds)
    #c = Parameter('c', value=guess_constant, min=guess_constant - guess_bounds, max=guess_constant + guess_bounds)
    #pars.add_many(m, c)

    pars.add('m', value=guess_slope, min=guess_slope - guess_bounds, max=guess_slope + guess_bounds)
    pars.add('c', value=guess_constant, min=guess_constant - guess_bounds, max=guess_constant + guess_bounds)
    return pars


def do_model_fitting_log_space(x: np.ndarray, y: np.ndarray, error: np.ndarray, mod: Model,
                               param: Parameters) -> Model.fit:
    """
    Fit the given values using linear fit function

    :param x: Wavelength values in meters
    :param y: Flux vF(v) in W/m^2
    :param error: Errors of flux in W/m^2
    :param mod: lmfit Model to use for fitting
    :param param: lmfit Parameters to use during fitting
    :return: output from lmfit mod.fit
    """
    error = tools.prepare_error_for_weights(error, y)  # Convert to errors that can be used for weights

    x = np.log10(x)
    y = np.log10(y)

    out = mod.fit(y, params=param, x=x, method="least_squares", weights=np.ones(np.size(x)))
    return out


def dlt_elem_by_indices(array: np.ndarray, indices: np.ndarray) -> np.ndarray:
    """
    Deletes elements from array that are located at indicated indices array

    :param array: Array from where to delete the indices
    :param indices: Indices to delete from the original array
    :return: Reduced array
    """
    return np.delete(array, indices)


def three_arr_dlt_elem_by_indices(arr1: np.ndarray, arr2: np.ndarray, arr3: np.ndarray, indices: np.ndarray) -> Tuple[
    np.ndarray, np.ndarray, np.ndarray]:
    """
    Deletes elements from three arrays that are located at the indicated indices array. All arrays lose elements at the same indices

    :param arr1: Array from where to delete the indices
    :param arr2: Array from where to delete the indices
    :param arr3: Array from where to delete the indices
    :param indices: Indices to delete from the original arrays
    :return: Three reduced array
    """
    arr1 = dlt_elem_by_indices(arr1, indices)
    arr2 = dlt_elem_by_indices(arr2, indices)
    arr3 = dlt_elem_by_indices(arr3, indices)
    return arr1, arr2, arr3


def get_one_var_from_fields(data, index: int, field_name: str, var_type) -> Union[int, str, float]:
    """
    Returns one value from the data given its field name and index as var_type

    :param data: FITS table data that was extracted from a votable
    :param index: Index at which the variable is located
    :param field_name: The name of the field of the variable in the data table
    :param var_type: Type of the array to return
    :return: Variable at index in field_name as type the same as var_type
    """
    try:
        return data.field(field_name)[index].astype(var_type)
    except:  # sometimes breaks for strings, so "temporary" is done like this:
        return data.field(field_name)[index]


def convert_vizier_votable_to_data(file: str) -> Tuple[
    np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Converts Vizier .vot votable data to separated numpy arrays

    :param file: File path
    :return: Frequency (GHz), Flux (Jy), Flux_error (Jy), Table name, ID/coord/target within table, filter
    """
    votable = parse_single_table(file)
    # ra = votable.array["_RAJ2000"]
    # dec = votable.array["_DEJ2000"]
    tabname = votable.array["_tabname"]  # String
    id_ = votable.array["_ID"]  # String
    # time = votable.array["_tab1_31"].astype(float)    # Does not exist in every table, so will break stuff if used
    # time_err = votable.array["_tab1_32"].astype(float)  # Does not exist in every table, so will break stuff if used
    freq = votable.array["sed_freq"].astype(float)
    flux = votable.array["sed_flux"].astype(float)
    flux_err = votable.array["sed_eflux"].astype(float)
    sed_filter = votable.array["sed_filter"]  # String

    return freq.filled(0), flux.filled(0), flux_err.filled(
        0.00000001), tabname, id_, sed_filter  # If freq/flux/error do not exist, return 0 or minimal value


def remove_specific_filter(catalogues: np.ndarray, error: np.ndarray, sed_filter: np.ndarray, filter_to_remove: str,
                           x: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray,
                                                                  np.ndarray]:
    """
    Removes specific filter from all x/y values

    :param catalogues: array of strings catalogues corresponding to x/y
    :param error: array of floats errors corresponding to x/y
    :param sed_filter: array of strings sed_filter names corresponding to x/y
    :param filter_to_remove: string of filter name to remove
    :param x: array of floats wavelengths in meters
    :param y: array of floats flux or mag
    :return: original 5 arrays (with each value corresponding to one another), but with a specific filter removed
    from all arrays
    """
    indices_to_remove = np.where(sed_filter == filter_to_remove)[0]
    x, y, error = three_arr_dlt_elem_by_indices(x, y, error, indices_to_remove)
    catalogues = dlt_elem_by_indices(catalogues, indices_to_remove)
    sed_filter = dlt_elem_by_indices(sed_filter, indices_to_remove)
    return catalogues, error, sed_filter, x, y


def clean_sed_points(vot_table_path: str, object_id: str, x_matched_tables: XMatchTables) -> Tuple[
    np.ndarray, np.ndarray, np.ndarray, np.ndarray,
    np.ndarray, np.ndarray, np.ndarray, np.ndarray,
    np.ndarray, list, list]:
    """
    Takes in vot table and corresponding object_id. Loads data from Vizier and other x-matched catalogues. Then it cleans the data and separates into Vizier data, good data and data with upper limits. Bad data is completely rejected and deleted.

    :param vot_table_path: File path to the vot table
    :param object_id: Object id to analyse
    :param x_matched_tables: The x-matched catalogues
    :return: 11 arrays: wavelength_good_data, magnitude_good_data, error_good_data, same with Vizier data, same with "upper" limit data, array with separations to different databases, array with corresponding names of databases
    """
    if cf.use_vizier_data and os.path.isfile(vot_table_path):
        freq, y, error, tabname, id_, sed_filter = convert_vizier_votable_to_data(vot_table_path)

        x = cf.light_speed / freq / pow(10, 9)  # Convert GHz frequency to wavelength
        y = y * pow(10, -26) * pow(10, 9) * freq  # Convert flux (F(v)) from Jansky to W/m^2 (vF(v) = lambda * F(lambda))
        error = error * freq * pow(10, -26) * pow(10, 9)  # Convert error from Jansky to W/m^2

        catalogues = []
        for i in range(np.size(tabname)):  # Properly extract catalogue names
            catalogues.append(tabname[i].split('/')[-1])
        catalogues = np.asarray(catalogues)

        # remove flux above "start of MIR"
        indices_to_remove = np.where(x >= cf.ir_excess_start)[0]
        if np.size(indices_to_remove > 0):
            x, y, error = three_arr_dlt_elem_by_indices(x, y, error, indices_to_remove)
            catalogues = dlt_elem_by_indices(catalogues, indices_to_remove)
            sed_filter = dlt_elem_by_indices(sed_filter, indices_to_remove)

        # remove TIC, because they are not always fully reliable and usually duplicate of something else
        tic_indices = np.where(catalogues == "tic")[0]
        if np.size(tic_indices > 0):
            x, y, error = three_arr_dlt_elem_by_indices(x, y, error, tic_indices)
            catalogues = dlt_elem_by_indices(catalogues, tic_indices)
            sed_filter = dlt_elem_by_indices(sed_filter, tic_indices)

        # remove all WISE, Spitzer, 2MASS, GAIA, IRAS
        for filter_to_remove in ["WISE:W1", "WISE:W2", "WISE:W3", "WISE:W4", "Spitzer/IRAC:3.6", "Spitzer/IRAC:4.5",
                                 "Spitzer/IRAC:5.8", "Spitzer/IRAC:8.0", "Spitzer/MIPS:24", "Spitzer/MIPS:70",
                                 "Spitzer/MIPS:160", "2MASS:Ks", "2MASS:J", "2MASS:H", "IRAS:12", "IRAS:25", "IRAS:60",
                                 "IRAS:100", "GAIA/GAIA2:Grp", "GAIA/GAIA2:G", "GAIA/GAIA2:Gbp", "GAIA/GAIA3:Grp",
                                 "GAIA/GAIA3:G", "GAIA/GAIA3:Gbp"]:
            catalogues, error, sed_filter, x, y = remove_specific_filter(catalogues, error, sed_filter, filter_to_remove, x,
                                                                         y)

        # remove duplicate entries, i.e. x, y and error are the same
        data = np.vstack((x, y, error)).T
        new_array = [tuple(row) for row in data]
        uniques = np.unique(new_array, axis=0)
        x, y, error = np.split(uniques, 3, axis=1)
        x, y, error = x.flatten(), y.flatten(), error.flatten()

        x_viz, y_viz, error_viz = np.asarray(x), np.asarray(y), np.asarray(error)
        x_good, y_good, error_good = np.array([]), np.array([]), np.array([])
        # x_good, y_good, error_good = x, y, error  # If want to use Vizier for fits as well, uncomment
    else:
        x_good, y_good, error_good = np.array([]), np.array([]), np.array([])
        x_viz, y_viz, error_viz = np.array([]), np.array([]), np.array([])

    x_upper, y_upper, error_upper = np.array([]), np.array([]), np.array([])

    # Add other catalogues to x_good etc
    separation_arcsec = []
    separation_table = []
    extra_catalogues_dict = x_matched_tables.extra_catalogues_dict
    flux_name = "_flux"
    flux_err_name = "_err"
    flag_name = "_flag"
    wavelength_name = "_wave"
    prefactor_name = "_pref"
    flux_unit_name = "_flux_unit"

    for i in range(len(x_matched_tables.table_names)):  # load each individual table for a specific star
        table = x_matched_tables.table_names[i]
        vot_table_data = x_matched_tables.all_table_data[i]
        #vot_table_data, _ = tools.load_fits_table(extra_catalogues_dict[table])
        if tools.check_if_data_contains_id(vot_table_data, object_id):  # If table contains the object_id, i.e. star
            separation_table.append(table[:-6].replace("_", " "))  # Separation table takes in name of the table minus its xmatch size
            separation_arcsec.append(get_one_var_from_fields(
                vot_table_data, find_index_with_id(vot_table_data, object_id), "Separation", float))
            names_flux = extra_catalogues_dict[table + flux_name]
            names_error = extra_catalogues_dict[table + flux_err_name]
            names_flag = extra_catalogues_dict[table + flag_name]
            wavelengths = extra_catalogues_dict[table + wavelength_name]
            prefactor = extra_catalogues_dict[table + prefactor_name]
            flux_unit = extra_catalogues_dict[table + flux_unit_name]
            error_good, error_upper, x_good, x_upper, y_good, y_upper = get_sed_data_points_from_table(vot_table_data,
                                                                                                       object_id,
                                                                                                       names_error,
                                                                                                       names_flag,
                                                                                                       names_flux,
                                                                                                       flux_unit,
                                                                                                       prefactor,
                                                                                                       wavelengths,
                                                                                                       x_good, y_good,
                                                                                                       error_good,
                                                                                                       x_upper, y_upper,
                                                                                                       error_upper)

    # Completely remove negative flux (or very small flux)
    minimum_flux = np.power(10.0, -50)

    indices_to_remove = np.where(y_good <= minimum_flux)[0]
    if np.size(indices_to_remove > 0):
        x_good, y_good, error_good = three_arr_dlt_elem_by_indices(x_good, y_good, error_good, indices_to_remove)

    indices_to_remove = np.where(y_upper <= minimum_flux)[0]
    if np.size(indices_to_remove > 0):
        x_upper, y_upper, error_upper = three_arr_dlt_elem_by_indices(x_upper, y_upper, error_upper, indices_to_remove)

    indices_to_remove = np.where(y_viz <= minimum_flux)[0]
    if np.size(indices_to_remove > 0):
        x_viz, y_viz, error_viz = three_arr_dlt_elem_by_indices(x_viz, y_viz, error_viz, indices_to_remove)

    return x_good, y_good, error_good, x_upper, y_upper, error_upper, x_viz, y_viz, error_viz, separation_arcsec, \
           separation_table


def get_sed_data_points_from_table(vot_table_data, object_id: str, names_error: list, names_flag: list,
                                   names_flux: list, flux_unit: bool, prefactor: float, wavelengths: list,
                                   x_good: np.ndarray, y_good: np.ndarray, error_good: np.ndarray, x_upper: np.ndarray,
                                   y_upper: np.ndarray, error_upper: np.ndarray) -> Tuple[
                                   np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Takes in data from the extra table and adds it to the old data

    :param vot_table_data: FITS data from specific .vot table
    :param object_id: object ID
    :param names_error: Names of the errors field in the catalogue
    :param names_flag: Names of the flags field in the catalogue
    :param names_flux: Names of the flux field in the catalogue
    :param flux_unit: True if flux, False if magnitude
    :param prefactor: Prefactor of flux
    :param wavelengths: Wavelengths of the corresponding fluxes
    :param x_good: Old array with good parameters
    :param y_good: Old array with good parameters
    :param error_good: Old array with good parameters
    :param x_upper: Old array with upper limit parameters
    :param y_upper: Old array with upper limit parameters
    :param error_upper: Old array with upper limit parameters
    :return: 6 1D appended arrays of original x,y,error data in the following order: error_good, error_upper, x_good, x_upper, y_good, y_upper
    """

    flux, error, flag = get_flux_with_error_and_flag(vot_table_data, names_flux, names_error, names_flag, object_id,
                                                     wavelengths, prefactor, flux_unit)
    x_good_new, y_good_new, err_good_new, x_up_new, y_up_new, err_up_new = sort_flux(wavelengths, flux, error, flag)

    if len(x_good_new) != 0:
        x_good = np.append(x_good, x_good_new)
        y_good = np.append(y_good, y_good_new)
        error_good = np.append(error_good, err_good_new)
    if len(x_up_new) != 0:
        x_upper = np.append(x_upper, x_up_new)
        y_upper = np.append(y_upper, y_up_new)
        error_upper = np.append(error_upper, err_up_new)

    return error_good, error_upper, x_good, x_upper, y_good, y_upper


def get_flux_with_error_and_flag(data, names_flux: list, names_error: list, names_flag: list, source_id: str,
                                 wavelengths: list, freq_prefactor: float, flux_unit: bool) -> Tuple[list, list, list]:
    """
    Takes data and loads it from data, removing data with bad flags and giving appropriate flags/error to each. Magnitude is converted to flux

    :param data: FITS data from specific .vot table
    :param names_flux: Names of the flux field in the catalogue
    :param names_error: Names of the errors field in the catalogue
    :param names_flag: Names of the flags field in the catalogue
    :param source_id: object ID
    :param wavelengths: Wavelengths of the corresponding fluxes
    :param freq_prefactor: Prefactor of flux
    :param flux_unit: True if flux, False if magnitude
    :return: Three arrays: flux (W/m^2), errors (W/m^2), flags
    """
    index = find_index_with_id(data, str(source_id))
    fluxes = []
    errors = []
    flags = []

    length_of_flags = len(names_flag)
    if length_of_flags == 1:
        flags_combined = True  # If flags are combined into 1 string with all flags
    else:
        if length_of_flags == len(names_flux):
            flags_combined = False  # If flags are in separate columns for each flux value
        elif length_of_flags == 0:
            pass    # no flags, ignore
        else:
            raise ValueError("Flag parameter cannot be interpreted. Expected same amount of flags as fluxes or one "
                             "combined flag")

    for i in range(len(names_flux)):
        flux_temp = get_one_var_from_fields(data, index, names_flux[i], float)
        flux_temp_og = np.copy(flux_temp)
        if tools.check_if_not_nan_or_zero(flux_temp):  # If flux exists at all and not zero
            if flux_unit:  # If Jansky/F(v), convert to vF(v)
                flux_temp = flux_temp * pow(10, -26) * conv_wavelength_to_freq(wavelengths[i]) * freq_prefactor
            else:  # Otherwise convert magnitude to vF(v)
                flux_temp = convert_mag_to_flux(flux_temp, wavelengths[i], freq_prefactor)
            fluxes.append(flux_temp)  # Add flux to new flux
            error_temp = get_one_var_from_fields(data, index, names_error[i], float)
            if tools.check_if_not_nan_or_zero(error_temp):  # If error exists
                if flux_unit:  # If Jansky/F(v), convert to vF(v)
                    error_temp = error_temp * pow(10, -26) * conv_wavelength_to_freq(wavelengths[i]) * freq_prefactor
                else:  # Otherwise convert magnitude to vF(v)
                    error_temp = conv_mag_error_to_flux(flux_temp_og, error_temp, wavelengths[i], freq_prefactor)
                errors.append(abs(error_temp))
            else:  # If error does not exist, append 0
                errors.append(0)
            if length_of_flags != 0:  # If flag exists
                if flags_combined:  # If combined flags, then add them symbol by symbol
                    flags.append(list(get_one_var_from_fields(data, index, names_flag[0], str))[i])
                else:  # If separate flags, add only appropriate one for the flux
                    flags.append(get_one_var_from_fields(data, index, names_flag[i], str))
            else:  # If no flag, assume that data is OK
                flags.append("A")
        else:
            fluxes.append(0)  # Flag to remove data completely later on
            errors.append(0)  # Because flux does not exist now
            flags.append("X")

    return fluxes, errors, flags


def sort_flux(wavelengths: list, flux: list, err_flux: list, flags: list) -> Tuple[list, list, list, list, list, list]:
    """
    Sorts flux into good/upper flux depending on flags. Flag: "U" or "3" or ">" means upper. "X", "S", "4", "N" means that data is removed/rejected. Everything else is put into good data.

    :param flux: Flux vF(v) in W/m^2
    :param err_flux: Errors of flux vF(v) in W/m^2
    :param flags: Flags to corresponding fluxes
    :param wavelengths: Wavelengths of fluxes in meters
    :return: 6 arrays sorted whether good or upper limits. Order: x_good, y_good, err_good, x_up, y_up, error_up
    """
    wavelengths, flux, err_flux, flags = np.asarray(wavelengths), np.asarray(flux), np.asarray(err_flux), np.asarray(
        flags)

    # Remove all fluxes that are <= 0 or NaN, flags that are "X", "N", "S", "4"
    indices_to_use = np.invert(
        np.logical_or.reduce((flux <= 0, np.isnan(flux), flags == "X", flags == "N", flags == "S", flags == "4")))
    wavelengths, flux, err_flux, flags = wavelengths[indices_to_use], flux[indices_to_use], err_flux[indices_to_use], \
                                         flags[indices_to_use]

    # Upper limit flags that are noted as "U", "3" or ">"
    indices_upper_limit = np.logical_or.reduce((flags == "U", flags == "3", flags == ">"))
    x_up, y_up, error_up = list(wavelengths[indices_upper_limit]), list(flux[indices_upper_limit]), list(
        err_flux[indices_upper_limit])

    # Remaining data is assumed to be good
    indices_good = np.invert(indices_upper_limit)
    x_good, y_good, error_good = list(wavelengths[indices_good]), list(flux[indices_good]), list(err_flux[indices_good])

    """
    for i in range(len(flux)):
        if tools.check_if_not_nan_or_zero(flux[i]) and flux[i] > 0:  # If flux exists
            if flags[i] == "U" or flags[i] == 3 or flags[i] == "3" or flags[i] == ">":  # Upper flags
                x_up.append(wavelengths[i])
                y_up.append(flux[i])
                if tools.check_if_not_nan_or_zero(err_flux[i]):  # If error exists
                    error_up.append(err_flux[i])
                else:  # Otherwise assume error is 0
                    error_up.append(0)
            elif flags[i] == "X" or flags[i] == "N" or flags[i] == "S" or flags[i] == 4 or flags[i] == "4":  # Flag for bad data
                pass
            else:  # Otherwise assume that data is OK
                x_good.append(wavelengths[i])
                y_good.append(flux[i])
                if tools.check_if_not_nan_or_zero(err_flux[i]):
                    error_good.append(err_flux[i])
                else:
                    error_good.append(0)"""

    return x_good, y_good, error_good, x_up, y_up, error_up


def conv_mag_error_to_flux(mag: float, error: float, wavelength: float, prefactor: float) -> float:
    """
    Converts magnitude error to flux error

    :param mag: Magnitude
    :param error: Magnitude error
    :param wavelength: Wavelength in meters
    :param prefactor: Prefactor of magnitude, if it exists (i.e. milli/micro would be 1e-3, 1e-6)
    :return: Error in units of W/m^2
    """
    flux = convert_mag_to_flux(mag, wavelength, prefactor)
    neg_err = abs(convert_mag_to_flux(mag - error, wavelength, prefactor) - flux)
    pos_err = abs(convert_mag_to_flux(mag + error, wavelength, prefactor) - flux)

    return max(neg_err, pos_err)


def convert_mag_to_flux(mag: float, wavelength: float, prefactor: float) -> float:
    """
    Converts magnitude to flux using values provided in catalogues. Generally does not take into account reddening. Uses wavelength to guess the filter. If wavelength is unknown, an error will occur.

    :param mag: Magnitude of the filter
    :param wavelength: Wavelength of the filter in meters
    :param prefactor: Prefactor of the magnitude (usually 1)
    :return: Flux in W/m^2
    """
    percentage_error = 0.00001
    flux = np.power(10.0, -mag * prefactor / 2.5) * pow(10, -26) * conv_wavelength_to_freq(wavelength)

    wavelengths_array = np.array([cf.g_dr2, cf.bp_dr2, cf.rp_dr2, cf.g_edr3, cf.bp_edr3, cf.rp_edr3,
                                  cf.W1, cf.W2, cf.W3, cf.W4, cf.j_2mass, cf.h_2mass, cf.k_2mass,
                                  cf.irac1, cf.irac2, cf.irac3, cf.irac4, cf.mips1, cf.mips2, cf.mips3,
                                  cf.iras12, cf.iras25])
    f0_array = np.array([cf.g_dr2_F0, cf.bp_dr2_F0, cf.rp_dr2_F0, cf.g_edr3_F0, cf.bp_edr3_F0, cf.rp_edr3_F0,
                         cf.W1_F0, cf.W2_F0, cf.W3_F0, cf.W4_F0, cf.j_2mass_F0, cf.h_2mass_F0, cf.k_2mass_F0,
                         cf.irac1_F0, cf.irac2_F0, cf.irac3_F0, cf.irac4_F0, cf.mips1_F0, cf.mips2_F0, cf.mips3_F0,
                         cf.iras12_F0, cf.iras25_F0])

    index_with_wavelength = tools.find_nearest_index(wavelengths_array, wavelength)
    if compare_two_floats(wavelength, wavelengths_array[index_with_wavelength], percentage_error):
        return flux * f0_array[index_with_wavelength]
    else:
        raise ValueError('Unknown wavelength input')


def fit_sed_linear_fit(star_obj: Star) -> float:
    """
    Fits linear Rayleigh Jeans fit for a SED from Star object

    :param star_obj: Star class object
    :return: Constant for fitting from y = -3 * x - c function in log space fit
    """
    mod = get_model_linear_sed_fit_rayleigh_jeans()
    pars = get_param_for_rayleigh_jeans(star_obj.x_sed_linear_fit, star_obj.y_sed_linear_fit)
    out = do_model_fitting_log_space(star_obj.x_sed_linear_fit, star_obj.y_sed_linear_fit,
                                     star_obj.error_sed_linear_fit, mod, pars)
    return out.params['c'].value


def calculate_sed_excess_from_points(x: np.ndarray, y: np.ndarray, error: np.ndarray, const: float) -> list:
    """
    Calculates excess ratio and flux difference given a straight line fit. Three types of excess calculated: average excess beyond MIR, last filter excess and biggest excess

    :param x: Wavelength in meters
    :param y: W/m^2 flux
    :param error: W/m^2 flux errors
    :param const: constant from linear Rayleigh Jeans fit
    :return: list of the ratio parameters:  [last_filter_wavelength, last_filter_excess_error, last_filter_excess_ratio, average_filter_excess_diff, average_filter_excess_ratio, weight_average_filter_excess_diff, weight_average_filter_excess_ratio, biggest_filter_diff_wavelength, biggest_filter_excess_err, biggest_filter_ratio_wavelength, biggest_filter_excess_ratio]
    """
    filters_amount = np.size(x)

    average_filter_excess_diff = 0.0    # Average excess per point
    average_filter_excess_ratio = 0.0

    biggest_filter_excess_err = cf.dflt_no_vle       # Biggest excess difference
    biggest_filter_excess_ratio = cf.dflt_no_vle       # Biggest excess ratio
    biggest_filter_ratio_wavelength = cf.dflt_no_vle

    last_filter_excess_error = cf.dflt_no_vle    # Last filter excess
    last_filter_excess_ratio = cf.dflt_no_vle
    last_filter_wavelength = cf.dflt_no_vle

    for i in range(filters_amount):
        #difference = y[i] - abs(error[i]) - model_function_rayleigh_jeans_in_linear_space(x[i], const)
        #ratio = (y[i] - abs(error[i])) / model_function_rayleigh_jeans_in_linear_space(x[i], const)

        expected_no_excess_flux = model_function_rayleigh_jeans_in_linear_space(x[i], const)
        difference = y[i] - expected_no_excess_flux
        ratio = y[i] / expected_no_excess_flux
        ratio_error = abs(error[i]) / expected_no_excess_flux

        if ratio > biggest_filter_excess_ratio:
            biggest_filter_excess_ratio = ratio
            biggest_filter_excess_err = ratio_error
            biggest_filter_ratio_wavelength = x[i]
        if last_filter_wavelength < x[i]:
            last_filter_excess_error = ratio_error
            last_filter_excess_ratio = ratio
            last_filter_wavelength = x[i]
        elif last_filter_wavelength == x[i] and ratio > last_filter_excess_ratio:
            last_filter_excess_error = ratio_error
            last_filter_excess_ratio = ratio
            last_filter_wavelength = x[i]

        average_filter_excess_diff += difference
        average_filter_excess_ratio += ratio

    if filters_amount != 0:  # If filters exists, then do final calculation
        average_filter_excess_diff = average_filter_excess_diff / filters_amount
        average_filter_excess_ratio = average_filter_excess_ratio / filters_amount
    else:       # Else just return -9999
        average_filter_excess_diff = cf.dflt_no_vle
        average_filter_excess_ratio = cf.dflt_no_vle

    if cf.debug_mode:
        print(f"Biggest ratio wv and ratio: {biggest_filter_ratio_wavelength * pow(10, 6)} "
              f"{biggest_filter_excess_ratio} avg ratio {average_filter_excess_ratio} Last wv and ratio: "
              f"{last_filter_wavelength * pow(10, 6)} {last_filter_excess_ratio}")

    return [last_filter_wavelength, last_filter_excess_error, last_filter_excess_ratio,
            average_filter_excess_diff, average_filter_excess_ratio,
            biggest_filter_excess_err, biggest_filter_ratio_wavelength, biggest_filter_excess_ratio]


def calculate_ir_slope(star_obj: Star, print_variables: bool, show_images: bool, save_images: bool, ir_slope_end=cf.ir_slope_end) -> list:
    """
    Calculates SED line slope, depending on parameters in config_file "ir_slope_start" and "ir_slope_end". No weights/errors are applied in calculation of IR slope. They are not taken into account in error of slope calculation.

    :param star_obj: Star class object
    :param print_variables: If want to print fitted variables
    :param show_images: If want to show images
    :param save_images: If want to save images
    :param ir_slope_end: Where to end the IR slope
    :return: List of fitted parameters [slope, slope_error_lmfit, ir_slope_error, constant_from_slope_fit]
    """

    min_x, max_x = get_min_and_max_x(star_obj.x_good, star_obj.x_upper)

    x_fit = []
    y_fit = []
    err_fit = []

    for i in range(np.size(star_obj.x_good)):
        if cf.ir_slope_start <= star_obj.x_good[i] <= ir_slope_end:
            x_fit.append(star_obj.x_good[i])
            y_fit.append(star_obj.y_good[i])
            err_fit.append(star_obj.error_good[i])

    if np.size(np.asarray(x_fit)) >= 2:  # If enough points, then do the fit
        index_start = tools.find_nearest_index(np.asarray(x_fit), cf.ir_slope_start)
        index_end = tools.find_nearest_index(np.asarray(x_fit), ir_slope_end)
    else:
        index_start = 0
        index_end = 0

    x_fit, y_fit, err_fit = np.asarray(x_fit), np.asarray(y_fit), np.asarray(err_fit)

    if index_start != index_end:    # Do fit only when enough points that are good
        mod = get_model_linear()
        pars = get_param_for_sed_linear_slope(np.log10(x_fit), np.log10(y_fit), ir_slope_end=ir_slope_end)
        out = do_model_fitting_log_space(x_fit, y_fit, np.zeros(np.size(err_fit)), mod, pars)
        #out = minimize_test(x_fit, y_fit, err_fit)

        c = out.params['c'].value   # Constant from linear fit
        m = out.params['m'].value   # Slope
        slope_error_lmfit = out.params['m'].stderr  # Error from the fitter; usually an underestimation though

        x_for_err, y_for_err = np.log10(x_fit), np.log10(y_fit)
        err_for_err = 1 / np.log(10) * err_fit / y_fit  # Convert error to log(error), assuming error << y_value

        ir_slope_error = calculate_ir_slope_error(x_for_err, y_for_err, err_for_err, m, c)  # Calculate error of the slope
    else:
        #print(str(star_obj.source_id) + " : not enough points to fit ir slope")
        m, slope_error_lmfit, ir_slope_error, c = cf.dflt_no_vle, cf.dflt_no_vle, cf.dflt_no_vle, cf.dflt_no_vle

    if print_variables:
        print(f"m: {m} c: {c} my slope error: {ir_slope_error} lmfit slope error: {slope_error_lmfit}")

    if show_images or save_images:  # Save/show fit
        fig, ax = plt.subplots()
        ax.errorbar(star_obj.x_good * pow(10, 6), star_obj.y_good, yerr=star_obj.error_good, fmt='ok', linewidth=2,
                    label="Used data")
        if np.size(star_obj.x_viz) != 0:
            ax.errorbar(star_obj.x_viz * pow(10, 6), star_obj.y_viz, yerr=star_obj.error_viz, fmt='o', linewidth=2,
                        label="Vizier data",
                        color='green', alpha=0.3)
        if np.size(star_obj.x_upper) != 0:
            ax.errorbar(star_obj.x_upper * pow(10, 6), star_obj.y_upper, yerr=star_obj.error_upper, fmt='^',
                        linewidth=2, label="Upper limit")

        if ir_slope_error > -98:
            wavelengths_array = np.linspace(cf.ir_slope_start, ir_slope_end, 10 ** 2)
            model_fit = model_function_sed_slope_in_linear_space(wavelengths_array, m, c)
            ax.loglog(wavelengths_array * pow(10, 6), model_fit, label=f"SED slope {round(m, 2)}", color="red")

        ticks_array = tools.get_ticks_for_plot(max_x)
        ax.set_xticks(ticks_array)
        ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())

        ax.legend(loc="best")
        ax.set(xlabel='Wavelength [μm]',
               ylabel='νF(ν) [W/m$^2$]')

        if save_images:
            plt.savefig(star_obj.ir_slope_directory_png)
            if cf.save_pdfs:
                plt.savefig(star_obj.ir_slope_directory_pdf)
        if show_images:
            plt.show()
        plt.close('all')

    return [m, slope_error_lmfit, ir_slope_error, c]


def calculate_ir_slope_error(x: np.ndarray, y: np.ndarray, error: np.ndarray, slope: float, const: float) -> float:
    """
    Calculate the linear slope st.dev.: sqrt [ Σ(y_i – ŷ_i)^2 / (n – 2) ] / sqrt [ Σ(x_i – x_mean)^2 ]

    :param x: Wavelength in log(meters)
    :param y: Flux in log(W/m^2)
    :param error: Error of flux in log(W/m^2)
    :param slope: Slope of the fit
    :param const: Constant of the linear line fit
    :return: Standard error of the slope/st.dev. of the slope
    """
    n = np.size(x)
    if n <= 2:  # Check if at least 2 points exists, otherwise just return 0 error
        return 0.0
    x_mean = np.mean(x)

    y_slope = x * slope + const  # Model y fit
    """
    y_pos_err = y + error   # Take into account errors for y
    y_neg_err = y - error

    diff1 = y_slope - y_pos_err     # See difference between the fit and y±error
    diff2 = y_slope - y_neg_err

    diff_multiplied = diff1 * diff2     # Multiply the difference. -1 means that slope within error bars
    indices_slope_within_error = np.where(diff_multiplied < 0)

    minimum_diff = np.amin(np.array([np.abs(diff1), np.abs(diff2)]), axis=0)    # Finds minimum distance to the slope
    if np.size(indices_slope_within_error) > 0:     # If error bar within slope, then error = 0 for that point
        minimum_diff[indices_slope_within_error] = minimum_diff[indices_slope_within_error] * 0.0

    yi_y_2 = np.sum(np.square(minimum_diff))    # Sum of squares of the y values"""
    yi_y_2 = np.sum(np.square(y - y_slope))  # Sum of squares of the y values
    ssxx = np.sum(np.square(x - x_mean))    # Sum of (x-x_mean)^2

    slope_error = np.sqrt(yi_y_2 / (n - 2)) / np.sqrt(ssxx)  # Standard deviation of the slope
    return slope_error


def get_min_and_max_x(arr1: np.ndarray, arr2: np.ndarray) -> Tuple[float, float]:
    """
    Returns min and max value within two arrays. Check if either array is empty and ignores those

    :param arr1: First array
    :param arr2: Second array
    :return: min and maximum value
    """

    if np.size(arr2) != 0 and np.size(arr1) != 0:
        min_x = min(np.min(arr1), np.min(arr2))
        max_x = max(np.max(arr1), np.max(arr2))
    elif np.size(arr1) != 0:
        min_x = np.min(arr1)
        max_x = np.max(arr1)
    else:
        min_x = np.min(arr2)
        max_x = np.max(arr2)
    return min_x, max_x


def extrapolate_and_integrate_sed_excess(star_obj: Star, print_variables: bool, show_images: bool, save_images: bool) -> list:
    """
    Calculates SED flux at IR via integration. Extrapolates using linear spline with minimal smoothing. Also calculates excess from Rayleigh Jeans, to see actual excess. Does in both normal and log spaces

    :param star_obj: Star class object
    :param print_variables: If want to print fitted variables
    :param show_images: If want to show images
    :param save_images: If want to save images
    :return: list of the excess
    """

    min_x, max_x = get_min_and_max_x(star_obj.x_good, star_obj.x_upper)

    const = star_obj.sed_linefit_rayleigh_jeans_const  # constant from Rayleigh Jeans slope
    lower_bound = 4.0 * pow(10, -6)  # lower bound for x for integration
    upper_bound = 25.0 * pow(10, -6)  # upper bound for x for integration

    extrapolation_lower_limit = min(2.0 * pow(10, -6), lower_bound)
    extrapolation_upper_limit = max(25.0 * pow(10, -6), upper_bound)

    x_fit = []
    y_fit = []
    error_fit = []

    for i in range(np.size(star_obj.x_good)):
        if extrapolation_lower_limit <= star_obj.x_good[i] <= extrapolation_upper_limit:
            x_fit.append(star_obj.x_good[i])
            y_fit.append(star_obj.y_good[i])
            error_fit.append(star_obj.error_good[i])

    if np.size(np.asarray(x_fit)) >= 2:
        x_fit, y_fit, error_fit = prepare_data_for_interpolate(x_fit, y_fit, np.zeros(np.size(error_fit)))

        index_start = tools.find_nearest_index(np.asarray(x_fit), extrapolation_lower_limit)
        index_end = tools.find_nearest_index(np.asarray(x_fit), extrapolation_upper_limit)
    else:
        index_start = 0
        index_end = 0

    x_fit = np.asarray(x_fit)
    y_fit = np.asarray(y_fit)
    error_fit = np.asarray(error_fit)

    if index_start != index_end:    # Check that have enough good points for extrapolation and integration
        error_fit = error_fit / y_fit

        sorted_indices = x_fit.argsort()    # Sorts indices to be in ascending order
        x_fit = np.log10(x_fit[sorted_indices])
        y_fit = np.log10(y_fit[sorted_indices])
        error_fit = error_fit[sorted_indices]

        # fitted_function = interp1d(x_fit, y_fit, fill_value='extrapolate', kind='slinear', assume_sorted=False)

        extrapolation_fit = interpolate.splrep(x_fit, y_fit, w=1 / error_fit, k=1, s=0)

        # Functions to integrate, log space
        func_sed_log_flux_space = lambda xx: interpolate.splev(xx, extrapolation_fit)
        func_rayleigh_jeans_log_space = lambda xx: model_function_rayleigh_jeans_in_log_space(xx, const)

        """
        NEW ADDITION. If the -3 slope is above purple slope (extrapolated one) at the lower limit value, move -3 slope
        down
        """

        dy = func_sed_log_flux_space(np.log10(lower_bound)) - func_rayleigh_jeans_log_space(np.log10(lower_bound))
        if dy >= 0:
            # All good
            #print("All good")
            pass
        else:
            print(f"{star_obj.source_id} Too high -3 slope, reducing by {dy}")
            const = const - dy
            #star_obj.sed_linefit_rayleigh_jeans_const = const
            func_rayleigh_jeans_log_space = lambda xx: model_function_rayleigh_jeans_in_log_space(xx, const)

        # Finding lower bound, where intersection between graphs happens
        # func_to_find_root = lambda xx: interpolate.splev(xx, extrapolation_fit) - model_function_rayleigh_jeans_in_log_space(xx, const)
        # lower_bound = np.power(10.0, find_max_root(func_to_find_root, np.log10(lower_bound), np.log10(upper_bound), 5))

        # Integral of log fits
        sed_integ_flux_log_space = integrate.quad(func_sed_log_flux_space, np.log10(lower_bound), np.log10(upper_bound), points=x_fit)
        rayleigh_integ_flux_log_space = integrate.quad(func_rayleigh_jeans_log_space, np.log10(lower_bound), np.log10(upper_bound), points=x_fit)

        # Functions to integrate, linear space
        func_sed_lin_flux_space = lambda xx: pow(10.0, interpolate.splev(np.log10(xx), extrapolation_fit))
        func_rayleigh_jeans_lin_space = lambda xx: model_function_rayleigh_jeans_in_linear_space(xx, const)

        # Integral of linear fits
        sed_integ_flux_lin_space = integrate.quad(func_sed_lin_flux_space, lower_bound, upper_bound,
                                                  points=np.power(10.0, x_fit))
        rayleigh_integ_flux_lin_space = integrate.quad(func_rayleigh_jeans_lin_space, lower_bound, upper_bound,
                                                       points=np.power(10.0, x_fit))

        # First method of doing integration manually, using the extrapolation function
        #sed_integ_manual_log_space = integrate_using_trapezoids(func_sed_log_flux_space, x_fit, np.log10(lower_bound), np.log10(upper_bound))
        #ray_integ_manual_log_space = integrate_using_trapezoids(func_rayleigh_jeans_log_space, np.array([]), np.log10(lower_bound),
        #                                                        np.log10(upper_bound))

        # Second method of doing integration manually, without using extrapolation function
        """
        x1 = np.log10(lower_bound)
        x2 = np.log10(upper_bound)

        x_fit_n, y_fit_n, error_fit_n = tools.take_within_x_boundaries(x_fit, y_fit, error_fit, np.log10(lower_bound), np.log10(upper_bound), 0)

        if np.size(x_fit[x_fit <= x1]) == 0:
            index_start = tools.find_nearest_index(np.asarray(x_fit), x1)
            index_end = index_start + 1

            xx1 = x_fit[index_start]
            xx2 = x_fit[index_end]

            yy1 = y_fit[index_start]
            yy2 = y_fit[index_end]
        else:
            index_start = tools.find_nearest_index(np.asarray(x_fit[x_fit <= x1]), x1)
            index_end = index_start + 1 #tools.find_nearest_index(np.asarray(x_fit[x_fit > x1]), x1)

            xx1 = x_fit[index_start]
            xx2 = x_fit[index_end]

            yy1 = y_fit[index_start]
            yy2 = y_fit[index_end]

        m_ext = (yy1 - yy2) / (xx1 - xx2)
        c_ext = (xx1 * yy2 - xx2 * yy1) / (xx1 - xx2)

        y_lower = m_ext * x1 + c_ext

        if np.size(x_fit[x_fit > x2]) == 0:
            index_end = tools.find_nearest_index(np.asarray(x_fit), x2)
            index_start = index_end - 1

            xx1 = x_fit[index_start]
            xx2 = x_fit[index_end]

            yy1 = y_fit[index_start]
            yy2 = y_fit[index_end]
        else:
            index_end = tools.find_nearest_index(np.asarray(x_fit[x_fit > x2]), x2) + np.size(x_fit[x_fit <= x2])
            index_start = index_end - 1 #tools.find_nearest_index(np.asarray(x_fit[x_fit <= x2]), x2)

            xx1 = x_fit[index_start]
            xx2 = x_fit[index_end]

            yy1 = y_fit[index_start]
            yy2 = y_fit[index_end]

        m_ext = (yy1 - yy2) / (xx1 - xx2)
        c_ext = (xx1 * yy2 - xx2 * yy1) / (xx1 - xx2)

        y_upper = m_ext * x2 + c_ext

        x_fit_n = np.append(x1, x_fit_n)
        x_fit_n = np.append(x_fit_n, x2)

        y_fit_n = np.append(y_lower, y_fit_n)
        y_fit_n = np.append(y_fit_n, y_upper)

        sed_integ_manual_log_space = integrate_using_trapezoids_using_points(x_fit_n, y_fit_n)

        y1 = model_function_rayleigh_jeans_in_log_space(np.log10(lower_bound), const)
        y2 = model_function_rayleigh_jeans_in_log_space(np.log10(upper_bound), const)
        ray_integ_manual_log_space = integrate_using_trapezoids_using_points(np.array([x1, x2]), np.array([y1, y2]))

        print(sed_integ_flux_log_space[0], sed_integ_manual_log_space, rayleigh_integ_flux_log_space[0], ray_integ_manual_log_space)"""
    else:
        #print("Not enough data points")

        sed_integ_flux_lin_space, rayleigh_integ_flux_lin_space, sed_integ_flux_log_space, \
        rayleigh_integ_flux_log_space = [cf.dflt_no_vle, cf.dflt_no_vle], [cf.dflt_no_vle, cf.dflt_no_vle], [cf.dflt_no_vle, cf.dflt_no_vle], [cf.dflt_no_vle, cf.dflt_no_vle]
        sed_integ_manual_log_space = cf.dflt_no_vle
        ray_integ_manual_log_space = cf.dflt_no_vle

    if print_variables:
        print(sed_integ_flux_lin_space[0], sed_integ_flux_lin_space[1], rayleigh_integ_flux_lin_space[0],
            rayleigh_integ_flux_lin_space[1], sed_integ_flux_log_space[0], sed_integ_flux_log_space[1],
            rayleigh_integ_flux_log_space[0], rayleigh_integ_flux_log_space[1])

    if show_images or save_images:
        fig, ax = plt.subplots()
        ax.errorbar(star_obj.x_good * pow(10, 6), star_obj.y_good, yerr=star_obj.error_good, fmt='ok', linewidth=2,
                    label="Data")

        wavelengths_array_2 = np.linspace(min(lower_bound, cf.sed_linear_line_wavelength_start),
                                          max(upper_bound, cf.sed_linear_line_wavelength_end), 10 ** 2)
        model_fit_v2 = (model_function_rayleigh_jeans_in_linear_space(wavelengths_array_2, const))
        ax.loglog(wavelengths_array_2 * pow(10, 6), model_fit_v2, label="-3 slope", color="red")

        if sed_integ_flux_lin_space[0] > cf.dflt_no_vle:   # If extrapolation is done, also plot those
            wavelengths_array = np.linspace(extrapolation_lower_limit, extrapolation_upper_limit, 10 ** 2)
            ax.plot(wavelengths_array * pow(10, 6), func_sed_lin_flux_space(wavelengths_array), label="SED fit",
                    color="purple")

            line_min_y = min(np.min(star_obj.y_good), np.min(model_fit_v2))
            line_max_y = max(np.max(star_obj.y_good), np.max(model_fit_v2))

            plt.vlines(lower_bound * pow(10, 6), line_min_y, line_max_y, label="Bounds", color="blue")
            plt.vlines(upper_bound * pow(10, 6), line_min_y, line_max_y, color="blue")

        if np.size(star_obj.x_viz) != 0:
            ax.errorbar(star_obj.x_viz * pow(10, 6), star_obj.y_viz, yerr=star_obj.error_viz, fmt='o', linewidth=2,
                        label="Vizier data", color='green', alpha=0.3)
        if np.size(star_obj.x_upper) != 0:
            ax.errorbar(star_obj.x_upper * pow(10, 6), star_obj.y_upper, yerr=star_obj.error_upper, fmt='^',
                        linewidth=2, label="Upper limit")

        ticks_array = tools.get_ticks_for_plot(max_x)
        ax.set_xticks(ticks_array)
        ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())

        ax.legend(loc='lower left')
        ax.set(xlabel='Wavelength [μm]',
               ylabel='νF(ν) [W/m$^2$]')

        if save_images:
            plt.savefig(star_obj.sed_integrated_excess_png)
        if show_images:
            plt.show()
        plt.close('all')

    #return [sed_integ_flux_lin_space[0], sed_integ_flux_lin_space[1], rayleigh_integ_flux_lin_space[0],
    #        rayleigh_integ_flux_lin_space[1], sed_integ_manual_log_space, sed_integ_flux_log_space[0],
    #        ray_integ_manual_log_space, rayleigh_integ_flux_log_space[0]]
    return [sed_integ_flux_lin_space[0], sed_integ_flux_lin_space[1], rayleigh_integ_flux_lin_space[0],
            rayleigh_integ_flux_lin_space[1], sed_integ_flux_log_space[0], sed_integ_flux_log_space[1],
            rayleigh_integ_flux_log_space[0], rayleigh_integ_flux_log_space[1]]


def integrate_using_trapezoids_using_points(x, y, min_x=0, max_x=1e99):
    # Bounds are not expected to be included in the points
    #points = np.sort(points)
    #points_low = x[lower_bound <= x]
    #points = points_low[points_low <= upper_bound]
    #points = np.append(points, np.array([upper_bound]))
    #points = np.append(points, np.array([lower_bound, upper_bound]))
    #points = np.sort(points)

    if min_x > max_x:
        min_x, max_x = max_x, min_x

    sorted_indices = x.argsort()
    x = x[sorted_indices]
    y = y[sorted_indices]

    x_new, y_new = [], []
    for i in range(np.size(x)):
        if min_x <= x[i] <= max_x:
            x_new.append(x[i])
            y_new.append(y[i])

    x = x_new
    y = y_new

    points_lower = np.delete(x, -1)
    points_upper = np.delete(x, 0)

    y_lower = np.delete(y, -1)
    y_upper = np.delete(y, 0)

    area = 0

    #for i in range(np.size(points) - 1):
    #    area += area_under_trapezoid(points[i], points[i+1], func(points[i]), func(points[i+1]))

    area = np.sum(area_under_trapezoid(points_lower, points_upper, y_lower, y_upper))

    return area


def integrate_using_trapezoids(func, points, lower_bound, upper_bound):
    # Bounds are not expected to be included in the points
    #points = np.sort(points)
    points_low = points[lower_bound <= points]
    points = points_low[points_low <= upper_bound]
    #points = np.append(points, np.array([upper_bound]))
    points = np.append(points, np.array([lower_bound, upper_bound]))
    points = np.sort(points)
    points_lower = np.delete(points, -1)
    points_upper = np.delete(points, 0)

    area = 0

    #for i in range(np.size(points) - 1):
    #    area += area_under_trapezoid(points[i], points[i+1], func(points[i]), func(points[i+1]))

    area = np.sum(area_under_trapezoid(points_lower, points_upper, func(points_lower), func(points_upper)))

    return area


def area_under_trapezoid(x1, x2, y1, y2):
    return np.abs(x2 - x1) * (y1 + y2) * 0.5


def find_max_root(func_to_find_root, lower_bound, upper_bound, intervals):
    intervals_to_try = np.linspace(0, 1, intervals)[::-1]
    for a in intervals_to_try:
        try:
            lower_bound = scipy.optimize.brentq(func_to_find_root, get_position_between_points(lower_bound, upper_bound, a), upper_bound)
            #print(lower_bound)
            return lower_bound
        except:
            pass
    #print("Could not find intersection")
    return lower_bound


def get_position_between_points(a, b, position):
    if a < b:
        return a * (1 - position) + position * b
    else:
        return b * (1 - position) + position * a


def prepare_data_for_interpolate(x: Union[np.ndarray, list], y: Union[np.ndarray, list], error: Union[np.ndarray, list]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    For inter/extra-polation it is forbidden to have same x-values. This function takes weighted mean average of points with the same x-value

    :param x: Wavelength in meters
    :param y: Flux in W/m^2
    :param error: Error of flux in W/m^2
    :return: Same arrays, but with some points combined, i.e. no points have same x-values here
    """
    x, y, error = np.asarray(x), np.asarray(y), np.asarray(error)

    mean_error = np.mean(error)     # Check before that have at least 2 values already
    if mean_error == 0:  # If no values have error, just assume that error is 1 for equal weight
        error = np.ones(np.size(error))

    indices_to_remove = np.where(error <= 10e-30)[0]    # To make sure there are no 0 errors
    if np.size(indices_to_remove > 0):
        error[indices_to_remove] = error[indices_to_remove] * 0 + mean_error

    duplicates = list_duplicates(x)

    x_new = []
    y_new = []
    err_new = []

    indices_used = np.array([]).astype(int)

    for dup in duplicates:  # Go through duplicates and combine those
        indices = dup[1]
        indices_used = np.append(indices_used, indices)     # Add the duplicates to know which ones are used already

        new_x = dup[0]
        new_y = calc_weighted_arith_mean(y[indices], 1 / error[indices])    # Combine y-s
        new_err = calc_error_for_weighted_arith_mean(error[indices], 1 / error[indices])

        x_new.append(new_x)
        y_new.append(new_y)
        err_new.append(new_err)

    indices_used = np.asarray(indices_used).flatten()

    remaining_x = np.delete(x, indices_used)    # Add remaining good values
    remaining_y = np.delete(y, indices_used)
    remaining_err = np.delete(error, indices_used)

    x_new, y_new, err_new = np.asarray(x_new), np.asarray(y_new), np.asarray(err_new)

    x_new = np.append(x_new, remaining_x)
    y_new = np.append(y_new, remaining_y)
    err_new = np.append(err_new, remaining_err)

    return x_new, y_new, err_new


def calc_error_for_weighted_arith_mean(err: np.ndarray, weight: np.ndarray) -> float:
    """
    Calculate error for weighted arithmetic mean

    :param err: Array with errors
    :param weight: Array with weights for the errors
    :return: Error depending on the weights
    """
    return 1 / (np.sum(weight)) * np.sqrt(np.sum(np.square(err * weight)))


def calc_weighted_arith_mean(y: np.ndarray, weight: np.ndarray) -> float:
    """
    Calculates weighted arithmetic mean

    :param y: y values
    :param weight: weights
    :return: New weighted arithmetic mean
    """
    return np.sum(y * weight) / np.sum(weight)


def list_duplicates(seq: np.ndarray):
    """
    Returns a dub inside array, where each one represents duplicate arrays

    :param seq: Array where to find duplicates
    :return: Dups inside array, where each one represents duplicate arrays
    """
    # source: https://stackoverflow.com/questions/5419204/index-of-duplicates-items-in-a-python-list
    tally = defaultdict(list)
    for i, item in enumerate(seq):
        tally[item].append(i)
    return ((key, locs) for key, locs in tally.items()
            if len(locs) > 1)


def plot_image(star_obj: Star, image_showing: bool, save_image: bool):
    """
    Plots image of the SED plot and saves/show them if required

    :param star_obj: Star class object
    :param save_image: if want to save image
    :param image_showing: if want to show image
    """
    # print(out.fit_report())

    if image_showing or save_image:
        min_x, max_x = tools.find_min_max_x_for_plot(star_obj)

        sed_line_min_x = max(min_x, cf.sed_linear_line_wavelength_start * 0.8)
        sed_line_max_x = min(max_x, cf.ir_excess_end)

        wavelengths_array = np.linspace(sed_line_min_x, sed_line_max_x, 10 ** 2)

        model_fit_v2 = model_function_rayleigh_jeans_in_linear_space(wavelengths_array,
                                                                     star_obj.sed_linefit_rayleigh_jeans_const)

        fig, ax = plt.subplots()
        ax.loglog(wavelengths_array * pow(10, 6), model_fit_v2, label="-3 slope", color="red")

        if np.size(star_obj.x_good) != 0:
            ax.errorbar(star_obj.x_good * pow(10, 6), star_obj.y_good, yerr=star_obj.error_good, fmt='ok', linewidth=2,
                    label="Used points")

        if np.size(star_obj.x_viz) != 0:
            ax.errorbar(star_obj.x_viz * pow(10, 6), star_obj.y_viz, yerr=star_obj.error_viz, fmt='o', linewidth=2,
                        label="Vizier data", color='green', alpha=0.3)

        if np.size(star_obj.x_upper) != 0:
            ax.errorbar(star_obj.x_upper * pow(10, 6), star_obj.y_upper, yerr=star_obj.error_upper, fmt='^',
                        linewidth=2, label="Upper limit")

        ticks_array = tools.get_ticks_for_plot(max_x)
        ax.set_xticks(ticks_array)
        ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
        plt.title("SED plot")
        ax.legend(loc="best")
        ax.set(xlabel='Wavelength [μm]',
               ylabel='νF(ν) [W/m$^2$]')

        if save_image:
            plt.savefig(star_obj.sed_fit_directory_png)
            if cf.save_pdfs:
                plt.savefig(star_obj.sed_fit_directory_pdf)

        if image_showing:
            plt.show()

        plt.close('all')

        plt.title("Cross-match separations with other catalogues")
        plt.bar(star_obj.separation_table, star_obj.separation_arcsec)
        plt.xlabel('Catalogue', fontsize=10)
        plt.ylabel('Separation [arcsec]')
        plt.grid(True)

        if save_image:
            plt.savefig(star_obj.sed_bar_dir_png)
            if cf.save_pdfs:
                plt.savefig(star_obj.sed_bar_dir_pdf)

        if image_showing:
            plt.show()

        plt.close('all')
