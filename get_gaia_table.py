import numpy as np
import tools
import config_file as cf
from typing import Union, Tuple, Callable, Any, List


def find_index_with_id(data, source_id: str) -> Union[int, bool]:
    """
    Find the index where id is located in data

    :param data: Gaia FITS table from .vots
    :param source_id: source_id
    :return index where id is located or False if index is not in there
    """
    # 2022.05.03 updated the function
    index_array = np.where(data.field('source_id') == np.int64(source_id))[0]
    if np.size(index_array) == 0:
        return False
    else:
        return index_array[0]
    """try:
        return np.where(data.field('source_id') == np.int64(source_id))[0][0]
    except IndexError:
        return False"""

    """
    try:
        return np.where(data.field('source_id').astype(str) == str(source_id))[0][0]
    except:
        return np.where(data.field('source_id').astype(str) == str(source_id))[0]"""


def get_star_info(data, source_id: str) -> dict:
    """
    Get all star info from FITS (.vot) table given specific id
    
    :param data: Gaia FITS table from .vots
    :param source_id: ID of the object
    :return: Dictionary with elements as fields from that star. To reference use: VARIABLE_NAME['FIELD_NAME']
    """
    index = find_index_with_id(data, source_id)

    source_id_g2 = get_field_from_data(data, index, cf.source_id_g2, convert_to_float=False)
    par_ge3 = get_field_from_data(data, index, cf.field_name_parallax_g3)   # should be in mas units
    distance = 1000.0 / par_ge3
    par_error_ge3 = get_field_from_data(data, index, cf.field_name_parallax_error_g3)   # should be in mas units
    ra = get_field_from_data(data, index, cf.field_name_ra_g3)
    dec = get_field_from_data(data, index, cf.field_name_dec_g3)
    l = get_field_from_data(data, index, cf.field_name_l_g3)
    b = get_field_from_data(data, index, cf.field_name_b_g3)
    pmra = get_field_from_data(data, index, cf.field_name_pmra_g3)
    pmdec = get_field_from_data(data, index, cf.field_name_pmdec_g3)
    g_mag_g2 = get_field_from_data(data, index, cf.field_name_g_mag_g2)
    bp_mag_g2 = get_field_from_data(data, index, cf.field_name_bp_mag_g2)
    rp_mag_g2 = get_field_from_data(data, index, cf.field_name_rp_mag_g2)
    rad_vel_g2 = get_field_from_data(data, index, cf.field_name_rad_vel_g2)
    period_g2 = get_field_from_data(data, index, cf.field_name_period_g2)   # should be in days
    period_err_g2 = get_field_from_data(data, index, cf.field_name_period_error_g2)  # should be in days
    extinction_g2 = get_field_from_data(data, index, cf.field_name_a_g_val_g2)
    teff_template_g2 = get_field_from_data(data, index, cf.field_name_rv_template_teff_g2)
    teff_val_g2 = get_field_from_data(data, index, cf.field_name_teff_val_g2)
    radius_g2 = get_field_from_data(data, index, cf.field_name_radius_g2)
    radius_lower_g2 = get_field_from_data(data, index, cf.field_name_radius_lower_g2)
    radius_upper_g2 = get_field_from_data(data, index, cf.field_name_radius_upper_g2)
    name_simbad = get_field_from_data(data, index, cf.field_name_main_id_simbad, convert_to_float=False)
    main_type_simbad = get_field_from_data(data, index, cf.field_name_main_type_simbad, convert_to_float=False)
    other_types_simbad = get_field_from_data(data, index, cf.field_name_other_types_simbad, convert_to_float=False)
    g_mag_ge3 = get_field_from_data(data, index, cf.field_name_g_mag_g3)
    bp_mag_ge3 = get_field_from_data(data, index, cf.field_name_bp_mag_g3)
    rp_mag_ge3 = get_field_from_data(data, index, cf.field_name_rp_mag_g3)

    # to reference use: VARIABLE_NAME['FIELD_NAME']

    return {"dist": distance, "parallax_ge3": par_ge3, "parallax_error_ge3": par_error_ge3, "ra": ra, "dec": dec, "l": l, "b": b, "pmra": pmra, "pmdec": pmdec,
            "g_mag_g2": g_mag_g2, "bp_mag_g2": bp_mag_g2, "rp_mag_g2": rp_mag_g2, "rad_vel_g2": rad_vel_g2,
            "period_g2": period_g2, "period_err_g2": period_err_g2, "extinction_g2": extinction_g2,
            "teff_template_g2": teff_template_g2, "teff_val_g2": teff_val_g2, "name_simbad": name_simbad, "main_type_simbad": main_type_simbad,
            "other_types_simbad": other_types_simbad, "g_mag_ge3": g_mag_ge3, "bp_mag_ge3": bp_mag_ge3,
            "rp_mag_ge3": rp_mag_ge3, "radius_g2": radius_g2, "radius_lower_g2": radius_lower_g2,
            "radius_upper_g2": radius_upper_g2, "source_id_g2": source_id_g2}


def get_field_from_data(data, index: int, field: str, convert_to_float=True) -> Union[float, str]:
    try:
        field_value = data.field(field)[index]
        if convert_to_float:
            return field_value.astype(float)
        else:
            return field_value
    except:
        if convert_to_float:
            return np.nan
        else:
            return "NAN"


def get_data_from_table(table_directory: str):
    """
    Returns FITS data table, given .vot filename path

    :param table_directory: Path to the .vot table
    :return: FITS data table
    """
    data, cols = tools.load_fits_table(table_directory)
    return data

