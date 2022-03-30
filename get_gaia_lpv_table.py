import numpy as np
import tools
import config_file as cf


def find_index_with_id(data, source_id: str) -> int:
    """
    Find the index where id is located in data

    :param data: Gaia FITS table from .vots
    :param source_id: source_id
    """
    try:
        return np.where(data.field('source_id').astype(str) == str(source_id))[0][0]
    except:
        return np.where(data.field('source_id').astype(str) == str(source_id))[0]


def get_star_info(data, source_id: str) -> dict:
    """
    Get all star info from FITS (.vot) table given specific id
    
    :param data: Gaia FITS table from .vots
    :param source_id: ID of the object
    :return: Dictionary with elements as fields from that star. To reference use: VARIABLE_NAME['FIELD_NAME']
    """
    index = find_index_with_id(data, str(source_id))
    distance = data.field(cf.field_name_distance)[index].astype(float)
    par_ge3 = data.field(cf.field_name_parallax_g3)[index].astype(float)
    par_error_ge3 = data.field(cf.field_name_parallax_error_g3)[index].astype(float)
    ra = data.field(cf.field_name_ra_g3)[index].astype(float)
    dec = data.field(cf.field_name_dec_g3)[index].astype(float)
    l = data.field(cf.field_name_l_g3)[index].astype(float)
    b = data.field(cf.field_name_b_g3)[index].astype(float)
    pmra = data.field(cf.field_name_pmra_g3)[index].astype(float)
    pmdec = data.field(cf.field_name_pmdec_g3)[index].astype(float)
    g_mag_g2 = data.field(cf.field_name_g_mag_g2)[index].astype(float)
    bp_mag_g2 = data.field(cf.field_name_bp_mag_g2)[index].astype(float)
    rp_mag_g2 = data.field(cf.field_name_rp_mag_g2)[index].astype(float)
    rad_vel_g2 = data.field(cf.field_name_rad_vel_g2)[index].astype(float)
    period_g2 = data.field(cf.field_name_period_g2)[index].astype(float)
    period_err_g2 = data.field(cf.field_name_period_error_g2)[index].astype(float)
    extinction_g2 = data.field(cf.field_name_a_g_val_g2)[index].astype(float)
    teff_template_g2 = data.field(cf.field_name_rv_template_teff_g2)[index].astype(float)
    teff_val_g2 = data.field(cf.field_name_teff_val_g2)[index].astype(float)
    radius_g2 = data.field(cf.field_name_radius_g2)[index].astype(float)
    radius_lower_g2 = data.field(cf.field_name_radius_lower_g2)[index].astype(float)
    radius_upper_g2 = data.field(cf.field_name_radius_upper_g2)[index].astype(float)
    name_simbad = data.field(cf.field_name_main_id_simbad)[index]
    main_type_simbad = data.field(cf.field_name_main_type_simbad)[index]
    other_types_simbad = data.field(cf.field_name_other_types_simbad)[index]
    g_mag_ge3 = data.field(cf.field_name_g_mag_g3)[index].astype(float)
    bp_mag_ge3 = data.field(cf.field_name_bp_mag_g3)[index].astype(float)
    rp_mag_ge3 = data.field(cf.field_name_rp_mag_g3)[index].astype(float)

    # to reference use: VARIABLE_NAME['FIELD_NAME']

    return {"dist": distance, "parallax_ge3": par_ge3, "parallax_error_ge3": par_error_ge3, "ra": ra, "dec": dec, "l": l, "b": b, "pmra": pmra, "pmdec": pmdec,
            "g_mag_g2": g_mag_g2, "bp_mag_g2": bp_mag_g2, "rp_mag_g2": rp_mag_g2, "rad_vel_g2": rad_vel_g2,
            "period_g2": period_g2, "period_err_g2": period_err_g2, "extinction_g2": extinction_g2,
            "teff_template_g2": teff_template_g2, "teff_val_g2": teff_val_g2, "name_simbad": name_simbad, "main_type_simbad": main_type_simbad,
            "other_types_simbad": other_types_simbad, "g_mag_ge3": g_mag_ge3, "bp_mag_ge3": bp_mag_ge3,
            "rp_mag_ge3": rp_mag_ge3, "radius_g2": radius_g2, "radius_lower_g2": radius_lower_g2,
            "radius_upper_g2": radius_upper_g2}


def get_data_from_table(table_directory: str):
    """
    Returns FITS data table, given .vot filename path

    :param table_directory: Path to the .vot table
    :return: FITS data table
    """
    data, cols = tools.load_fits_table(table_directory)
    return data

