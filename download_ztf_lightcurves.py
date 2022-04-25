import wget
import numpy as np
import tools
import get_gaia_table
import config_file as cf
import os

#band_name = ['g', 'r', 'i']
#start_time_mjd = 58194.0
#end_time_mjd = 58483.0
#url_og = "https://irsa.ipac.caltech.edu/cgi-bin/ZTF/nph_light_curves?POS=CIRCLE+" + ra + "+" + dec + "+" + search_radius + "&BANDNAME=" + band_name + "&NOBS_MIN=3&TIME=" + start_time_mjd + "+" + end_time_mjd + "&BAD_CATFLAGS_MASK=32768&FORMAT=ipac_table"


def check_if_file_exists(file_directory):
    return os.path.isfile(file_directory)


search_radius = 1.7 / 3600  # 2 arcsec radius
gaia_lpv_table = get_gaia_lpv_table.get_data_from_table(cf.input_gaia_table_path)
ids = tools.get_specific_column_from_data_str(gaia_lpv_table, "source_id").astype(str)
ras = tools.get_specific_column_from_data_str(gaia_lpv_table, "ra_ge3").astype(str)
decs = tools.get_specific_column_from_data_str(gaia_lpv_table, "dec_ge3").astype(str)

for i in range(np.size(ids)):
    output_directory = 'input_lightcurves/all/ztf_dr10/' + ids[i] + '.vot'

    file_exist = check_if_file_exists(output_directory)
    if not file_exist:
        url = "https://irsa.ipac.caltech.edu/cgi-bin/ZTF/nph_light_curves?POS=CIRCLE+" + str(ras[i]) + "+" + \
              str(decs[i]) + "+" + str(search_radius) + "&NOBS_MIN=3&BAD_CATFLAGS_MASK=32768&FORMAT=VOTABLE"
        filename = wget.download(url, out=output_directory)
        print(filename)
    else:
        print(ids[i], 'already exists')