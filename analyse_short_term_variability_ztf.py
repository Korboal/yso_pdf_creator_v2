import numpy as np
import tools
import lightcurve_class
from astropy.io.votable import parse_single_table
import get_gaia_table
import config_file as cf
import matplotlib.pyplot as plt
import scipy.fftpack
from scipy.fft import fft, fftfreq
from scipy.signal import lombscargle
import matplotlib

#matplotlib.use('macosx')

min_measurements_per_night = 40


def test_one_synthetic():
    gaia_lpv_table = get_gaia_lpv_table.get_data_from_table(
        cf.input_gaia_table_path)  # Get main Gaia info (GDR2 + GEDR3)
    ids = tools.get_specific_column_from_data_str(gaia_lpv_table, "source_id").astype(str)

    path_ztf = cf.input_ztf_lightcurves_to_analyse

    ids_done = tools.load_data("temp_out.txt")

    for source_id in ids:
        print(source_id)
        if source_id not in ids_done:
            #if True:
            #source_id = "510301664500565120"

            ztf_votable_data = parse_single_table(path_ztf + source_id + ".vot")
            for band in ["g", "r"]:
                ztf_light_curve = lightcurve_class.LightCurveZTF(ztf_votable_data, band, show_image=False)

                time = ztf_light_curve.data_t

                #if source_id == "3208974820820244096":
                if True and np.size(time) >= min_measurements_per_night:
                    time_int = np.copy(time).astype(int)
                    unique, counts = np.unique(time_int, return_counts=True)
                    where_too_many_light_curve_measurements = np.where(counts >= min_measurements_per_night)[0]
                    if np.size(where_too_many_light_curve_measurements) > 0:
                        for index in where_too_many_light_curve_measurements:
                            #print(counts)
                            args_to_bin = np.where(time_int == unique[index])[0]

                            ztf_light_curve_temp = lightcurve_class.LightCurveZTF(None, band, show_image=True)

                            ztf_light_curve_temp.data_t = ztf_light_curve.data_t[args_to_bin]
                            ztf_light_curve_temp.data_y = ztf_light_curve.data_y[args_to_bin]
                            ztf_light_curve_temp.data_length = np.size(ztf_light_curve_temp.data_t)
                            ztf_light_curve_temp.data_error = ztf_light_curve.data_error[args_to_bin]

                            ztf_light_curve_temp.fit_light_curve(True)

                            if max(ztf_light_curve_temp.ls_power) > 0.3:
                                ztf_light_curve_temp.draw_folded_light_curve()
                                ztf_light_curve_temp.draw_fitted_light_curve()
                                ztf_light_curve_temp.draw_frequency_gram()
                                ztf_light_curve_temp.draw_periodogram()



def main():
    test_one_synthetic()

if __name__ == '__main__':
    main()