from hyperion.model import ModelOutput
from hyperion.util.constants import pc
import matplotlib.pyplot as plt
import numpy as np

# Hyperion data: uses the Robitaille 2017 YSO models and extracts dust and total SED emission.
# Plots the raw SED with different parts as its own colour.
# Hyperion package: Mac/Linux only (i.e. not Windows). Requires original grid1.1 data, which is heavier
# than the original models.

if __name__ == '__main__':
    ids = ["3326710663360978944"]   # ID of the star to plot

    fitted_data = np.loadtxt('sedfitter/all_params_2021-12-21_all_r17_sp__s_i_new_parallax_1pc_0.3upper_av_varied.txt', dtype=str)
    source_ids = fitted_data[:, 0]
    model_names = fitted_data[:, 4]
    fitted_av = fitted_data[:, 6].astype(float)
    fitted_scale = fitted_data[:, 7].astype(float)
    star_radii = fitted_data[:, 8].astype(float)
    star_temp = fitted_data[:, 9].astype(float)
    inclination = fitted_data[:, 16].astype(float)

    for j in ids:
        i = np.where(source_ids == j)[0][0]

        model = ModelOutput('grids-1.1/sp--s-i/output/' + model_names[i][0:2].lower() + "/" + model_names[i][:-3] + '.rtout')
        print("Model name:", model_names[i], ". Fitted AV:", fitted_av[i], ". Fitted scale:", fitted_scale[i])

        inclination_type = int(inclination[i] / 10)
        print("Inclination:", inclination[i], ". Inclination type:", inclination_type)

        aperture = -1   # Take biggest aperture
        sed_total_sed = model.get_sed(inclination=inclination_type, aperture=aperture, distance=1000 * pc)  # Total SED
        sed_direct_stellar = model.get_sed(inclination=inclination_type, aperture=aperture, distance=1000 * pc, component='source_emit')  # Direct stellar photons
        sed_scattered_stellar = model.get_sed(inclination=inclination_type, aperture=aperture, distance=1000 * pc, component='source_scat')  # Scattered stellar photons
        sed_direct_dust = model.get_sed(inclination=inclination_type, aperture=aperture, distance=1000 * pc, component='dust_emit')  # Direct dust photons
        sed_scattered_dust = model.get_sed(inclination=inclination_type, aperture=aperture, distance=1000 * pc, component='dust_scat')  # Scattered dust photons

        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)

        # Total SED
        ax.loglog(sed_total_sed.wav, sed_total_sed.val, color='black', lw=3, alpha=0.5, label='total')

        # Direct stellar photons
        ax.loglog(sed_direct_stellar.wav, sed_direct_stellar.val, color='blue', label='source (direct)')

        # Scattered stellar photons
        ax.loglog(sed_scattered_stellar.wav, sed_scattered_stellar.val, color='teal', label='source (scattered)')

        # Direct dust photons
        ax.loglog(sed_direct_dust.wav, sed_direct_dust.val, color='red', label='dust (direct)')

        # Scattered dust photons
        ax.loglog(sed_scattered_dust.wav, sed_scattered_dust.val, color='orange', label='dust (scattered)')

        ax.set_xlabel(r'$\lambda$ [$\mu$m]')
        ax.set_ylabel(r'$\lambda F_\lambda$ [ergs/s/cm$^2$]')
        ax.set_xlim(0.1, 2000.)
        # ax.set_ylim(6.e-14, 3.e-9)  # Adjust y to your liking

        ax.legend(loc='best', fontsize=8)

        plt.show()
        plt.close('all')
