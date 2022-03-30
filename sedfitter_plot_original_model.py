from sedfitter.sed import SEDCube
import matplotlib.pyplot as plt
import numpy as np
from astropy import units as u
from sedfitter.utils import integrate_subset
from astropy import constants as const
from sedfitter.extinction import Extinction

# Plots original model fit. Can plot two SEDs at once, if necessary (uncomment if needed).
# Also calculates theoretical luminosity in Ls units (does it work well???)
# total_lumin: Total luminosity by integrating the SED
# star_lumin: Luminosity of star based on fitted temperature and radius using Steffan-Boltzmann law

if __name__ == '__main__':
    seds = SEDCube.read('sp--s-i/flux.fits')
    fitted_data = np.loadtxt('sedfitter/all_params_2021-12-20_all_r17_sp__s_i_avlimit_new_parallax_1pc_0.3upper.txt', dtype=str)
    source_ids = fitted_data[:, 0]
    model_names = fitted_data[:, 4]
    fitted_av = fitted_data[:, 6].astype(float)
    fitted_scale = fitted_data[:, 7].astype(float)
    star_radii = fitted_data[:, 8].astype(float)
    star_temp = fitted_data[:, 9].astype(float)

    """seds_2 = SEDCube.read('s---s-i/flux.fits')
    fitted_data_2 = np.loadtxt('sedfitter/all_params_2021-12-20_all_r17_s___s_i_avlimit_new_parallax_1pc_0.3upper.txt', dtype=str)
    source_ids_2 = fitted_data_2[:, 0]
    model_names_2 = fitted_data_2[:, 4]
    fitted_av_2 = fitted_data_2[:, 6].astype(float)
    fitted_scale_2 = fitted_data_2[:, 7].astype(float)
    star_radii_2 = fitted_data_2[:, 8].astype(float)
    star_temp_2 = fitted_data_2[:, 9].astype(float)"""

    extinction = Extinction.from_file('whitney.txt', columns=[0, 3], wav_unit=u.micron, chi_unit=u.cm ** 2 / u.g)

    print("Total ", len(source_ids))
    print("# source_id_520\ttotal_lumin\tstar_lumin")

    for i in range(len(source_ids)):
        # j = np.where(source_ids_2 == source_ids[i])[0][0]

        sed = seds.get_sed(model_names[i])

        sed = sed.scale_to_av(fitted_av[i], extinction.get_av)  # To include AV

        x = sed.wav.to(u.Hz, equivalencies=u.spectral())
        y = sed.flux[-1].to(u.watt / u.m / u.m / u.hertz)

        # Total luminosity in terms of Ls units by integrating total SED
        integral = integrate_subset(x, y, np.min(x), np.max(x)) * 4 * np.pi * np.power(sed.distance.to(u.m), 2)
        star_total_lumin = integral / const.L_sun

        # Star's flux by using fitted temperature and radius using Steffan-Boltzmann law
        star_only_lumin = const.sigma_sb * 4 * np.pi * np.power(star_radii[i] * const.R_sun, 2) * np.power(star_temp[i] * u.K, 4) / const.L_sun
        star_only_lumin = star_only_lumin

        print(source_ids[i], star_total_lumin, star_only_lumin)

        # If needs to plot use this:
        """
        sed = seds.get_sed(model_names[i])
        # sed_2 = seds_2.get_sed(model_names_2[j])

        sed = sed.scale_to_av(fitted_av[i], extinction.get_av)  # To include AV
        # sed_2 = sed_2.scale_to_av(fitted_av_2[i], extinction.get_av)  # To include AV

        plt.loglog(sed.wav, sed.flux[-1], color='red')
        # plt.loglog(sed_2.wav, sed_2.flux[-1], color='blue')
        plt.xlim(0.1, 20)
        plt.ylim(1e-4, 50)  # Adjust y limit if necessary
        plt.show()
        plt.close()"""

