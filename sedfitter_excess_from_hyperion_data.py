from hyperion.model import ModelOutput
from hyperion.util.constants import pc
import numpy as np
from astropy import units as u
from sedfitter.utils import integrate_subset

# Hyperion data: uses the Robitaille 2017 YSO models and extracts dust and total SED emission.
# The dust emission is divided by total SED emission (i.e. star + scattered light + dust), thus giving
# which fraction of SED is the dust emission. Hyperion data allows to extract that original dust + total SED.
# Hyperion package: Mac/Linux only (i.e. not Windows). Requires original grid1.1 data, which is heavier
# than the original models.
# Version where RATIO of dust to total is found in lambdaF(lambda) VS lambda linear space

if __name__ == '__main__':
    fitted_data = np.loadtxt('sedfitter/all_params_2021-12-21_all_r17_spu_smi_av5_new_parallax_1pc_0.3upper_n0.5slope.txt', dtype=str)
    source_ids = fitted_data[:, 0]
    model_names = fitted_data[:, 4]
    chi2 = fitted_data[:, 5].astype(float)
    fitted_av = fitted_data[:, 6].astype(float)
    fitted_scale = fitted_data[:, 7].astype(float)
    star_radii = fitted_data[:, 8].astype(float)
    star_temp = fitted_data[:, 9].astype(float)
    inclination = fitted_data[:, 16].astype(float)  # sp__s_i model
    inclination = fitted_data[:, 20].astype(float)  # spu_smi model

    print("Total sources:", len(source_ids))

    print("# source_id_520\tfull_dust_emission\tdust_emission_up_to_25")

    for i in range(len(source_ids)):
        model_name_to_use = model_names[i]
        inclination_to_use = inclination[i]
        fitted_scale_to_use = fitted_scale[i]

        # sp__s_i model:
        #model = ModelOutput('grids-1.1/sp--s-i/output/' + model_name_to_use[0:2].lower() + "/" + model_name_to_use[:-3] + '.rtout')
        # spu_smi model:
        model = ModelOutput('grids-1.1/spu-smi/output/' + model_name_to_use[0:2].lower() + "/" + model_name_to_use[:-3] + '.rtout')

        inclination_type = int(inclination_to_use / 10)
        aperture = -1   # Take biggest aperture
        sed_total_sed = model.get_sed(inclination=inclination_type, aperture=aperture, distance=1000 * pc)  # Total SED
        #sed_direct_stellar = model.get_sed(inclination=inclination_type, aperture=aperture, distance=1000 * pc, component='source_emit')  # Direct stellar photons
        #sed_scattered_stellar = model.get_sed(inclination=inclination_type, aperture=aperture, distance=1000 * pc, component='source_scat')  # Scattered stellar photons
        sed_direct_dust = model.get_sed(inclination=inclination_type, aperture=aperture, distance=1000 * pc, component='dust_emit')  # Direct dust photons
        #sed_scattered_dust = model.get_sed(inclination=inclination_type, aperture=aperture, distance=1000 * pc, component='dust_scat')  # Scattered dust photons

        # Extract the dust flux from SED
        x = (sed_direct_dust.wav * u.um).to(u.m)    # Convert wavelength to meters
        x = x / x.unit                              # For integration use without units
        y = (sed_direct_dust.val * u.erg / u.s / u.cm / u.cm).to(u.watt / u.m / u.m)    # Convert flux to W/m^2
        y = y / y.unit

        # Extract the total SED flux
        x_total = (sed_total_sed.wav * u.um).to(u.m)
        x_total = x_total / x_total.unit
        y_total = (sed_total_sed.val * u.erg / u.s / u.cm / u.cm).to(u.watt / u.m / u.m)
        y_total = y_total / y_total.unit

        if len(x) != 0:     # To make sure there is emission from dust
            full_dust_emission = integrate_subset(x, y, np.min(x), np.max(x))
            full_sed_emission = integrate_subset(x_total, y_total, np.min(x_total), np.max(x_total))
        else:
            full_dust_emission = 0
            full_sed_emission = 1

        # Now we want to limit flux from 0 to 25 um; the dust flux
        x = sed_direct_dust.wav                 # Units are micrometers
        indices_to_use = np.where(x <= 25.0)    # Indices where wavelength is below 25 um
        x = x[indices_to_use]
        x = (x * u.um).to(u.m)                  # Convert to meters
        x = x / x.unit
        y = sed_direct_dust.val
        y = y[indices_to_use]                   # Only take flux below 25 um
        y = (y * u.erg / u.s / u.cm / u.cm).to(u.watt / u.m / u.m)
        y = y / y.unit

        # Total SED flux
        x_total = sed_total_sed.wav                 # Micrometers
        indices_to_use = np.where(x_total <= 25.0)  # Only where below 25 um
        x_total = x_total[indices_to_use]
        x_total = (x_total * u.um).to(u.m)
        x_total = x_total / x_total.unit
        y_total = sed_total_sed.val
        y_total = y_total[indices_to_use]
        y_total = (y_total * u.erg / u.s / u.cm / u.cm).to(u.watt / u.m / u.m)
        y_total = y_total / y_total.unit

        if len(x) != 0:     # Make sure there is any flux at all
            dust_emission_25 = integrate_subset(x, y, np.min(x), np.max(x))
            sed_emission_25 = integrate_subset(x_total, y_total, np.min(x_total), np.max(x_total))
        else:
            dust_emission_25 = 0
            sed_emission_25 = 1

        print(source_ids[i], full_dust_emission / full_sed_emission, dust_emission_25 / sed_emission_25)
