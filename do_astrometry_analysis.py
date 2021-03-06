import numpy as np
import tools
import config_file as cf


def do_nebula_pm_analysis(filename_stars_with_nebulae, nebulae_data):
    min_stars_to_analyse_nebula = 3
    max_ratio_of_pm_to_be_ok = 1.3
    max_angle_between_star_and_neb_pm_to_be_ok = 30  # degrees
    max_ratio_between_star_and_neb_pm_to_be_ok = 1.5

    stdevs_to_be_ok = 5

    #pm_nebulae_data = tools.load_data("mol_clouds/nebula_pm_v2/cantat_gaudin_2020_members_g2_x_ge3_low_err_over100.txt")

    pm_nebulae_data = tools.load_several_txt_files("mol_clouds/pm_nebula_v2/")

    stars_with_nebulae_data = tools.load_data(filename_stars_with_nebulae)
    stars_with_nebulae = stars_with_nebulae_data[:, 6]

    list_star_membership_data = tools.listdir("mol_clouds/stars_for_pm_nebula/")

    for i in range(np.size(list_star_membership_data)):
        list_star_membership_data[i] = list_star_membership_data[i][6:][:-4].lower()

    #nebulae_star_membership_data = tools.load_data("mol_clouds/stars_for_pm_nebula/stars_taurus.txt")

    unique_nebulae = np.unique(stars_with_nebulae)

    print(f"Amount of SFR to go through {np.size(unique_nebulae)}")

    for nebula_name in unique_nebulae:
        print(nebula_name)
        arg_with_neb = np.where(stars_with_nebulae == nebula_name)
        if nebula_name.lower() in np.char.lower(pm_nebulae_data[:, 0]):
            #print(f"HELLO we found your nebula {nebula_name}")
            index_pm_to_use = np.where(np.char.lower(pm_nebulae_data[:, 0]) == nebula_name.lower())[0][0]
            nebula_pmra = float(pm_nebulae_data[:, 1][index_pm_to_use])
            nebula_pmra_std = float(pm_nebulae_data[:, 2][index_pm_to_use])
            nebula_pmdec = float(pm_nebulae_data[:, 3][index_pm_to_use])
            nebula_pmdec_std = float(pm_nebulae_data[:, 4][index_pm_to_use])
        else:
            nebula_pmra, nebula_pmdec = 0, 0
            nebula_pmra_std, nebula_pmdec_std = 0, 0
        """if True:
            nebula_name_temp = nebula_name.lower()
            if nebula_name_temp in list_star_membership_data:
                nebulae_star_membership_data = tools.load_data(f"mol_clouds/stars_for_pm_nebula/stars_{nebula_name}.txt")
                pmra_neb_all_stars = nebulae_star_membership_data[:, 9].astype(float)
                pmdec_neb_all_stars = nebulae_star_membership_data[:, 11].astype(float)

                nebula_pmra = np.median(pmra_neb_all_stars)
                nebula_pmdec = np.median(pmdec_neb_all_stars)
            else:
                nebula_pmra, nebula_pmdec = 0, 0"""

        if abs(nebula_pmra) > 0 and abs(nebula_pmdec) > 0:
            pm_neb_length = np.sqrt(np.square(nebula_pmra) + np.square(nebula_pmdec))

            neb_data = np.where(nebulae_data[:, 0] == nebula_name)
            ra_neb = np.mean(nebulae_data[neb_data, 1].astype(float))
            dec_neb = np.mean(nebulae_data[neb_data, 2].astype(float))
            dist_neb = np.mean(nebulae_data[neb_data, 3].astype(float))
            size_neb = np.mean(nebulae_data[neb_data, 4].astype(float))

            tools.save_in_txt_topcat([nebula_name, ra_neb, dec_neb, dist_neb, size_neb, nebula_pmra, nebula_pmdec, nebula_pmra_std, nebula_pmdec_std,
                                      int(np.size(arg_with_neb))], cf.output_textfile_nebulae_only_pm)

            for star_index in list(arg_with_neb[0]):
                pmra_star = stars_with_nebulae_data[star_index, 3].astype(float)
                pmdec_star = stars_with_nebulae_data[star_index, 4].astype(float)

                pm_star_length = np.sqrt(np.square(pmra_star) + np.square(pmdec_star))

                angle_pm_star_neb = np.arccos((pmra_star * nebula_pmra + pmdec_star * nebula_pmdec) / pm_neb_length / pm_star_length) / np.pi * 180
                ratio_pm_star_neb = max(pm_star_length, pm_neb_length) / min(pm_star_length, pm_neb_length)

                if nebula_pmra - abs(stdevs_to_be_ok) * nebula_pmra_std < pmra_star < nebula_pmra + abs(stdevs_to_be_ok) * nebula_pmra_std and \
                        nebula_pmdec - abs(stdevs_to_be_ok) * nebula_pmdec_std < pmdec_star < nebula_pmdec + abs(stdevs_to_be_ok) * nebula_pmdec_std:
                    part_of_neb = "yes"
                else:
                    part_of_neb = "no"
                tools.save_in_txt_topcat(
                    [stars_with_nebulae_data[star_index, 0], stars_with_nebulae_data[star_index, 1],
                     stars_with_nebulae_data[star_index, 2],
                     stars_with_nebulae_data[star_index, 3], stars_with_nebulae_data[star_index, 4],
                     stars_with_nebulae_data[star_index, 5],
                     stars_with_nebulae_data[star_index, 6], stars_with_nebulae_data[star_index, 7],
                     stars_with_nebulae_data[star_index, 8],
                     part_of_neb], cf.output_textfile_nebulae_star_pm)
        else:
            nebula_pmra = 0
            nebula_pmdec = 0

            for star_index in list(arg_with_neb[0]):
                part_of_neb = "??"
                tools.save_in_txt_topcat(
                    [stars_with_nebulae_data[star_index, 0], stars_with_nebulae_data[star_index, 1],
                     stars_with_nebulae_data[star_index, 2],
                     stars_with_nebulae_data[star_index, 3], stars_with_nebulae_data[star_index, 4],
                     stars_with_nebulae_data[star_index, 5],
                     stars_with_nebulae_data[star_index, 6], stars_with_nebulae_data[star_index, 7],
                     stars_with_nebulae_data[star_index, 8],
                     part_of_neb], cf.output_textfile_nebulae_star_pm)

            neb_data = np.where(nebulae_data[:, 0] == nebula_name)
            ra_neb = np.mean(nebulae_data[neb_data, 1].astype(float))
            dec_neb = np.mean(nebulae_data[neb_data, 2].astype(float))
            dist_neb = np.mean(nebulae_data[neb_data, 3].astype(float))
            size_neb = np.mean(nebulae_data[neb_data, 4].astype(float))

            tools.save_in_txt_topcat([nebula_name, ra_neb, dec_neb, dist_neb, size_neb, nebula_pmra, nebula_pmdec,
                                      nebula_pmra_std, nebula_pmdec_std, int(np.size(arg_with_neb))],
                                      cf.output_textfile_nebulae_only_pm)
        """else:
            nebula_pmra = 0
            nebula_pmdec = 0

            for star_index in list(arg_with_neb[0]):
                part_of_neb = "??"
                tools.save_in_txt_topcat(
                    [stars_with_nebulae_data[star_index, 0], stars_with_nebulae_data[star_index, 1],
                     stars_with_nebulae_data[star_index, 2],
                     stars_with_nebulae_data[star_index, 3], stars_with_nebulae_data[star_index, 4],
                     stars_with_nebulae_data[star_index, 5],
                     stars_with_nebulae_data[star_index, 6], stars_with_nebulae_data[star_index, 7],
                     stars_with_nebulae_data[star_index, 8],
                     part_of_neb], cf.output_textfile_nebulae_star_pm)

            neb_data = np.where(nebulae_data[:, 0] == nebula_name)
            ra_neb = np.mean(nebulae_data[neb_data, 1].astype(float))
            dec_neb = np.mean(nebulae_data[neb_data, 2].astype(float))
            dist_neb = np.mean(nebulae_data[neb_data, 3].astype(float))
            size_neb = np.mean(nebulae_data[neb_data, 4].astype(float))

            tools.save_in_txt_topcat([nebula_name, ra_neb, dec_neb, dist_neb, size_neb, nebula_pmra, nebula_pmdec, int(np.size(arg_with_neb))],
                                     cf.output_textfile_nebulae_only_pm)"""