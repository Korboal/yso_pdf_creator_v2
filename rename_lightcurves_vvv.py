import numpy as np
import os
import tools
import pandas as pd
import config_file as cf

# Program to rename lightcurves from their names to their *source_id*.xml.
# Assuming that each light curve .xml file containts light curve for only 1 star
# source_id is used to name the files.
# Rename following two lines for old and new location for file

files_to_rename_path = "input_lightcurves/all/vvvdr4/unnamed/"     # Location of files to rename
files_new_location_path = "input_lightcurves/all/vvvdr4/"      # New files location

light_curve_files = tools.get_all_file_names_in_a_folder(files_to_rename_path)
files_amount = np.size(light_curve_files)

tb = pd.read_csv(cf.xmatched_new_catalogues_directory + "VVVDR4" + "_" + cf.file_category + ".csv")
gdr2_source_ids = list(tb["source_id_gdr2"].values.astype(str))
vvv_source_ids = list(tb["sourceID"].values.astype(str))


for i in range(files_amount):
    #print(int(i / files_amount * 100), "%")

    try:
        file = files_to_rename_path + light_curve_files[i]
        votable = tools.load_fits_table(file)
        ids = votable[0]["SOURCEID"].astype(str)

        source_id = ids[0]

        try:
            result = vvv_source_ids.index(source_id)
            gdr2_id = gdr2_source_ids[result]

            print(len(ids))

            old_file = os.path.join(file)
            # If you want different unique name for each light curve, change the following line:
            new_file = os.path.join(files_new_location_path + str(gdr2_id) + ".fits")
            os.rename(old_file, new_file)
        except:
            pass
    except:
        print(0)
