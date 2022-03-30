import numpy as np
import os
import tools
from astropy.io.votable import parse_single_table

# Program to rename lightcurves from their names to their *source_id*.xml.
# Assuming that each light curve .xml file containts light curve for only 1 star
# source_id is used to name the files.
# Rename following two lines for old and new location for file

files_to_rename_path = "xml_files/all_old/"     # Location of files to rename
files_new_location_path = "xml_files/all/"      # New files location

xml_files = tools.get_all_file_names_in_a_folder(files_to_rename_path)
files_amount = np.size(xml_files)

for i in range(files_amount):
    print(int(i / files_amount * 100), "%")
    file = files_to_rename_path + xml_files[i]
    votable = parse_single_table(file)
    ids = votable.array["source_id"].astype(str)
    source_id = ids[0]

    old_file = os.path.join(file)
    # If you want different unique name for each light curve, change the following line:
    new_file = os.path.join(files_new_location_path + str(source_id) + ".xml")
    os.rename(old_file, new_file)
