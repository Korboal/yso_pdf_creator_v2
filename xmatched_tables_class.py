import os

class XMatchTables:
    def __init__(self, extra_catalogues_dir: dict):
        self.extra_catalogues_dict = extra_catalogues_dir
        self.all_table_data = None
        self.table_names = None

    def __setstate__(self):
        from tools import load_fits_table
        self.table_names = self.extra_catalogues_dict["table_names"]
        self.all_table_data = []
        for table in self.table_names:  # load each individual table for a specific star
            if os.path.isfile(self.extra_catalogues_dict[table]):
                vot_table_data, _ = load_fits_table(self.extra_catalogues_dict[table])
                self.all_table_data.append(vot_table_data)
            else:
                print("ERROR, table", table, "NOT found. Loading of the table will not be done.")
