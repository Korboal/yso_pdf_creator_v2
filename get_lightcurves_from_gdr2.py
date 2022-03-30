import numpy as np
import webbrowser
import time
import tools

gdr2_ids = tools.load_ids_to_analyse("ids/_yso_ids.txt")

for i in range(np.size(gdr2_ids)):
    url = "https://geadata.esac.esa.int/data-server/data?ID=Gaia+DR2+" + str(gdr2_ids[i]) + "&RETRIEVAL_TYPE=EPOCH_PHOTOMETRY"
    #print(gdr2_ids[i])
    webbrowser.open_new(url)
    time.sleep(1)