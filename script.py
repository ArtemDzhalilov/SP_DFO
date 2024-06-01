import os
import re
import mne
import matplotlib.pyplot as plt
dirs = os.listdir()
from pathlib import Path
import neurokit2 as nk
mne.set_log_level('error')
def read_data(path):
    files = os.listdir(path)

    files = [os.path.join(path, f) for f in files if f.endswith(".EDF")]
    for file in files:
        header = mne.io.read_raw_edf(os.getcwd()+"/" + file, )
        header.load_data()
        #header.compute_psd().plot()
        low_cut = 0.1
        hi_cut = 30

        #mne.io.Raw.plot(header, start=3, duration=0.2, color='blue', show=True, n_channels=1)
        #channel = 'Fp1-M2'
        #start_time = 1.0  # in seconds
        #end_time = 1.1

        #fig, ax = plt.subplots(figsize=[15, 5])
        #ax.plot(header.get_data(picks=channel, tmin=start_time, tmax=end_time).T)
        #plt.show()
        #header.plot()
        #print(mne.events_from_annotations(header))
        return header
dir_path = os.getcwd()
def execute(path):
    def replace_time_format(text):
        pattern = r'(\d+\.\d+\.\d+):(\d+):(\d+)'
        return re.sub(pattern, r'\1.\2.\3', text)

    files = os.listdir(path)
    #print(files)
    paths1 = [os.path.join(path, f) for f in files if f.endswith(".REC")]
    if len(paths1)==0:
        return 0
    old_filename = Path(paths1[0])
    new_extension = '.EDF'

    # Создаем новое имя файла с новым расширением
    new_filename = old_filename.with_suffix(new_extension)

    # Переименовываем файл
    try:
        old_filename.rename(new_filename)
    except:
        pass
    paths = [os.path.join(path, f) for f in files if f.endswith(".EDF")]
    for path in paths:
        #print(1)
        with open(os.getcwd() +'/' + path, 'r+', encoding='latin') as f:
            line = f.read()
            f.seek(0)
            f.write(replace_time_format(line))
            f.close()
        print(path)


dirs_filtered = [dir for dir in dirs if 'N' in dir]
dirs_filtered_2 = []

for s in dirs_filtered:
  dirs_n = os.listdir(s)
  dirs_filtered_new = [s + "\\" + dir for dir in dirs_n if 'N' in dir]
  dirs_filtered_2.append(dirs_filtered_new)
data = []

for i in range(len(dirs_filtered_2)):
    for j in range(len(dirs_filtered_2[i])):
        execute(dirs_filtered_2[i][j])
        data.append(read_data(dirs_filtered_2[i][j]))


#for i in range(len(data)):
    #print(data[i].info)