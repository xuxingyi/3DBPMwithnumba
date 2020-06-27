import h5py


f = h5py.File(r'C:/Users/74506/Desktop/数据处理脚本及信号处理流程演示/0409/211mA_16.h5','r')
BPM1 = f['Waveforms']['Channel 1']['Channel 1Data'][()]



f.close()

for a1 in a:
    a1 = a1+5