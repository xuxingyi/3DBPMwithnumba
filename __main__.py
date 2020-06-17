import numpy as np
import numba as nb
import h5py
import matplotlib.pyplot as plt
import scipy.interpolate


def peak(bpm):
    flag_for_baseline = False
    peak_index = -1
    for i in range(np.size(bpm) // 100):
        if not flag_for_baseline:
            if (max(bpm[i * 100:(i + 1) * 100]) -
                    min(bpm[i * 100:(i + 1) * 100])) < 4000:
                #print(flag_for_baseline)
                flag_for_baseline = True
        elif (max(bpm[i * 100:(i + 1) * 100]) - min(bpm[i * 100:(i + 1) * 100])) > 4000:
            for k in range(270):
                if (bpm[(i - 1) * 100 + k + 10] == min(bpm[(i - 1) * 100 + k:(i - 1) * 100 + k + 20])) and (
                        max(bpm[(i - 1) * 100 + k:(i - 1) * 100 + k + 20]) - min(
                        bpm[(i - 1) * 100 + k:(i - 1) * 100 + k + 20])) > 4000:
                    peak_index = (i - 1) * 100 + k + 10
                    break
            break
    return peak_index

def baseline(bpm):
    flag_for_baseline = False
    peak_index1 = -1
    for i in range(100, np.size(bpm) // 100):
        if not flag_for_baseline:
            if (max(bpm[i * 100:(i + 1) * 100]) -
                    min(bpm[i * 100:(i + 1) * 100])) < 4000:
                flag_for_baseline = True
        elif (max(bpm[i * 100:(i + 1) * 100]) - min(bpm[i * 100:(i + 1) * 100])) > 4000:
            for k in range(270):
                if (bpm[(i - 1) * 100 + k + 10] == min(bpm[(i - 1) * 100 + k:(i - 1) * 100 + k + 20])) and (
                        max(bpm[(i - 1) * 100 + k:(i - 1) * 100 + k + 20]) - min(
                        bpm[(i - 1) * 100 + k:(i - 1) * 100 + k + 20])) > 4000:
                    peak_index1 = (i - 1) * 100 + k + 10
                    break
            break
    baseline_index = np.arange(peak_index1 - 1688, peak_index1 - 688)
    baseline_value = np.mean(bpm[baseline_index], dtype="float64")
    return baseline_value


def calculate_t_by_bunchnumbers(data, t):
    # calculate T value by counting the bunch numbers
    turn_num_t = np.size(data) // 28000000
    turn_num1 = np.array([2, 5, 10, 20, 100, 200, 500])
    turn_num2 = np.arange(1, turn_num_t + 1)
    scan_turn_num = np.hstack((turn_num1, turn_num2 * 1000))
    n = len(scan_turn_num)
    for j in range(n):
        turn_num = scan_turn_num[j]
        data_index_s = np.floor(np.arange(turn_num) * 720 *
                              t).astype("int32")
        data_index_e = data_index_s + BunchSize
        # collect specified bunch data together
        bunch_data_first = data[data_index_s[0]:data_index_e[0]]
        bunch_data_end = data[data_index_s[turn_num - 1]:data_index_e[turn_num - 1]]
        # find the peak index for each turn of this bunch
        IndEnd = np.argmin(bunch_data_end)
        IndFirst = np.argmin(bunch_data_first)
        Peak1Index = IndFirst + data_index_s[0] - 1
        PeakEndIndex = IndEnd + data_index_s[-1] - 1
        # calculate the T using peak index
        t = (PeakEndIndex - Peak1Index) / 720 / (turn_num - 1)
    return(t)






if __name__ == '__main__':
    f = h5py.File(
        r'C:/Users/74506/Desktop/数据处理脚本及信号处理流程演示/0409/211mA_16.h5',
        'r')
    BPM1 = f['Waveforms']['Channel 1']['Channel 1Data'][()].astype("int32")
    Peak_index = peak(BPM1)
    Base_line = baseline(BPM1)
    Data = BPM1[Peak_index - 10:].copy() - Base_line

    # define the basic number
    HarmonicNum = 720
    # initial bucket size value, sampling rate 20GHz, 40*50ps = 2ns
    T = 40
    BunchSize = 40
    # define which bunch will be processed
    BunchIndex = 0
    # define the data index for the first bunch and the dirst turn
    DataIndexStart = BunchIndex * BunchSize
    T = calculate_t_by_bunchnumbers(Data, T)






