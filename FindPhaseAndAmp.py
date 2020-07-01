import scipy.io as sio
import numpy as np
import matplotlib.pyplot as plt
import scipy.interpolate
from scipy.interpolate import interp1d
from scipy import signal
import h5py
from numba import njit, prange

@njit(parallel=True)
def find_phase_amp(data, filling1, lut, lut_len, lut_start ,n , t, phasebalance, lut_shift):
    bunchindexscan = np.arange(720)
    bunchsize = 40
    #turnnum = 7000
    turnnum = int(np.floor(len(data) / 720 / t)) - 1
    lutmatrix = np.empty((40, lut_len))
    bunchdata = np.empty((40, turnnum))
    bunchphase0 = np.empty(turnnum)
    bunchphasefit = np.empty(turnnum)
    bunchdatafit = np.empty((40, turnnum))
    bunchphase = np.empty((turnnum, n))
    bunchamp = np.empty((n, turnnum))
    tmp3 = np.empty(2000)
    for j in range(720):
        print(j)
        if (filling1[j] == 1):
            bunchindex = bunchindexscan[j]
            # define the data index for the bunch in the first turn
            dataindexstart = int(np.floor(bunchindex * t))
            # build lut matrix for the bunch  # bunchindex
            tmp1 = lut[:, bunchindex]
            tmp2 = np.concatenate((tmp1, tmp1, tmp1))
            for i in range(lut_len):
                lutmatrix[:, i] = tmp2[np.arange(
                    lut_start + i + 1000, lut_start + i + 20000 + 1000, 500)]
            # collect the bunch data using the final t value
            #dataindexs = np.floor(np.arange(turnnum) * 720 *t).astype("int32") + dataindexstart
            exdataindexs = np.floor(np.arange(turnnum) * 720 *t) + dataindexstart
            dataindexs = np.arange(len(exdataindexs))
            for index in range(len(exdataindexs)):
                dataindexs[index] = int(exdataindexs[index])
            dataindexe = dataindexs + bunchsize
            for i in range(turnnum):
                bunchdata[:, i] = data[np.arange(
                    dataindexs[i], dataindexe[i])].reshape((bunchsize,))
                bunchphase0[i] = (dataindexs[i] - i * t * 720 - bunchindex * t) * 50
                datamatrix = np.repeat(bunchdata[:,i],lut_len).reshape(40,2000)
                correlations = lutmatrix * datamatrix
                for nm in prange(2000):
                    tmp3[nm] = np.mean(correlations[:,nm])
                #tmp3 = np.mean(correlations, axis=0)
                ind = np.argmax(tmp3)
                # print("ind=", ind)
                bunchdatafit[:, i] = lutmatrix[:, ind]
                bunchphasefit[i] = ind * 0.1
                bunchphase[i, j] = bunchphase0[i] - bunchphasefit[i] - phasebalance[bunchindex] + lut_shift[bunchindex]
                sizen = 40
                k1 = np.sum(bunchdatafit[:, i]* bunchdata[:, i])
                k2 = np.sum(bunchdatafit[:, i])
                k3 = np.sum(bunchdata[:, i])
                k4 = np.sum(bunchdatafit[:, i]*bunchdatafit[:, i])
                bunchamp[j, i] = (k1 - k2 * k3 / sizen)/(k4 - k2 * k2 / sizen )
                #z1 = np.polyfit(bunchdatafit[:, i], bunchdata[:, i], 1)
                #bunchamp[j, i] = z1[0]
            #if bunchamp[j, 1] > 0.000001 and bunchamp[j, 1] < 0.2:
                #print("error")
    return bunchphase, bunchamp



# Get the right RF frequency information from BPM signal
f = h5py.File(
        r'20200220_213mA_AC_inject_6.h5',
        'r')
BPM1 = f['Waveforms']['Channel 1']['Channel 1Data'][()].astype("int32")

BPM3 = f['Waveforms']['Channel 3']['Channel 3Data'][()].astype("int32")
T = np.load("T.npy")
LUT1 = np.load("LUT1.npy")
LUT2 = np.load("LUT2.npy")
LUT1_shift = np.load("LUT1_shift.npy")
LUT2_shift = np.load("LUT2_shift.npy")
PhaseBalance = np.load("PhaseBalance.npy")
filling = np.zeros(720,)
for i in range(720):
    if(max(LUT2[:,i])>2000):
        filling[i] = 1

# define the peak index of the first bunch
flagForBaseline = False
PeakIndex = 0
for i in range(np.size(BPM1) // 100):
    if(flagForBaseline == False):
        if((max(BPM1[i * 100:(i + 1) * 100]) - min(BPM1[i * 100:(i + 1) * 100])) < 4000):
            flagForBaseline = True
    elif((max(BPM1[i * 100:(i + 1) * 100]) - min(BPM1[i * 100:(i + 1) * 100])) > 4000):
        for k in range(270):
            if((BPM1[(i - 1) * 100 + k + 10] == min(BPM1[(i - 1) * 100 + k:(i - 1) * 100 + k + 20])) and (max(BPM1[(i - 1) * 100 + k:(i - 1) * 100 + k + 20]) - min(BPM1[(i - 1) * 100 + k:(i - 1) * 100 + k + 20])) > 4000):
                PeakIndex = (i - 1) * 100 + k + 10
                break
        break

PeakIndex1 = 0
flagForBaseline = False
for i in range(100,np.size(BPM1) // 100):
    if(flagForBaseline == False):
        if((max(BPM1[i * 100:(i + 1) * 100]) - min(BPM1[i * 100:(i + 1) * 100])) < 4000):
            flagForBaseline = True
    elif((max(BPM1[i * 100:(i + 1) * 100]) - min(BPM1[i * 100:(i + 1) * 100])) > 4000):
        for k in range(270):
            if((BPM1[(i - 1) * 100 + k + 10] == min(BPM1[(i - 1) * 100 + k:(i - 1) * 100 + k + 20])) and (max(BPM1[(i - 1) * 100 + k:(i - 1) * 100 + k + 20]) - min(BPM1[(i - 1) * 100 + k:(i - 1) * 100 + k + 20])) > 4000):
                PeakIndex1 = (i - 1) * 100 + k + 10
                break
        break

BaselineIndex = np.arange(PeakIndex1 - 1288, PeakIndex1 - 688)

Filling = np.arange(720)

# define the basic number
HarmonicNum = 720
# initial bucket size value, sampling rate 20GHz, 40*50ps = 2ns
BunchSize = 40
# define which bunch will be processed
BunchIndexScan = Filling

# copy raw data, processing pickup #1
Data = BPM1[PeakIndex - 10:].copy()
LUT = LUT1
LUT_shift = LUT1_shift
# remove DC offset
Baseline = np.mean(Data[BaselineIndex], dtype="float64")
Data = Data - Baseline
# for loop to process all defined bunch
N = len(BunchIndexScan)
LUTlength = 2000
LUTstart = 18000
bunchphase1, bunchamp1 = find_phase_amp(Data, filling, LUT, LUTlength, LUTstart, N, T, PhaseBalance, LUT_shift)

# copy raw data, processing pickup #1
Data = BPM3[PeakIndex - 10:].copy()
LUT = LUT2
LUT_shift = LUT2_shift
# remove DC offset
Baseline = np.mean(Data[BaselineIndex], dtype="float64")
Data = Data - Baseline
# for loop to process all defined bunch
N = len(BunchIndexScan)
LUTlength = 2000
LUTstart = 18000
bunchphase3, bunchamp3 = find_phase_amp(Data, filling, LUT, LUTlength, LUTstart, N, T, PhaseBalance, LUT_shift)

np.save("BunchPhase1", bunchphase1)
np.save("BunchPhase3", bunchphase3)
np.save("BunchAmp1", bunchamp1)
np.save("BunchAmp3", bunchamp3)






TurnNum = np.floor(len(Data) / 720 / T).astype("int32") - 1
LutMatrix = np.zeros((40, LUTlength)).astype("float64")
BunchData = np.zeros((40, TurnNum)).astype("float64")
BunchPhase0 = np.zeros(TurnNum).astype("float64")
DataMatrix = np.zeros((40, LUTlength)).astype("float64")
BunchPhaseFit = np.zeros(TurnNum).astype("float64")
BunchDataFit = np.zeros((40, TurnNum)).astype("float64")
BunchPhase = np.zeros((TurnNum, N)).astype("float64")
BunchAmp = np.zeros((N, TurnNum)).astype("float64")
TurnNum = np.floor(len(Data) / 720 / T).astype('int32') - 1
for j in range(N):
    print(j)
    if(filling[j] ==1):
        BunchIndex = BunchIndexScan[j]
        # define the data index for the bunch in the first turn
        DataIndexStart = np.floor(BunchIndex * T).astype("int32")
        # build LUT matrix for the bunch  # BunchIndex
        tmp1 = LUT[:, BunchIndex]
        tmp2 = np.concatenate((tmp1, tmp1, tmp1))
        for i in range(LUTlength):
            LutMatrix[:, i] = tmp2[np.arange(LUTstart + i + 1000, LUTstart + i + 20000 + 1000, 500)]
        # collect the bunch data using the final T value
        DataIndexS = np.floor(np.arange(TurnNum) * 720 *
                              T).astype("int32") + DataIndexStart
        DataIndexE = DataIndexS + BunchSize
        for i in range(TurnNum):
            BunchData[:, i] = Data[np.arange(
                DataIndexS[i], DataIndexE[i])].reshape((BunchSize,))
            BunchPhase0[i] = (DataIndexS[i] - i * T * 720 - BunchIndex * T) * 50
            DataMatrix = np.transpose(np.tile(BunchData[:, i], (LUTlength, 1)))
            tmp3 = np.mean(LutMatrix * DataMatrix, axis=0)
            ind = np.argmax(tmp3)
            #print("ind=", ind)
            BunchDataFit[:, i] = LutMatrix[:, ind]
            BunchPhaseFit[i] = ind * 0.1
            BunchPhase[i, j] = BunchPhase0[i] - BunchPhaseFit[i] - PhaseBalance[BunchIndex]
            z1 = np.polyfit(BunchDataFit[:, i], BunchData[:, i], 1)
            BunchAmp[j, i] = z1[0]
        if(BunchAmp[j,1]>0.000001 and BunchAmp[j,1]<0.2):
            print("error")

BunchPhase1 = np.copy(BunchPhase)
BunchAmp1 = np.copy(BunchAmp)

# copy raw data, processing pickup #3
Data = BPM3[PeakIndex - 10:].copy()
LUT = LUT2
# remove DC offset
Baseline = np.mean(Data[BaselineIndex], dtype="float64")
Data = Data - Baseline
# for loop to process all defined bunch
N = len(BunchIndexScan)
LUTlength = 2000
LUTstart = 18000
for j in range(N):
    print(j)
    if(filling[j]==1):
        BunchIndex = BunchIndexScan[j]
        # define the data index for the bunch in the first turn
        DataIndexStart = np.floor(BunchIndex * T).astype("int32")
        # build LUT matrix for the bunch  # BunchIndex
        tmp1 = LUT[:, BunchIndex]
        tmp2 = np.concatenate((tmp1, tmp1, tmp1))
        for i in range(LUTlength):
            LutMatrix[:, i] = tmp2[np.arange(
                LUTstart + i + 1000, LUTstart + i + 20000 + 1000, 500)]
        # collect the bunch data using the final T value
        DataIndexS = np.floor(np.arange(TurnNum) * 720 *
                              T).astype("int32") + DataIndexStart
        DataIndexE = DataIndexS + BunchSize
        for i in range(TurnNum):
            BunchData[:, i] = Data[np.arange(
                DataIndexS[i], DataIndexE[i])].reshape((BunchSize,))
            BunchPhase0[i] = (DataIndexS[i] - i * T * 720 - BunchIndex * T) * 50
            DataMatrix = np.tile(BunchData[:, i], (LUTlength, 1)).T
            tmp3 = np.mean(LutMatrix * DataMatrix, axis=0)
            ind = np.argmax(tmp3)
            # print("ind=", ind)
            BunchDataFit[:, i] = LutMatrix[:, ind]
            BunchPhaseFit[i] = ind * 0.1
            BunchPhase[i, j] = BunchPhase0[i] - BunchPhaseFit[i] - PhaseBalance[BunchIndex]
            z1 = np.polyfit(BunchDataFit[:, i], BunchData[:, i], 1)
            BunchAmp[j, i] = z1[0]

BunchPhase3 = np.copy(BunchPhase)
BunchAmp3 = np.copy(BunchAmp)

np.save("BunchPhase1", BunchPhase1)
np.save("BunchPhase3", BunchPhase3)
np.save("BunchAmp1", BunchAmp1)
np.save("BunchAmp3", BunchAmp3)
