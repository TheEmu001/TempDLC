import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import numpy.polynomial.polynomial as poly

from scipy.spatial.distance import pdist, squareform
from sklearn.metrics.pairwise import euclidean_distances

# prevent numpy exponential
# notation on print, default False
np.set_printoptions(suppress=True)

# path = "Vglut-cre C137 F4+_2DLC_resnet50_VGlutEnclosedBehaviorApr25shuffle1_151500.csv"
# path = "Vglut-cre C137 F3-_2DLC_resnet50_VGlutEnclosedBehaviorApr25shuffle1_151500.csv"
# path = "Vglut-cre C162 F1DLC_resnet50_EnclosedBehaviorMay27shuffle1_307000.csv"
bodyparts = ['frameNo','bin_leftX', 'bin_leftY', 'bin_leftLike', 'bin_rightX', 'bin_rightY',
       'bin_rightLike', 'headX', 'headY', 'headLike', 'snoutX', 'snoutY', 'snoutLike', 'backX',
       'backY', 'backLike', 'left_earX', 'left_earY', 'left_earLike', 'right_earX',
       'right_earY', 'right_earLike', 'tail_baseX', 'tail_baseY', 'tail_baseLike',
       'left_forepawX', 'left_forepawY', 'left_forepawLike', 'right_forepawX',
       'right_forepawY', 'right_forepawLike', 'left_hindpawX', 'left_hindpawY',
       'left_hindpawLike', 'right_hindpawX', 'right_hindpawY', 'right_hindpawLike']

def distance_det(file, fps, legend_label, line_color):
    data_df = pd.read_csv(filepath_or_buffer=file, header=0, skiprows=3, names=bodyparts, low_memory=False)
    # calculate the time elapsed per frame and append column
    data_df['Time Elapsed'] = data_df['frameNo'] / fps

    # calculate the difference from row under to row before
    # then calculate absolute value
    data_df['|diff X|'] = data_df['snoutX'].diff(-1)
    data_df['|diff X|'] = data_df['|diff X|'].abs()

    data_df['|diff Y|'] = data_df['snoutY'].diff(-1)
    data_df['|diff Y|'] = data_df['|diff Y|'].abs()

    # calculating the cummulative sum down the column
    data_df['sumX'] = data_df['|diff X|'].cumsum()
    data_df['sumY'] = data_df['|diff Y|'].cumsum()

    # squaring delta x and y values
    data_df['deltax^2'] = data_df['|diff X|']**2
    data_df['deltay^2'] = data_df['|diff Y|']**2

    # adding deltaX^2 + deltaY^2
    data_df['deltaSummed'] = data_df['deltax^2'] + data_df['deltay^2']

    # taking square root of deltaX^2 + deltaY^2
    data_df['eucDist'] = data_df['deltaSummed']**(1/2)
    data_df['eucDistSum'] = data_df['eucDist'].cumsum()

    print(data_df)

    # what's being plotted
    # plt.plot(data_df['Time Elapsed'], data_df['sumX'],color='blue', marker='o', markersize=0.1, linewidth=0.1, label='xSum')
    # plt.plot(data_df['Time Elapsed'], data_df['sumY'],color='red', marker='o', markersize=0.1, linewidth=0.1, label='ySum')
    plt.plot(data_df['Time Elapsed'], data_df['eucDistSum'], color=line_color, marker='o', markersize=0.1, linewidth=0.1, label=legend_label)

    # plot formatting
    plt.xlabel('time (seconds)')
    plt.ylabel('distance travelled (pixels)')
    plt.legend(loc=2)
    # plt.title('total distance traveled vs. time: ' + path)
    animal = []
    animal[:] = ' '.join(file.split()[2:5])
    # plt.title('Total Distance vs. Time for: ' + ' '.join(file.split()[:2]) + " "+ ''.join(animal[:2]))
    plt.title('Cummulative Distance')
    leg = plt.legend()
    for i in leg.legendHandles:
        i.set_linewidth(3)


if __name__ == '__main__':

    """Saline Data"""
    distance_det(file='Paper_Redo_Saline_Ai14_OPRK1_C2_F0_Top DownDLC_resnet50_BigBinTopSep17shuffle1_250000filtered.csv',
                 fps=30, legend_label='F0 Saline', line_color='green')
    distance_det(file='Paper_Redo_Saline_Ai14_OPRK1_C2_F1_Top DownDLC_resnet50_BigBinTopSep17shuffle1_240000.csv',
                 fps=30, legend_label='F1 Saline', line_color='lime')
    distance_det(file='Paper_Redo_Saline_Ai14_OPRK1_C2_F2_Top DownDLC_resnet50_BigBinTopSep17shuffle1_250000filtered.csv',
                 fps=30, legend_label='F2 Saline', line_color='lightgreen')

    distance_det(file='Paper_Redo_Saline_Ai14_OPRK1_C1_M1_trial2_Top DownDLC_resnet50_BigBinTopSep17shuffle1_240000filtered.csv',
                 fps=30, legend_label='M1 Saline', line_color='green')
    distance_det(file='Paper_Redo_Saline_Ai14_OPRK1_C1_M2_trial2_Top DownDLC_resnet50_BigBinTopSep17shuffle1_240000filtered.csv',
                 fps=30, legend_label='M2 Saline', line_color='lime')
    distance_det(file='Paper_Redo_Saline_Ai14_OPRK1_C1_M3_trial2_Top DownDLC_resnet50_BigBinTopSep17shuffle1_240000filtered.csv',
                 fps=30, legend_label='M3 Saline', line_color='lightgreen')
    distance_det(file='Paper_Redo_Saline_Ai14_OPRK1_C1_M4_trial2_Top DownDLC_resnet50_BigBinTopSep17shuffle1_240000filtered.csv',
                 fps=30, legend_label='M4 Saline', line_color='lightgreen')


    """PreTreat Data"""
    # distance_det(file='Paper_Redo_10_18_PreTreat_Nal_5mgkg_U50_Ai14_OPRK1_C1_F0_Top DownDLC_resnet50_BigBinTopSep17shuffle1_250000filtered.csv',
    #              fps=60, legend_label='F0 Naltrexone', line_color='maroon')
    # distance_det(file='Paper_Redo_10_18_PreTreat_Nal_5mgkg_U50_Ai14_OPRK1_C1_F1_Top DownDLC_resnet50_BigBinTopSep17shuffle1_250000filtered.csv',
    #              fps=60, legend_label='F1 Naltrexone', line_color='crimson')
    # distance_det(file='Paper_Redo_10_18_PreTreat_Nal_5mgkg_U50_Ai14_OPRK1_C1_F2_Top DownDLC_resnet50_BigBinTopSep17shuffle1_250000filtered.csv',
    #              fps=60, legend_label='F2 Naltrexone', line_color='coral')

    # distance_det(file='Paper_Redo_PreTreat5mg_kgU50_Ai14_OPRK1_C1_M1_Top DownDLC_resnet50_BigBinTopSep17shuffle1_240000filtered.csv',
    #              fps=60, legend_label='M1 Naltrexone', line_color='gold')
    # distance_det(file='Paper_Redo_5mg_kgU50PreTreatNaltrexone_Ai14_OPRK1_C1_M2_Top DownDLC_resnet50_BigBinTopSep17shuffle1_240000.csv',
    #              fps=60, legend_label='M2 Naltrexone', line_color='goldenrod')
    # distance_det(file='Paper_Redo_5mg_kgU50PreTreatNaltrexone_Ai14_OPRK1_C1_M3_Top DownDLC_resnet50_BigBinTopSep17shuffle1_240000.csv',
    #              fps=60, legend_label='M3 Naltrexone', line_color='orange')
    # distance_det(file='Paper_Redo_5mg_kgU50PreTreatNaltrexone_Ai14_OPRK1_C1_M4_Top DownDLC_resnet50_BigBinTopSep17shuffle1_240000.csv',
    #              fps=60, legend_label='M4 Naltrexone', line_color='orangered')



    """U50 Data"""
    # distance_det(file='Paper_Redo_10_17_5mgkg_U50_Ai14_OPRK1_C2_F0_Top DownDLC_resnet50_BigBinTopSep17shuffle1_250000filtered.csv',
    #              fps=60, legend_label='F0 5mgkg U50', line_color='blue')
    # distance_det(file='Paper_Redo_10_17_5mgkg_U50_Ai14_OPRK1_C2_F1_Top DownDLC_resnet50_BigBinTopSep17shuffle1_250000filtered.csv',
    #              fps=60, legend_label='F1 5mgkg U50', line_color='lightblue')
    # distance_det(file='Paper_Redo_10_17_5mgkg_U50_Ai14_OPRK1_C2_F2_Top DownDLC_resnet50_BigBinTopSep17shuffle1_250000filtered.csv',
    #              fps=60, legend_label='F2 5mgkg U50', line_color='cyan')
    #
    #
    # distance_det(file='Paper_Redo_10_17_5mgkg_U50_Ai14_OPRK1_C1_M4_Top DownDLC_resnet50_BigBinTopSep17shuffle1_250000filtered.csv',
    #              fps=60, legend_label='M4 5mgkg U50',line_color='orchid')
    # distance_det(file='Paper_Redo_10_17_5mgkg_U50_Ai14_OPRK1_C1_M3_Top DownDLC_resnet50_BigBinTopSep17shuffle1_250000filtered.csv',
    #              fps=60, legend_label='M3 5mgkg U50', line_color='pink')
    # distance_det(file='Paper_Redo_10_16_5mgkg_U50_Ai14_OPRK1_C1_M2_Top DownDLC_resnet50_BigBinTopSep17shuffle1_250000filtered.csv',
    #              fps=60, legend_label='M2 5mgkg U50', line_color='plum')

    plt.show()
