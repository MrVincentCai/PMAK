import numpy as np
import math
import pandas as pd

# Configuration for AAC/DPC
CharMap = {'G': 0, 'S': 1, 'A': 2, 'T': 3, 'V': 4, 'I': 5, 'L': 6, 'Y': 7, 'F': 8, 'H': 9, 'P': 10,
           'D': 11, 'M': 12, 'E': 13, 'W': 14, 'K': 15, 'C': 16, 'R': 17, 'N': 18, 'Q': 19,
           'X': 20, 'B': 20, 'Z': 20, 'J': 20, 'U': 20, 'O': 20
           }

# Configuration for CTriad
CTriad = {'A': 0, 'G': 0, 'V': 0, 'I': 1, 'L': 1, 'F': 1, 'P': 1,
          'Y': 2, 'M': 2, 'T': 2, 'S': 2, 'H': 3, 'N': 3, 'Q': 3, 'W': 3,
          'R': 4, 'K': 4, 'D': 5, 'E': 5, 'C': 6,
          'X': 7, 'B': 7, 'Z': 7, 'J': 7, 'U': 7, 'O': 7
          }

# Configuration for GAAC/GDPC/GTPC
GCMap = {'T': 0, 'V': 0, 'L': 0, 'I': 0, 'M': 0, 'G': 0, 'A': 0,
         'S': 0, 'C': 0, 'D': 1, 'N': 1, 'E': 1, 'Q': 1, 'K': 2,
         'R': 2, 'H': 2, 'F': 3, 'Y': 3, 'W': 3, 'P': 4,
         'X': 5, 'B': 5, 'Z': 5, 'J': 5, 'U': 5, 'O': 5
         }

# Configuration for CTD
CTDMap = [['RKEDQN', 'GASTPHY', 'CLVIMFW'], ['QSTNGDE', 'RAHCKMV', 'LYPFIW'], ['QNGSWTDERA', 'HMCKV', 'LPFYI'],
          ['KPDESNQT', 'GRHA', 'YMFWLCVI'], ['KDEQPSRNTG', 'AHYMLV', 'FIWC'], ['RDKENQHYP', 'SGTAW', 'CVLIMF'],
          ['KERSQD', 'NTPG', 'AYHWVMFLIC'], ['GASCTPD', 'NVEQIL', 'MHKFRYW'], ['LIFWCMVY', 'PATGS', 'HQRKNED'],
          ['GASDT', 'CPNVEQIL', 'KMHFRYW'], ['KR', 'ANCQGHILMFPSTWYV', 'DE'], ['EALMQKRH', 'VIYCWFT', 'GNPSD'],
          ['ALFCGIVW', 'PKQEND', 'MPSTHY']
          ]

MoranMap = {'A': 0, 'R': 1, 'N': 2, 'D': 3, 'C': 4, 'Q': 5, 'E': 6, 'G': 7, 'H': 8, 'I': 9,
            'L': 10, 'K': 11, 'M': 12, 'F': 13, 'P': 14, 'S': 15, 'T': 16, 'W': 17, 'Y': 18, 'V': 19,
            'X': 20, 'B': 20, 'Z': 20, 'J': 20, 'U': 20, 'O': 20
            }

SOCN_Map = {'A': 0, 'C': 1, 'D': 2, 'E': 3, 'F': 4, 'G': 5, 'H': 6, 'I': 7, 'K': 8, 'L': 9,
            'M': 10, 'N': 11, 'P': 12, 'Q': 13, 'R': 14, 'S': 15, 'T': 16, 'V': 17, 'W': 18, 'Y': 19,
            'X': 20, 'B': 20, 'Z': 20, 'J': 20, 'U': 20, 'O': 20
            }

SOCN_2_Map = {'S': 0, 'R': 1, 'L': 2, 'P': 3, 'T': 4, 'A': 5, 'V': 6, 'G': 7, 'I': 8, 'F': 9,
              'Y': 10, 'C': 11, 'H': 12, 'Q': 13, 'N': 14, 'K': 15, 'D': 16, 'E': 17, 'M': 18, 'W': 19,
              'X': 20, 'B': 20, 'Z': 20, 'J': 20, 'U': 20, 'O': 20
              }


def AAC_feature(Sequence):
    '''
    :param Sequence: Sequence
    :return: aac: 20D
    '''
    # Count all amino acids composition
    aac = np.zeros([len(Sequence), len(CharMap)], dtype=float)
    for i in range(len(Sequence)):
        for j in range(len(Sequence[i])):
            aac[i][CharMap[Sequence[i][j]]] += 1
    # Standardization
    for i in range(len(Sequence)):
        for j in range(len(CharMap)):
            aac[i][j] /= len(Sequence[i])
    return aac


def DPC_feature(Sequence):
    '''
    :param Sequence: Sequence
    :return: dpc: 400D
    '''
    # Count all Di-amino acids composition
    dpc = np.zeros([len(Sequence), len(CharMap)*len(CharMap)], dtype=float)
    for i in range(len(Sequence)):
        for j in range(len(Sequence[i])-1):
            index_l = CharMap[Sequence[i][j]]
            index_r = CharMap[Sequence[i][j+1]]
            dpc[i][index_l*21+index_r] += 1
    # Standardization
    for i in range(len(Sequence)):
        for j in range(len(CharMap)*len(CharMap)):
            dpc[i][j] /= (len(Sequence[i])-1)
    return dpc


def CTriad_feature(Sequence):
    '''
    :param Sequence: Sequence
    :return: ctr: 343D
    '''
    # Count all Ctriad amino acids composition
    ctr = np.zeros([len(Sequence), 8*8*8], dtype=float)
    for i in range(len(Sequence)):
        for j in range(len(Sequence[i])-2):
            index_f = CTriad[Sequence[i][j]]
            index_s = CTriad[Sequence[i][j + 1]]
            index_t = CTriad[Sequence[i][j + 2]]
            # ctr[i][index_f * 7 * 7 + index_s * 7 + index_t] += 1
            ctr[i][index_f * 8 * 8 + index_s * 8 + index_t] += 1
    # Standardization
    for i in range(len(Sequence)):
        for j in range(8*8*8):
            ctr[i][j] /= (len(Sequence[i])-2)
    return ctr


def GAAC_feature(Sequence):
    '''
    :param Sequence: Sequence
    :return: gaac: 5D
    '''
    # Count all grouped amino acid composition
    gaac = np.zeros([len(Sequence), 6], dtype=float)
    for i in range(len(Sequence)):
        for j in range(len(Sequence[i])):
            gaac[i][GCMap[Sequence[i][j]]] += 1
    # Standardization
    for i in range(len(Sequence)):
        for j in range(6):
            gaac[i][j] /= len(Sequence[i])
    return gaac


def GDPC_feature(Sequence):
    '''
    :param Sequence:
    :return: gaac:25D
    '''
    # Count all grouped Di-amino acids composition
    gdpc = np.zeros([len(Sequence), 6*6], dtype=float)
    for i in range(len(Sequence)):
        for j in range(len(Sequence[i])-1):
            index_l = GCMap[Sequence[i][j]]
            index_r = GCMap[Sequence[i][j + 1]]
            gdpc[i][index_l * 6 + index_r] += 1
    # Standardization
    for i in range(len(Sequence)):
        for j in range(6*6):
            gdpc[i][j] /= (len(Sequence[i])-1)
    return gdpc


def GTPC_feature(Sequence):
    '''
    :param Sequence:
    :return: gtpc:125D
    '''
    # Count all Tri-amino acids composition
    gtpc = np.zeros([len(Sequence), 6*6*6], dtype=float)
    for i in range(len(Sequence)):
        for j in range(len(Sequence[i])-2):
            index_f = GCMap[Sequence[i][j]]
            index_s = GCMap[Sequence[i][j + 1]]
            index_t = GCMap[Sequence[i][j + 2]]
            gtpc[i][index_f * 6 * 6 + index_s * 6 + index_t] += 1
    # Standardization
    for i in range(len(Sequence)):
        for j in range(6*6*6):
            gtpc[i][j] /= (len(Sequence[i])-2)
    return gtpc


def CTD_C_feature(Sequence):
    '''
    :param Sequence: Sequence
    :return: ctd_c:39D
    '''
    # Count all different composition
    ctd_c = np.zeros([len(Sequence), len(CTDMap)*len(CTDMap[0])], dtype=float)
    for i in range(len(Sequence)):
        for j in range(len(Sequence[i])):
            for k in range(len(CTDMap)):
                for l in range(len(CTDMap[0])):
                    if Sequence[i][j] in CTDMap[k][l]:
                        ctd_c[i][k*3+l] += 1
    # Standardization
    for i in range(len(Sequence)):
        for j in range(len(CTDMap)*len(CTDMap[0])):
            ctd_c[i][j] /= len(Sequence[i])
    return ctd_c


def CTD_T_feature(Sequence):
    '''
    :param Sequence:
    :return: ctd_t:39D
    '''
    # Count all different transition
    ctd_t = np.zeros([len(Sequence), len(CTDMap) * len(CTDMap[0])], dtype=float)
    for i in range(len(Sequence)):
        for j in range(len(Sequence[i])-1):
            for k in range(len(CTDMap)):
                if (Sequence[i][j] in CTDMap[k][0] and Sequence[i][j+1] in CTDMap[k][1]) or \
                   (Sequence[i][j] in CTDMap[k][1] and Sequence[i][j+1] in CTDMap[k][0]):
                    ctd_t[i][k * 3 + 0] += 1
                elif (Sequence[i][j] in CTDMap[k][1] and Sequence[i][j+1] in CTDMap[k][2]) or \
                     (Sequence[i][j] in CTDMap[k][2] and Sequence[i][j+1] in CTDMap[k][1]):
                    ctd_t[i][k * 3 + 1] += 1
                elif (Sequence[i][j] in CTDMap[k][0] and Sequence[i][j+1] in CTDMap[k][2]) or \
                     (Sequence[i][j] in CTDMap[k][2] and Sequence[i][j+1] in CTDMap[k][0]):
                    ctd_t[i][k * 3 + 2] += 1
                else:
                    pass
    # Standardization
    for i in range(len(Sequence)):
        for j in range(len(CTDMap) * len(CTDMap[0])):
            ctd_t[i][j] /= (len(Sequence[i])-1)
    return ctd_t


def CTD_D_feature(Sequence):
    '''
    :param Sequence:
    :return: ctd_d:195D
    '''
    # Count all different distribution index
    ctd_c = CTD_C_feature(Sequence)
    ctd_tag = np.zeros([len(Sequence), len(CTDMap) * len(CTDMap[0]), 5], dtype=float)
    ctd_d = np.zeros([len(Sequence), len(CTDMap) * len(CTDMap[0]), 5], dtype=float)
    for i in range(len(Sequence)):
        for j in range(len(CTDMap) * len(CTDMap[0])):
            ctd_tag[i][j][0] = int(ctd_c[i][j]*0.0)
            ctd_tag[i][j][1] = int(ctd_c[i][j]*0.25)
            ctd_tag[i][j][2] = int(ctd_c[i][j]*0.5)
            ctd_tag[i][j][3] = int(ctd_c[i][j]*0.75)
            ctd_tag[i][j][4] = int(ctd_c[i][j]*1.0)-1
    # Count all different distribution
    for k in range(len(CTDMap)):
        for l in range(len(CTDMap[0])):
            for i in range(len(Sequence)):
                tag = 0
                for j in range(len(Sequence[i])):
                    if Sequence[i][j] in CTDMap[k][l]:
                        if tag == ctd_tag[i][k*3+l][0]:
                            ctd_d[i][k * 3 + l][0] = j
                        if tag == ctd_tag[i][k*3+l][1]:
                            ctd_d[i][k * 3 + l][1] = j
                        if tag == ctd_tag[i][k*3+l][2]:
                            ctd_d[i][k * 3 + l][2] = j
                        if tag == ctd_tag[i][k*3+l][3]:
                            ctd_d[i][k * 3 + l][3] = j
                        if tag == ctd_tag[i][k*3+l][4]:
                            ctd_d[i][k * 3 + l][4] = j
                        tag += 1
    ctd_d = ctd_d.reshape([len(Sequence), -1])
    # Standardization
    for i in range(len(Sequence)):
        for j in range(len(CTDMap) * len(CTDMap[0]) * 5):
            ctd_d[i][j] /= len(Sequence[i])
    return ctd_d


def Get_AAindex():
    '''
    :return: Get different amino acis property
    '''
    # Load AAindex data
    AAindex = []
    with open('ConfigurationFile/AAindex_16.txt', 'r') as myfile:
        for line in myfile:
            if line[0] != 'H':
                Str_num = ''
                i = 0
                while line[i] != '\n':
                    if line[i] == ' ':
                        AAindex.append(float(Str_num))
                        Str_num = ''
                        i += 1
                    Str_num += line[i]
                    i += 1
                AAindex.append(float(Str_num))
    AAindex = np.array(AAindex).reshape([16, 20])

    default_value = np.zeros(16)  # or you could use np.mean(AAindex, axis=1)
    AAindex = np.column_stack([AAindex, default_value])

    # Standardization
    for i in range(len(AAindex)):
        Amean = np.mean(AAindex[i])
        Adev = np.std(AAindex[i])
        for j in range(len(AAindex[i])):
            AAindex[i][j] = (AAindex[i][j] - Amean) / Adev
    return AAindex


def Get_mean(Sequence, AAindex):
    '''
    :param Sequence: Sequence
    :param AAindex: Certain amino acid property index
    :return: Mean property
    '''
    P1 = 0
    for i in range(len(Sequence)):
        P1 += AAindex[MoranMap[Sequence[i]]]
    P1 /= len(Sequence)
    return P1


def Moran_coor(Sequence, min_length):
    '''
    :param Sequence: Sequence
    :return: moran: nlag*16D
    '''
    # Calculate moran correlation
    AAindex = Get_AAindex()
    nlag = min_length-1
    moran = np.zeros([len(Sequence), len(AAindex), nlag])
    for i in range(len(Sequence)):
        for j in range(len(AAindex)):
            P1 = Get_mean(Sequence[i], AAindex[j])
            for k in range(1, nlag+1):
                Id = 0
                for l in range(len(Sequence[i])-k):
                    Id += (AAindex[j][MoranMap[Sequence[i][l]]]-P1)*(AAindex[j][MoranMap[Sequence[i][l+k]]]-P1)
                Id /= (len(Sequence[i])-nlag)
                dishu = 0
                for l in range(len(Sequence[i])):
                    dishu += math.pow((AAindex[j][MoranMap[Sequence[i][l]]]-P1), 2)
                dishu /= (len(Sequence[i]))
                if dishu == 0:
                    Id = 0
                else:
                    Id /= dishu
                moran[i][j][k-1] = Id
    moran = moran.reshape([len(Sequence), -1])
    return moran


def SOCN_feature(Sequence, min_length):
    '''
    :param Sequence: Sequence
    :param min_length: Minimum sequence length
    :return: min_length-1
    '''
    dis = np.array(pd.read_excel('ConfigurationFile/SOCN.xlsx', sheet_name='main', engine='openpyxl'))

    # Add an extra row and column for rare amino acids (e.g., "X")
    default_row = np.zeros(dis.shape[1])  # Default values for rare amino acids (row)
    dis = np.vstack([dis, default_row])  # Add new row

    default_col = np.zeros(dis.shape[0])  # Default values for rare amino acids (column)
    dis = np.column_stack([dis, default_col])  # Add new column

    nlag = min_length-1
    socn = np.zeros([len(Sequence), nlag])
    # Calculate all socn
    for i in range(len(Sequence)):
        for j in range(1, nlag+1):
            for k in range(len(Sequence[i])-j):
                socn[i][j-1] += dis[SOCN_Map[Sequence[i][k]]][SOCN_Map[Sequence[i][k+j]]]
    # Standardization
    for i in range(len(Sequence)):
        for j in range(1, nlag+1):
            socn[i][j-1] /= (len(Sequence[i])-j)
    return socn


def SOCN_2_feature(Sequence, min_length):
    '''
    :param Sequence: Sequence
    :param min_length: Minimum sequence length
    :return: min_length-1
    '''
    dis = np.array(pd.read_excel('ConfigurationFile/SOCN_2.xlsx', sheet_name='main', engine='openpyxl'))
    dis = dis.astype(float)

    # Add an extra row and column for rare amino acids (e.g., "X")
    default_row = np.zeros(dis.shape[1])  # Default values for rare amino acids (row)
    dis = np.vstack([dis, default_row])  # Add new row

    default_col = np.zeros(dis.shape[0])  # Default values for rare amino acids (column)
    dis = np.column_stack([dis, default_col])  # Add new column

    # Standardization
    for i in range(len(dis)):
        max_v = max(dis[i])
        min_v = min(dis[i])
        if max_v != min_v:  # 确保不为0 避免特征中出现nan值
            for j in range(len(dis[i])):
                dis[i][j] = (dis[i][j] - min_v) / (max_v - min_v)
        else:
            dis[i] = np.zeros_like(dis[i])
    nlag = min_length-1
    socn_2 = np.zeros([len(Sequence), nlag])
    # Calculate distance
    for i in range(len(Sequence)):
        for j in range(1, nlag+1):
            for k in range(len(Sequence[i])-j):
                socn_2[i][j-1] += dis[SOCN_2_Map[Sequence[i][k]]][SOCN_2_Map[Sequence[i][k+j]]]
    # Standardization
    for i in range(len(Sequence)):
        for j in range(1, nlag + 1):
            socn_2[i][j - 1] /= (len(Sequence[i]) - j)
    return socn_2


def Shannon_entropy(Sequence):
    '''
    :param Sequence: Sequence
    :return: 1
    '''
    aac = AAC_feature(Sequence)
    # Calculate shannon entropy
    Sh = np.zeros([len(Sequence), 1], dtype=float)
    for i in range(len(aac)):
        for j in range(len(aac[i])):
            if aac[i][j] != 0:
                Sh[i][0] += -(aac[i][j]*math.log(aac[i][j], 2))
    # Standardization
    max_v = max(Sh.reshape([len(Sequence)]))
    min_v = min(Sh.reshape([len(Sequence)]))
    for i in range(len(Sh)):
        Sh[i][0] = (Sh[i][0]-min_v) / (max_v-min_v)
    return Sh


def Get_features(Sequence, min_length):
    '''
    :param Sequence: Input sequences
    :return: Output sequence features
    '''
    # Get sequence minimum length
    # min_length = 999
    # for i in range(len(Sequence)):
    #     min_length = min(min_length, len(Sequence[i]))
    print('Minimum sequence length: ', min_length)
    # Determine whether to generate CTriad/GTPC feature
    features_3 = 0
    features_6 = 0
    if min_length > 2:
        features_3 = CTriad_feature(Sequence)
        features_6 = GTPC_feature(Sequence)
    features_1 = AAC_feature(Sequence)
    features_2 = DPC_feature(Sequence)
    features_4 = GAAC_feature(Sequence)
    features_5 = GDPC_feature(Sequence)
    features_7 = CTD_C_feature(Sequence)
    features_8 = Shannon_entropy(Sequence)
    features_9 = CTD_T_feature(Sequence)
    features_10 = CTD_D_feature(Sequence)
    features_11 = Moran_coor(Sequence, min_length)
    features_12 = SOCN_feature(Sequence, min_length)
    features_13 = SOCN_2_feature(Sequence, min_length)
    if min_length < 3:
        features = np.concatenate((features_1, features_2, features_4, features_5,
                                   features_7, features_8, features_9, features_10,
                                   features_11, features_12, features_13), axis=1)
    else:
        features = np.concatenate((features_1, features_2, features_3, features_4, features_5,
                                   features_6, features_7, features_8, features_9, features_10,
                                   features_11, features_12, features_13), axis=1)
    return features
