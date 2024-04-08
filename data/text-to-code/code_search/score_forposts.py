

import pywt
import random
import math
import pickle
import json
import numpy as np
import pandas as pd
from nltk import FreqDist

#导入EM算法估计权重
import sys
sys.path.append("..")
from wtff_score.em_gmm import estimate

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()


##############################################选用SN-WTFF-HW框架打分##########################3
# 全体属性小波变换
def allso_signl_dwt(signal, wt_type):   # 4层小波变换 L=4
    LF1, HF1 = pywt.dwt(signal, wt_type)  # LF01:低频分量         HF01:高频分量
    LF2, HF2 = pywt.dwt(LF1, wt_type)     # LF02:低频分量         HF02:高频分量
    LF3, HF3 = pywt.dwt(LF2, wt_type)     # LF03:低频分量         HF03:高频分量
    LF4, HF4 = pywt.dwt(LF3, wt_type)     # LF04:低频分量         HF04:高频分量
    return HF1, HF2, HF3, HF4, LF4


# 全体属性小波的系数向量集
def allso_em_fuse(C1, C2, C3, C4, C5):
    # 估计线性融合时的权重
    w1, w2, w3, w4, w5 = estimate(C1.tolist()), estimate(C2.tolist()), estimate(C3.tolist()), estimate(
        C4.tolist()), estimate(C5.tolist())
    # K维特征融合后的系数集
    new_HF1 = [w1[i] * C1[:, i] for i in range(C1.shape[1])][0]
    new_HF2 = [w2[i] * C2[:, i] for i in range(C2.shape[1])][0]
    new_HF3 = [w3[i] * C3[:, i] for i in range(C3.shape[1])][0]
    new_HF4 = [w4[i] * C4[:, i] for i in range(C4.shape[1])][0]
    new_LF4 = [w5[i] * C5[:, i] for i in range(C5.shape[1])][0]
    return new_HF1, new_HF2, new_HF3, new_HF4, new_LF4  # 作为新的低频分量


# 全体属性小波逆变换
def allso_signl_idwt(new_HF1, new_HF2, new_HF3, new_HF4, new_LF4, wt_type):  # 4层小波逆变换 L=4
    new_LF3 = pywt.idwt(new_LF4, new_HF4, wt_type)  # (低频分量,高频分量)---→上一级(低频分量)
    new_LF2 = pywt.idwt(new_LF3, new_HF3, wt_type)
    new_LF1 = pywt.idwt(new_LF2, new_HF2, wt_type)
    new_signal = pywt.idwt(new_LF1, new_HF1, wt_type)
    return new_signal


# 全体融合打分
def allso_dwt_score(all_feat, wt_type):

    signl0, signl1, signl2, signl3, signl4, signl5, signl6 = all_feat[:,0], all_feat[:,1], all_feat[:,2], all_feat[:,3], all_feat[:,4], all_feat[:,5], all_feat[:,6]
    # 小波变换
    HF01, HF02, HF03, HF04, LF04 = allso_signl_dwt(signl0, wt_type)  # 第一棵树
    HF11, HF12, HF13, HF14, LF14 = allso_signl_dwt(signl1, wt_type)  # 第二棵树
    HF21, HF22, HF23, HF24, LF24 = allso_signl_dwt(signl2, wt_type)  # 第三棵树
    HF31, HF32, HF33, HF34, LF34 = allso_signl_dwt(signl3, wt_type)  # 第四棵树
    HF41, HF42, HF43, HF44, LF44 = allso_signl_dwt(signl4, wt_type)  # 第五棵树
    HF51, HF52, HF53, HF54, LF54 = allso_signl_dwt(signl5, wt_type)  # 第六棵树
    HF61, HF62, HF63, HF64, LF64 = allso_signl_dwt(signl6, wt_type)  # 第七棵树

    # 构建系数矩阵
    C1 = np.concatenate((HF01.reshape(-1, 1), HF11.reshape(-1, 1), HF21.reshape(-1, 1), HF31.reshape(-1, 1),
                         HF41.reshape(-1, 1), HF51.reshape(-1, 1), HF61.reshape(-1, 1)), axis=1)
    C2 = np.concatenate((HF02.reshape(-1, 1), HF12.reshape(-1, 1), HF22.reshape(-1, 1), HF32.reshape(-1, 1),
                         HF42.reshape(-1, 1), HF52.reshape(-1, 1), HF62.reshape(-1, 1)), axis=1)
    C3 = np.concatenate((HF03.reshape(-1, 1), HF13.reshape(-1, 1), HF23.reshape(-1, 1), HF33.reshape(-1, 1),
                         HF43.reshape(-1, 1), HF53.reshape(-1, 1), HF63.reshape(-1, 1)), axis=1)
    C4 = np.concatenate((HF04.reshape(-1, 1), HF14.reshape(-1, 1), HF24.reshape(-1, 1), HF34.reshape(-1, 1),
                         HF44.reshape(-1, 1), HF54.reshape(-1, 1), HF64.reshape(-1, 1)), axis=1)
    C5 = np.concatenate((LF04.reshape(-1, 1), LF14.reshape(-1, 1), LF24.reshape(-1, 1), LF34.reshape(-1, 1),
                         LF44.reshape(-1, 1), LF54.reshape(-1, 1), LF64.reshape(-1, 1)), axis=1)
    # 每个系数矩阵的维度不同
    new_HF1, new_HF2, new_HF3, new_HF4, new_LF4 = allso_em_fuse(C1, C2, C3, C4, C5)
    new_signal = allso_signl_idwt(new_HF1, new_HF2, new_HF3, new_HF4, new_LF4, wt_type)

    return new_signal  # 全体属性的综合打分


def main(lang_type):

    post_terms = pickle.load(open('./corpus/%s_corpus_qid_allterms.pickle' % lang_type, "rb"))

    # 所有选项 [(qid, aid), qhtmlstr, ahtmlstr, soical_terms]
    post_qids = set([i[0][0] for i in post_terms])

    # 社交数据 (qid,soical_terms)
    qid_terms = [(i[0][0],i[4]) for i in post_terms]

    # 社交数据 (qid,[qview,qanswer,qcomment,qfavorite,qscorenum,ccomment,cscorenum])
    qidnum_lis  = [i[0] for i in qid_terms]

    qview_lis     = [i[1][0] for i in qid_terms]
    qanswer_lis   = [i[1][1] for i in qid_terms]
    qcomment_lis  = [i[1][2] for i in qid_terms]
    qfavorite_lis = [i[1][3] for i in qid_terms]
    qscore_lis    = [i[1][4] for i in qid_terms]
    ccomment_lis  = [i[1][5] for i in qid_terms]
    cscore_lis    = [i[1][6] for i in qid_terms]

    # 数据字典
    data_dict = {'qid': qidnum_lis,'qview': qview_lis, 'qanswer': qanswer_lis, 'qcomment': qcomment_lis,'qfavorite': qfavorite_lis,'qscore': qscore_lis, 'ccomment': ccomment_lis, 'cscore': cscore_lis}

    # 索引列表
    column_list = ['qid', 'qview', 'qanswer', 'qcomment', 'qfavorite', 'qscore', 'ccomment','cscore']
    # 数据保存
    attr_data = pd.DataFrame(data_dict, columns=column_list)

    # 缺省值补全
    attr_data[attr_data == -10000] = np.nan  # 将缺省值置换nan  数据爬虫不存在
    attr_data[attr_data == 0] = np.nan  # 将缺省值置换nan  帖子本身没给值

    # 每列去除噪声点,不具备分布的特征需要去除噪点
    remove_keys = ['qview', 'qscore', 'cscore']  # 类别点
    for key in remove_keys:
        key_freq = FreqDist([i for i in attr_data[key].tolist() if not math.isnan(i)])
        freq_num = [i[0] for i in key_freq.most_common()]  # 按频率出现从大到小排序
        key_noise = freq_num[-int(0.01 * len(freq_num)):]  # 出现最小%1比例作为噪声点,分析数据
        key_data = [i if i not in key_noise else np.nan for i in attr_data[key].tolist()]
        key_Serie = pd.Series(key_data, index=attr_data.index)
        attr_data[key] = key_Serie

    # 每列NaN填充平均数
    column_keys = ['qview', 'qanswer', 'qcomment', 'qfavorite', 'qscore', 'ccomment', 'cscore']

    # 列属性依次循环
    for key in column_keys:
        #  填充数据
        attr_data[key] = attr_data[key].fillna(round(attr_data[key].mean()))

    # 数据归一化操作
    data_mn = attr_data.reindex(columns=column_keys).apply(lambda x: (x - x.min()) / (x.max() - x.min()), 0)
    # 相应的值替换
    attr_data[column_keys] = data_mn
    # ---------------------------------------------数据预处理------------------------------------

    # ---------------------------------------------数据需截断------------------------------------
    # 数据长度保证是2^x样式才能分解
    attr_length = len(attr_data)
    print('数据最开始爬取的原始长度是%d' %attr_length)
    # 开始循环查找，数据长度
    lang_lencut = math.floor(attr_length / (2**4)) * (2**4)
    print('数据用小波变换的截断长度是%d' % lang_lencut)
    '''
    多选      sql      python 
        开始  687254    862710
        截断  687248    862704
        变换            862704
    '''
    # ---------------------------------------------数据需截断------------------------------------
    # 被剪切的长度
    remove_lencut = len(attr_data) - lang_lencut
    print('数据用小波变换需截取的长度是%d'% remove_lencut)
    # 去除的qid 部分
    remove_qids = random.sample(post_qids, remove_lencut)

    # 把qid作为索引
    attr_data = attr_data.set_index('qid')

    # 保证4层可以分解
    attr_data = attr_data.drop(index=remove_qids)
    print('数据用小波变换保存的长度为%d'%len(attr_data))

    #索引qid添加位列
    attr_data['qid'] = attr_data.index

    attr_feat = attr_data.reindex(columns=column_keys).values

    # 特征融合打分
    fuse_score = allso_dwt_score(attr_feat, 'haar')

    # 融合打分赋值
    attr_data.insert(attr_data.shape[1], 'score', fuse_score)

    # 打分值排序，从大到小排序
    sort_attr = attr_data.sort_values(by='score', ascending=False)
    # 重建0-K索引值
    sort_attr = sort_attr.reset_index(drop=True)
    # 抽取前项索引值
    sort_attr = sort_attr.reindex(columns=['qid','score'])

    post_values = [(i,j) for i,j in zip(sort_attr['qid'], sort_attr['score'])]

    # 保存帖子得分
    with open('./corpus/%s_corpus_qid_post_to_values.json' % lang_type, "w") as f:
        json.dump(post_values,f)



python_type = 'python'
sql_type = 'sql'

if __name__ == '__main__':
    main(python_type)
    main(sql_type)