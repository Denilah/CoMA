
import json
import pickle
import random
import joblib

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()


def main(lang_type,qid_topk):

    # {id: quetion_label,...}  lablel=1, how-to-do -it， label=2, no
    classfier = joblib.load('../staqc_tolabel/data/question_type_labeled_data/%s_query_label.pkl'%lang_type)
    print('#########标注模型加载完毕！##########')

    # [qid, query, features]
    qquery_feats = json.load(open('./corpus/%s_corpus_qid_query_to_features.json' % lang_type, "r"))
    qquery_feats = sorted(qquery_feats, key=lambda d: d[0], reverse=True)
    query_qids = [i[0] for i in qquery_feats]
    #  python  862710   sql 668544
    print('%s的query-feature语料共计%s条' % (lang_type, len(query_qids)))

    # (qid,value)
    qpost_values = json.load(open('./corpus/%s_corpus_qid_post_to_values.json' % lang_type, "r"))
    post_qids = [i[0] for i in qpost_values]
    # python 862704    sql 668544
    print('%s的post-feature语料共计%s条' % (lang_type, len(post_qids)))
    print('#########数据融合打分结束！##########')

    # 共有qid python 862704  sql 668544
    common_qids = set(query_qids) & set(post_qids)
    print('%s的query-post语料共计%s条' % (lang_type, len(common_qids)))

    # 过滤找不到的
    qpost_values = [i for i in qpost_values if i[0] in common_qids]
    # 从大到小排序
    qpost_values = sorted(qpost_values, key=lambda d: d[1], reverse=True)
    # 先取top-qids
    post_cutqids = sorted([i[0] for i in qpost_values[:qid_topk]])

    fquery_feats = [i for i in qquery_feats if i[0] in post_cutqids]

    filter_qids  =  [i[0] for i in fquery_feats]
    filter_feats =  [i[2] for i in fquery_feats]

    filter_feats =  scaler.fit_transform(filter_feats)
    filter_labels = classfier.predict(filter_feats)
    print('########查询标注类别结束！##########')

    get_howqids = [i for (i,j) in zip(filter_qids,filter_labels) if j==1]
    #  python 266443     sql 250729
    print('%s的过滤后对应剩余qid长度为%s'%(lang_type,len(get_howqids)))

    # 读取语料数据集
    with open('./corpus/%s_corpus_qid2index_blocks_unlabeled.pickle'%lang_type, "rb") as f:
        # [(id,index),[[si]，[si+1]] 文本块，[[c]] 代码，[q] 查询, 块长度]
        corpus_iid_unlabled = pickle.load(f)
        corpus_iid_unlabled = sorted(corpus_iid_unlabled, key=lambda x: (x[0][0], x[0][1]))
        # 根据qid-index排序 [(qid,index), query, code]
        corpus_iid_q2c_labled = [[i[0],i[3][0],i[2][0][0]] for i in corpus_iid_unlabled]
        # python 1595943   sql 1074407
        print('%s的语料的全部长度为%s'%(lang_type,len(corpus_iid_q2c_labled)))
        
    # 读取最终的标签
    with open("./corpus/%s_corpus_qid2index_blocks_labeled_final.txt" % lang_type, "r") as f:
        # 读取标签list
        iid_labels = [eval(i) for i in f.read().splitlines()]
        # 根据how to过滤
        filter_labels = [i for i in iid_labels if i[0] in get_howqids]
        # 新qindex的排序
        sorted_labels = sorted(filter_labels, key=lambda x: (x[0], x[1]))
        
    corpus_q2c_labled = [i for i in corpus_iid_q2c_labled if i[0] in sorted_labels]

    # 统计qid个数
    corpus_q2c_qids = set([i[0][0] for i in corpus_q2c_labled])
    corpus_q2c_iids = set([i[0] for i in  corpus_q2c_labled])
    # python 53673   sql 42897
    print('%s分类前的qid总长度为%d' % (lang_type,len(corpus_q2c_qids)))
    # python 80741   sql 69581
    print('%s分类前的qid-index总长度为%d' %(lang_type,len(corpus_q2c_iids)))

    # 统计qid个数
    corpus_iid_count =[(j, len([i for i in corpus_q2c_labled if i[0][0] == j])) for j in corpus_q2c_qids]

    ###########单候选的拿出来###########
    corpus_single_qids = set([i[0] for i in corpus_iid_count  if i[1] == 1])
    print('%s分类后的single-qid总长度为%d' % (lang_type, len(corpus_single_qids)))
    print('%s分类后的single-qid-index总长度为%d' % (lang_type,len(corpus_single_qids)))

    corpus_single_labled =  [[i[0][0], i[1], i[2]] for i in corpus_q2c_labled if i[0][0] in corpus_single_qids]
    # 新qid-index的排序
    corpus_single_labled = sorted(corpus_single_labled, key=lambda x: x[0])

    #############多候选的拿出来############
    corpus_mutiple_qids = set([i[0] for i in corpus_iid_count  if i[1] != 1])
    print('%s分类后的mutiple-qid总长度为%d' %(lang_type, len(corpus_mutiple_qids)))

    # qid-index 重新排序
    corpus_mutiple_labled = []
    # 循环遍历
    for qid in corpus_mutiple_qids:
        # 拿出指定qid 列表数据
        corpus_iid_q2c = [i for i in corpus_q2c_labled if i[0][0] == qid]
        # 根据qid-index排序 [(qid,index),query,code]
        corpus_iid_q2c = sorted(corpus_iid_q2c, key=lambda x: (x[0][0], x[0][1]))
        # 拿出qid-count计数
        count = [i[1] for i in corpus_iid_count if i[0]==qid][0]
        # 重新index排序
        for i,j in zip(range(count),corpus_iid_q2c):
            corpus_mutiple_labled.append([(qid,i),j[1],j[2]])

    print('%s分类后的mutiple-qid-index总长度为%d' % (lang_type, len(corpus_mutiple_labled)))

    #############################################单候选#############################
    corpus_single_qid_titles = dict([(i[0], i[1]) for i in corpus_single_labled])
    pickle.dump(corpus_single_qid_titles,open('./corpus/codepro/%s_single_qid_to_title.pickle' % lang_type, 'wb'))
    corpus_single_qid_codes  = dict([(i[0], i[2]) for i in corpus_single_labled])
    pickle.dump(corpus_single_qid_codes, open('./corpus/codepro/%s_single_qid_to_code.pickle' % lang_type, 'wb'))

    #############################################多候选#############################
    corpus_mutiple_qid_titles = dict([(i[0][0], i[1]) for i in corpus_mutiple_labled])
    pickle.dump(corpus_mutiple_qid_titles,open('./corpus/codepro/%s_multiple_qid_to_title.pickle' % lang_type, 'wb'))
    corpus_mutiple_iid_codes  = dict([(i[0], i[2]) for i in corpus_mutiple_labled])
    pickle.dump(corpus_mutiple_iid_codes,open('./corpus/codepro/%s_multiple_iid_to_code.pickle' % lang_type, 'wb'))

    data_to_json = []

    for id in (corpus_mutiple_qid_titles.keys()):
        # 存储字典
        data_dict = {}

        # 对应查询
        query = corpus_mutiple_qid_titles[id]
        # 对应item
        mutiple_cors = [i for i in list(corpus_mutiple_iid_codes.items()) if i[0][0] == id]
        # 代码排序
        sorted_codes = [i[1] for i in sorted(mutiple_cors,key=lambda x:x[0][1])]
        # 目标代码
        tagged_code = sorted_codes[0]

        data_dict["instruction"]= query
        data_dict["input"]=sorted_codes
        data_dict["output"]=tagged_code

        data_to_json.append(data_dict)
    # 预估长度
    print('%s多候选数据的长度为%s'%(lang_type,len(data_to_json)))

    # 保存为json
    with open("'./corpus/codepro/%s_High-Quality-Realistic-Query-Search-Code-Challenge-Data"%lang_type,"w",encoding='utf-8') as w:
        json.dump(data_to_json,w)


# 保证以CSV保存字符串
def filter_char(line):
    # 换小写
    line = line.lower()
    # 去除\r
    line = line.replace('\r', ' ')
    return line


def merge_lang():

    with open("'./corpus/codepro/python_High-Quality-Realistic-Query-Search-Code-Challenge-Data","r",encoding='utf-8') as w:
        python_json =json.load(w)

    with open("'./corpus/codepro/sql_High-Quality-Realistic-Query-Search-Code-Challenge-Data","r",encoding='utf-8') as w:
        sql_json = json.load(w)

    json_data_list = python_json + sql_json

    random.shuffle(json_data_list)

    data_to_json = []

    for i in json_data_list:
        # 存储字典
        data_dict = {}

        data_dict["instruction"] = filter_char(i["instruction"])
        data_dict["input"] = [filter_char(j) for j in i["input"]]
        data_dict["output"] = filter_char(i["output"])

        data_to_json.append(data_dict)

    with open("'./corpus/codepro/High-Quality-Realistic-Query-Search-Code-Challenge-Data","w",encoding='utf-8') as w:
        json.dump(data_to_json,w,indent=4)


python_type = 'python'
sql_type = 'sql'

qid_topk = 400000

if __name__ == '__main__':
    #main(python_type,qid_topk)
    main(sql_type,qid_topk)
