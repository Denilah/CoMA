# coding=utf-8
import os
import re
import pickle

#网页转义
import html
#解析数据
from bs4 import BeautifulSoup

import sys
sys.path.append("../crawl_process/")

# 解析结构
from parsers import python_structured
from parsers import javang_structured
from parsers import sqlang_structured


def contain_non_English(line):
    return bool(re.search(r'[^\x00-\xff]',line))

def remove_unprint_chars(line):
    line= line.replace('\n','\\n')
    """移除所有不可见字符  保留\n"""
    line = ''.join(x for x in line if x.isprintable())
    line = line.replace('\\n', '\n')
    return line

#--------------------------linux代码，配置可执行-----------------------
def extract_xml(data_folder,posts_xml,python_xml,sqlang_xml,answer_xml):

    # grep 正则表达式  #提取xml中python标签
    cmd_python = 'egrep "python" %s > %s' % (data_folder + posts_xml, data_folder + python_xml)
    print(cmd_python)
    os.popen(cmd_python).read()

    # grep  正则表达式  #提取xml中sql标签
    cmd_sqlang = 'egrep "sql" %s > %s' % (data_folder + posts_xml, data_folder + sqlang_xml)
    print(cmd_sqlang)
    os.popen(cmd_sqlang).read()

    # grep 正则表达式  #所有回答xml数据
    cmd_answer = 'egrep \'PostTypeId="2"\' %s > %s'% (data_folder + posts_xml, data_folder + answer_xml)
    print(cmd_answer)
    os.popen(cmd_answer).read()

    print('XML数据抽取完毕！！')


# 解析属性标签
def get_tag_param(rowTag, tag):
    try:
        output = rowTag[tag]
        return output
    except:
        return '-10000'

#替换标签
def filter_other_tag(htmlSoup):
    nhtmlStr = str(htmlSoup)
    precodeTags = htmlSoup.find_all('pre')
    precodeTags = [str(i) for i in precodeTags]
    nprecodeTags = []
    for code in precodeTags:
        for j in ['<h2>', '<h3>', '<p>', '<ul>', '<li>']:
            if j in code:
                code = code.replace(j, '').replace(j.replace('<', '</'), '')
        nprecodeTags.append(code)
    for i, j in zip(precodeTags, nprecodeTags):
        nhtmlStr = nhtmlStr.replace(i, j)
    return nhtmlStr


# 匹配块
def match_str_block(p, block):
    matchstr = re.finditer(p, block)
    bocklist = [i.group() for i in matchstr]
    return bocklist


# 匹配中间块
def match_str_ptag(p_nl, ptagStr):
    ptaglist = match_str_block(p_nl, ptagStr)
    ptagtext = parse_tag_text(ptaglist[-1]) if len(ptaglist) > 1 else parse_tag_text(ptaglist[0])
    return ptagtext


# 匹配最后块
def match_str_btag(p_nl, ptagStr):
    ptaglist = match_str_block(p_nl, ptagStr)
    ptagtext = parse_tag_text(ptaglist[0])
    return ptagtext


# 取p标签文本
def parse_tag_text(bodyTag):
    bodysoup = BeautifulSoup(bodyTag, 'lxml')
    bodytext = bodysoup.findAll(text=True)
    bodystr = ' '.join(bodytext)
    return bodystr


def match_tag_code(p_hcode, p_pcode, p_pcode1, p_pcode2, ctagStr):
    # h标签内容
    hcodestr = p_hcode.findall(ctagStr)[0] if p_hcode.findall(ctagStr) != [] else ''
    # 找到pre/pre class
    pcodetag = match_str_block(p_pcode, ctagStr)[0]
    # pre标签内容 或 pre class内容
    pcodestr = p_pcode1.findall(pcodetag)[0] if p_pcode1.findall(pcodetag) != [] else p_pcode2.findall(pcodetag)[0]
    # 两种标签合并
    codestr = ' '.join([hcodestr, pcodestr])
    return codestr


def parse_tag_block(htmlSoup):

    nhtmlStr = filter_other_tag(htmlSoup)

    #  识别块标签
    p_block = re.compile(r'(<p>((?!<pre>|<pre class=[\s\S]*>|<h[2|3]>)[\s\S])*</p>)?\n*'
                         r'(<h[2|3]>((?!<p>|<pre>|<pre class=[\s\S]*>|<h[2|3]>|<ul>|<li>)[\s\S])*</h[2|3]>)?\n*'
                         r'<pre((?!<p>|<pre>|<pre class=[\s\S]*>|<h[2|3]>)[\s\S])*><code>((?!<p>|<pre>|<pre class=[\s\S]*>|<h[2|3]>)[\s\S])*</code></pre>\n*'
                         r'((?!<p>|<pre>|<pre class=[\s\S]*>|<h[2|3]>)[\s\S])*\n*'
                         r'(<p>((?!<pre>|<pre class=[\s\S]*>|<h[2|3]>)[\s\S])*</p>)?')

    all_blocklis = match_str_block(p_block, nhtmlStr)

    # 上文模块
    p_forward = re.compile(r'(<p>[\s\S]*</p>)?\n*[\s\S]*(<pre>|<pre class=[\s\S]*>)<code>')
    # 中间内容
    p_context = re.compile(r'<p>((?!<p>)[\s\S])*</p>')
    # 下文模块
    p_backward = re.compile(r'</code></pre>\n*[\s\S]*(<p>[\s\S]*</p>)?')

    # 代码模块
    p_code = re.compile(r'(<h[2|3]>[\s\S]*</h[2|3]>)?\n*(<pre>|<pre class=[\s\S]*>)<code>[\s\S]*</code></pre>')
    # h2/3标签
    p_hcode = re.compile(r'h[2|3]>([\s\S]*)</h[2|3]>')
    # pre两种
    p_pcode = re.compile(r'(<pre>|<pre class=[\s\S]*>)<code>[\s\S]*</code></pre>')

    p_pcode1 = re.compile(r'<pre><code>([\s\S]*)</code></pre>')
    p_pcode2 = re.compile(r'<pre class=[\s\S]*><code>([\s\S]*)</code></pre>')

    block_list = []

    # 遍历每个block
    for i in range(0, len(all_blocklis)):

        # print('########################Block %s#############################\n' % str(i), all_blocklis[i])

        if i == 0:
            n_f = match_str_ptag(p_context, match_str_block(p_forward, all_blocklis[i])[0]) if p_context.findall(match_str_block(p_forward, all_blocklis[i])[0]) != [] else '-10000'
            n_f = remove_unprint_chars(n_f)
            # print('--------forward nl description-------\n', n_f)
        else:
            n_f = match_str_ptag(p_context, match_str_block(p_backward, all_blocklis[i - 1])[0]) if p_context.findall(match_str_block(p_backward, all_blocklis[i - 1])[0]) != [] else '-10000'
            n_f = remove_unprint_chars(n_f)
            # print('--------forward nl description-------\n', n_f)

        code = match_tag_code(p_hcode, p_pcode, p_pcode1, p_pcode2, match_str_block(p_code, all_blocklis[i])[0])
        # 对网页标签转义
        c = html.unescape(code)
        c = remove_unprint_chars(c)
        # print('-------------context code----------\n', c)

        if i == len(all_blocklis) - 1:
            n_b = match_str_btag(p_context, match_str_block(p_backward, all_blocklis[i])[0]) if p_context.findall(match_str_block(p_backward, all_blocklis[i])[0]) != [] else '-10000'
            n_b = remove_unprint_chars(n_b)
            # print('-----------backward nl description--------\n', n_b)
        else:
            n_b = match_str_ptag(p_context, match_str_block(p_backward, all_blocklis[i])[0]) if p_context.findall(match_str_block(p_backward, all_blocklis[i])[0]) != [] else '-10000'
            n_b = remove_unprint_chars(n_b)
            # print('-----------backward nl description--------\n', n_b)

        # print('########################Block %s#############################\n' % str(i))

        # 模块的分组
        block_tuple = (n_f, c, n_b)

        block_list.append(block_tuple)

    return block_list


''' <row Id="9" PostTypeId="1" AcceptedAnswerId="1404" CreationDate="2008-07-31T23:40:59.743" Score="1436" ViewCount="372893" Body="&lt;p&gt;Given a &lt;code&gt;DateTime&lt;/code&gt; representing a person's birthday, how do I calculate their age in years?  &lt;/p&gt;&#xA;" OwnerUserId="1" LastEditorUserId="1710577" LastEditorDisplayName="Rich B" LastEditDate="2016-10-13T14:47:03.960" LastActivityDate="2017-02-23T17:49:01.897" Title="How do I calculate someone's age in C#?" Tags="&lt;c#&gt;&lt;.net&gt;&lt;datetime&gt;" AnswerCount="58" CommentCount="6" FavoriteCount="333" CommunityOwnedDate="2011-08-16T19:40:43.080" />'''

# 存储query的数据
def process_xmldata(data_folder,lang_type,split_num,save_folder):

    # 存储网页内容
    html_lis = []

    # 不包含社交属性语料
    corpus_lis = []

    # 查询和回复的社交属性
    qaterms_dict = {}

    # 帖子id
    qidnum_lis = []

    # 接受id
    accept_lis = []

    # 查询
    qterms_dict = {}

    # 计数
    count1 = 0
    count2 = 0

    with open(data_folder + '%s_tag_2022.xml'%lang_type, 'r', encoding='utf-8') as f:
        for line in f:
            count1 += 1
            if count1 % 100000 == 0:  # 总计
                print('2022XML查询阶段已经处理--%d条数据' % count1)
            if not contain_non_English(line):
                rowTag = BeautifulSoup(line, 'lxml').row  # row标签
                # 查询
                query = get_tag_param(rowTag, 'title')
                query = remove_unprint_chars(query)
                # 判断查询是否满足要求
                if query != '-10000':  # 存在
                    # 可接受
                    accept = get_tag_param(rowTag, 'acceptedanswerid')
                    if accept != '-10000' and lang_type in get_tag_param(rowTag, "tags") :  # 存在
                        count2 += 1
                        if count2 % 100000 == 0:  # 总计
                            print('2022XML查询阶段有效处理--%d条数据' % count2)
                        # 接受id
                        accept_lis.append(int(accept))
                        # 帖子id
                        qidnum = get_tag_param(rowTag, 'id')
                        qidnum_lis.append(int(qidnum))

                        # 字典数据 （更新/添加）
                        qterms_dict[(int(qidnum), int(accept))] = (query,line)

    print('2022XML查询阶段完全遍历处理-%d条数据' % count1)
    print('2022XML查询阶段有效总计处理-%d条数据' % count2)

    print('###################################2022XML查询阶段执行完毕！###################################')

    #  判断长度是否相等
    assert len(accept_lis) == len(qterms_dict.keys())

    print('2022XML的文件可解析回复qid长度-%s' % len(qidnum_lis))

    ###############################################找寻相关的XML文件#########################################

    # accept从小到大的序号排序
    sort_accept_lis = sorted(accept_lis)

    # 索引的长度
    accept_num = len(sort_accept_lis)
    print('2022XML回复答案aid需要的解析长度-%d' % accept_num)

    # 来对索引index分割,正则匹配命令太长，必须切割！
    split_index = [sort_accept_lis[i:i + split_num] for i in range(0, len(sort_accept_lis), split_num)]
    # 执行每个部分index
    parts_file = []

    for i in range(len(split_index)):
        # linux执行命令
        eping_str = '|'.join(['(%s)' % str(s) for s in split_index[i]])
        # 每份xml命名
        epart_xml = '%s_ans_2022.xml' % lang_type
        epart_xml = epart_xml.replace('ans_2022', 'ans_2022part%d' % (i + 1))
        # 添加到列表
        parts_file.append(data_folder + epart_xml)
        # 命令字符串
        eping_cmd = 'egrep \'<row Id="(%s)"\' %s > %s' % (eping_str, data_folder + 'answer_2022.xml', data_folder + epart_xml)
        # 执行命令符
        os.popen(eping_cmd).read()  # 得到part_xml文件
        print('索引的第%d部分检索完毕！' % (i + 1))

    ping_cmd = 'cat %s > %s' % (' '.join(parts_file), data_folder + '%s_ans_2022.xml' % lang_type)

    # 执行命令
    os.popen(ping_cmd).read()  # 得到ans_xml文件
    print('XML整体文件合并完毕!')

    remove_cmd = 'rm %s' % data_folder + '%s_ans_2022part*.xml' % lang_type
    os.popen(remove_cmd).read()
    print('XML部分文件删除完毕!')

    ###################################解析具有全部accept_lis索引的ans_xml文件#######################################
    # 序号id对应accept的id  qid：accept_id (aid)
    q2aids_dict = {}
    for i in qterms_dict.keys():
        q2aids_dict[i[0]] = i[1]

    a2qids_dict = {}
    for i in qterms_dict.keys():
        a2qids_dict[i[1]] = i[0]

    # 计数
    count1 = 0
    count2 =0

    with open(data_folder + '%s_ans_2022.xml' % lang_type, 'r', encoding='utf-8') as f:
        canfind_aids = []
        # 构建词典 aid:line
        line_dict = {}
        for line in f:
            count1 += 1
            if count1 % 100000 == 0:  # 总计
                print('2022XML查询阶段已经处理--%d条数据' % count1)
            if not contain_non_English(line):
                count2 += 1
                if count2 % 100000 == 0:  # 总计
                    print('2022XML回复阶段有效处理--%d条数据' % count2)
                sta = line.find("\"")
                end = line.find("\"", sta + 1)
                # 提取line对应的accept id
                aid = int(line[(sta + 1):end])
                canfind_aids.append(aid)
                line_dict[aid] = line
        # 对查询的键值和accept键值进行合并
        same_aids = set(accept_lis) & set(canfind_aids)
        print('2022XML回复答案id合并的有效长度-%d' % len(same_aids))
        # 对合并后的qid进行从小到大排序
        sort_same_qids = sorted([a2qids_dict[aid] for aid in same_aids])

        print('###################################2022XML回复阶段执行完毕！###################################')

        for qid in sort_same_qids:
            # 创建网页
            each_html = []
            # 添加内容
            each_html.append((qid, q2aids_dict[qid]))
            query = qterms_dict[(qid, q2aids_dict[qid])][0]
            each_html.append(query)

            qline = qterms_dict[(qid, q2aids_dict[qid])][1]
            # row标签
            qrowTag = BeautifulSoup(qline, 'lxml').row
            # body属性 一定存在
            qbody = get_tag_param(qrowTag, 'body')
            # <html><body>
            qhtmlSoup = BeautifulSoup(qbody, 'lxml')
            # 保存body部分
            each_html.append(str(qhtmlSoup))

            # 每行数据
            aline = line_dict[q2aids_dict[qid]]
            # row标签
            arowTag = BeautifulSoup(aline, 'lxml').row
            # 评论数
            ccomment = get_tag_param(arowTag, 'commentcount')
            # 打分数
            cscorenum = get_tag_param(arowTag, 'score')
            # 全部属性
            all_terms = qterms_dict[(qid, q2aids_dict[qid])][2] + [int(ccomment), int(cscorenum)]

            # body属性 一定存在
            abody = get_tag_param(arowTag, 'body')
            # <html><body>
            ahtmlSoup = BeautifulSoup(abody, 'lxml')
            # 保存body部分
            each_html.append(str(ahtmlSoup))

            # 抽取代码和上下文 直接用soup（不能转了str)
            block_list = parse_tag_block(ahtmlSoup)

            if block_list != []:
                # 加入属性
                qaterms_dict[qid] = all_terms
                # 加入到html列表
                html_lis.append(each_html)

                # 候选的遍历
                # [(id, index),[[si]，[si+1]] 文本块，[[c]] 代码，[q] 查询, 块长度，标签]
                for index in range(0, len(block_list)):
                    # 创建块组合
                    block_addlis = []
                    # 添加索引
                    block_addlis.append((qid, index))
                    # 添加上下文
                    block_tuple = block_list[index]
                    block_addlis.append([[block_tuple[0]], [block_tuple[-1]]])
                    # 添加代码
                    block_addlis.append([[block_tuple[1]]])
                    block_addlis.append([query])
                    block_addlis.append(len(block_list))
                    # 加入数据
                    corpus_lis.append(block_addlis)

    print('###################################2022XML回复阶段解析完毕！###################################')

    #  存在同一个qid 对不同的acceptedanswerid
    qid2aid_lis = qterms_dict.keys()

    corpus_qids = set([i[0][0] for i in corpus_lis])

    # 帖子id
    qidnum_lis = []

    # 接受id
    accept_lis = []

    # 计数
    count1 = 0
    count2 = 0

    with open(data_folder + '%s_tag_2021.xml'%lang_type, 'r', encoding='utf-8') as f:
        for line in f:
            count1 += 1
            if count1 % 100000 == 0:  # 总计
                print('2021XML查询阶段已经处理--%d条数据' % count1)
            if not contain_non_English(line):
                # row标签
                rowTag = BeautifulSoup(line, 'lxml').row
                # 查询
                query = get_tag_param(rowTag, 'title')
                query = remove_unprint_chars(query)
                # 判断查询是否满足要求
                if query != '-10000':  # 存在
                    accept = get_tag_param(rowTag, 'acceptedanswerid')
                    if accept != '-10000' and lang_type in get_tag_param(rowTag, "tags"):  # 存在
                        count2 += 1
                        if count2 % 100000 == 0:  # 总计
                            print('2021XML查询阶段有效处理--%d条数据' % count2)
                        # 接受id
                        accept_lis.append(int(accept))
                        # 帖子id
                        qidnum = get_tag_param(rowTag, 'id')
                        qidnum_lis.append(int(qidnum))

                        # 字典数据 （更新/添加）
                        qterms_dict[(int(qidnum), int(accept))] = (query,line)

    print('2021XML查询阶段完全遍历处理-%d条数据' % count1)
    print('2021XML查询阶段有效总计处理-%d条数据' % count2)

    print('###################################2021XML查询阶段执行完毕！###################################')

    # accept从小到大的序号排序
    sort_accept_lis = sorted(accept_lis)

    print('2021XML的文件可解析回复qid长度-%s' % len(qidnum_lis))

    ###############################################找寻相关的XML文件#########################################

    # 索引的长度
    accept_num = len(sort_accept_lis)
    print('2021XML回复答案aid需要的解析长度-%d' % accept_num)

    #来对索引index分割,正则匹配命令太长，必须切割！
    split_index = [sort_accept_lis[i:i + split_num] for i in range(0, len(sort_accept_lis), split_num)]
    # 执行每个部分index
    parts_file = []

    for i in range(len(split_index)):
        # linux执行命令
        eping_str = '|'.join(['(%s)' % str(s) for s in split_index[i]])
        # 每份xml命名
        epart_xml = '%s_ans_2021.xml' % lang_type
        epart_xml = epart_xml.replace('ans_2021', 'ans_2021part%d' % (i + 1))
        # 添加到列表
        parts_file.append(data_folder + epart_xml)
        # 命令字符串
        eping_cmd = 'egrep \'<row Id="(%s)"\' %s > %s' % (eping_str, data_folder + 'answer_2021.xml', data_folder + epart_xml)
        # 执行命令符
        os.popen(eping_cmd).read()  # 得到part_xml文件
        print('索引的第%d部分检索完毕！' % (i + 1))

    ping_cmd = 'cat %s > %s' % (' '.join(parts_file), data_folder + '%s_ans_2021.xml' % lang_type)

    # 执行命令
    os.popen(ping_cmd).read()  # 得到ans_xml文件
    print('XML整体文件合并完毕!')

    remove_cmd = 'rm %s' % data_folder + '%s_ans_2021part*.xml' % lang_type
    os.popen(remove_cmd).read()
    # print('XML部分文件删除完毕!')

    ##################################解析具有全部accept_lis索引的ans_xml文件#######################################
    # 序号id对应accept的id  qid：accept_id (aid)
    filter_qid2aid_lis = [i for i in qterms_dict.keys() if i not in qid2aid_lis]

    q2aids_dict = {}
    for i in filter_qid2aid_lis:
        q2aids_dict[i[0]] = i[1]

    a2qids_dict = {}
    for i in filter_qid2aid_lis:
        a2qids_dict[i[1]] = i[0]

    # 计数
    count1 = 0
    count2 = 0

    with open(data_folder + '%s_ans_2021.xml' % lang_type, 'r', encoding='utf-8') as f:
        canfind_aids = []
        # 构建词典 aid:line
        line_dict = {}
        for line in f:
            count1 += 1
            if count1 % 100000 == 0:  # 总计
                print('2021XML回复阶段已经处理--%d条数据'%count1)
            if not contain_non_English(line):
                count2 += 1
                if count2 % 100000 == 0:  # 总计
                    print('2021XML回复阶段有效处理--%d条数据'%count2)
                sta = line.find("\"")
                end = line.find("\"", sta + 1)
                # 提取line对应的accept id
                aid = int(line[(sta + 1):end])
                canfind_aids.append(aid)
                # 更新网页
                line_dict[aid] = line

        # 对查询的键值和accept键值进行合并
        same_aids = set(accept_lis) & set(canfind_aids)
        print('2021XML回复答案aid合并的有效长度-%d' % len(same_aids))
        # qterms_dict收集了aid,但a2qids_dict没收录
        common_aids = set(same_aids) & set(a2qids_dict.keys())
        # 对合并后的qid进行排序
        sort_same_qids = sorted([a2qids_dict[aid] for aid in common_aids])
        # 已经重复的不在加入
        filter_same_qids = [i for i in sort_same_qids if i not in corpus_qids]

        print('###################################2021XML回复阶段执行完毕！###################################')

        for qid in filter_same_qids:
            # 创建网页
            each_html = []
            # 添加内容
            each_html.append((qid, q2aids_dict[qid]))
            query = qterms_dict[(qid, q2aids_dict[qid])][0]
            each_html.append(query)

            qline = qterms_dict[(qid, q2aids_dict[qid])][1]
            # row标签
            qrowTag = BeautifulSoup(qline, 'lxml').row
            # body属性 一定存在
            qbody = get_tag_param(qrowTag, 'body')
            # <html><body>
            qhtmlSoup = BeautifulSoup(qbody, 'lxml')
            # 保存body部分
            each_html.append(str(qhtmlSoup))

            # 每行数据
            aline = line_dict[q2aids_dict[qid]]
            # row标签
            arowTag = BeautifulSoup(aline, 'lxml').row
            # 评论数
            ccomment = get_tag_param(arowTag, 'commentcount')
            # 打分数
            cscorenum = get_tag_param(arowTag, 'score')
            # 全部属性
            all_terms = qterms_dict[(qid, q2aids_dict[qid])][2] + [int(ccomment), int(cscorenum)]

            # body属性 一定存在
            abody = get_tag_param(arowTag, 'body')
            # <html><body>
            ahtmlSoup = BeautifulSoup(abody, 'lxml')
            # 保存body部分
            each_html.append(str(ahtmlSoup))

            # 抽取代码和上下文 直接用soup（不能转了str)
            block_list = parse_tag_block(ahtmlSoup)

            if block_list != []:
                # 加入属性
                qaterms_dict[qid] = all_terms
                # 加入到html列表
                html_lis.append(each_html)

                # 候选的遍历
                # [(id, index),[[si]，[si+1]] 文本块，[[c]] 代码，[q] 查询, 块长度，标签]
                for index in range(0, len(block_list)):
                    # 创建块组合
                    block_addlis = []
                    # 添加索引
                    block_addlis.append((qid, index))
                    # 添加上下文
                    block_tuple = block_list[index]
                    block_addlis.append([[block_tuple[0]], [block_tuple[-1]]])
                    # 添加代码
                    block_addlis.append([[block_tuple[1]]])
                    block_addlis.append([query])
                    block_addlis.append(len(block_list))
                    # 加入数据
                    corpus_lis.append(block_addlis)

    print('###################################2021XML回复阶段解析完毕！###################################')

    #  存在同一个qid 对不同的acceptedanswerid
    qid2aid_lis = qterms_dict.keys()

    corpus_qids = set([i[0][0] for i in corpus_lis])

    # 帖子id
    qidnum_lis = []

    # 接受id
    accept_lis = []

    # 计数
    count1 = 0
    count2 = 0

    with open(data_folder + '%s_tag_2020.xml'%lang_type, 'r', encoding='utf-8') as f:
        for line in f:
            count1 += 1
            if count1 % 100000 == 0:  # 总计
                print('2020XML查询阶段已经处理--%d条数据' % count1)
            if not contain_non_English(line):
                # row标签
                rowTag = BeautifulSoup(line, 'lxml').row
                # 查询
                query = get_tag_param(rowTag, 'title')
                query = remove_unprint_chars(query)
                # 判断查询是否满足要求
                if query != '-10000':  # 存在
                    accept = get_tag_param(rowTag, 'acceptedanswerid')
                    if accept != '-10000' and lang_type in get_tag_param(rowTag, "tags"):  # 存在
                        count2 += 1
                        if count2 % 100000 == 0:  # 总计
                            print('2020XML查询阶段有效处理--%d条数据' % count2)
                        # 接受id
                        accept_lis.append(int(accept))
                        # 帖子id
                        qidnum = get_tag_param(rowTag, 'id')
                        qidnum_lis.append(int(qidnum))

                        # 字典数据 （更新/添加）
                        qterms_dict[(int(qidnum), int(accept))] = (query,line)

    print('2020XML查询阶段完全遍历处理-%d条数据' % count1)
    print('2020XML查询阶段有效总计处理-%d条数据' % count2)

    print('###################################2020XML查询阶段执行完毕！###################################')

    # accept从小到大的序号排序
    sort_accept_lis = sorted(accept_lis)

    print('2020XML的文件可解析回复qid长度-%s' % len(qidnum_lis))

    ###############################################找寻相关的XML文件#########################################

    # 索引的长度
    accept_num = len(sort_accept_lis)
    print('2020XML回复答案aid需要的解析长度-%d' % accept_num)

    # 来对索引index分割,正则匹配命令太长，必须切割！
    split_index = [sort_accept_lis[i:i + split_num] for i in range(0, len(sort_accept_lis), split_num)]
    # 执行每个部分index
    parts_file = []

    for i in range(len(split_index)):
        # linux执行命令
        eping_str = '|'.join(['(%s)' % str(s) for s in split_index[i]])
        # 每份xml命名
        epart_xml = '%s_ans_2020.xml' % lang_type
        epart_xml = epart_xml.replace('ans_2020', 'ans_2020part%d' % (i + 1))
        # 添加到列表
        parts_file.append(data_folder + epart_xml)
        # 命令字符串
        eping_cmd = 'egrep \'<row Id="(%s)"\' %s > %s' % (eping_str, data_folder + 'answer_2020.xml', data_folder + epart_xml)
        # 执行命令符
        os.popen(eping_cmd).read()  # 得到part_xml文件
        print('索引的第%d部分检索完毕！' % (i + 1))

    ping_cmd = 'cat %s > %s' % (' '.join(parts_file), data_folder + '%s_ans_2020.xml' % lang_type)

    # 执行命令
    os.popen(ping_cmd).read()  # 得到ans_xml文件
    print('XML整体文件合并完毕!')

    remove_cmd = 'rm %s' % data_folder + '%s_ans_2020part*.xml' % lang_type
    os.popen(remove_cmd).read()
    # print('XML部分文件删除完毕!')

    ###################################解析具有全部accept_lis索引的ans_xml文件#######################################
    # 序号id对应accept的id  qid：accept_id (aid)
    filter_qid2aid_lis = [i for i in qterms_dict.keys() if i not in qid2aid_lis]

    q2aids_dict = {}
    for i in filter_qid2aid_lis:
        q2aids_dict[i[0]] = i[1]

    a2qids_dict = {}
    for i in filter_qid2aid_lis:
        a2qids_dict[i[1]] = i[0]

    # 计数
    count1 = 0
    count2 = 0

    with open(data_folder + '%s_ans_2020.xml' % lang_type, 'r', encoding='utf-8') as f:
        canfind_aids = []
        # 构建词典 aid:line
        line_dict = {}
        for line in f:
            count1 += 1
            if count1 % 100000 == 0:  # 总计
                print('2020XML回复阶段已经处理--%d条数据' % count1)
            if not contain_non_English(line):
                count2 += 1
                if count2 % 100000 == 0:  # 总计
                    print('2020XML回复阶段有效处理--%d条数据' % count2)
                sta = line.find("\"")
                end = line.find("\"", sta + 1)
                # 提取line对应的accept id
                aid = int(line[(sta + 1):end])
                canfind_aids.append(aid)
                # 更新网页
                line_dict[aid] = line

        # 对查询的键值和accept键值进行合并
        same_aids = set(accept_lis) & set(canfind_aids)
        print('2020XML回复答案aid合并的有效长度-%d' % len(same_aids))
        # qterms_dict收集了aid,但a2qids_dict没收录
        common_aids = set(same_aids) & set(a2qids_dict.keys())
        # 对合并后的qid进行排序
        sort_same_qids = sorted([a2qids_dict[aid] for aid in common_aids])
        # 已经重复的不在加入
        filter_same_qids = [i for i in sort_same_qids if i not in corpus_qids]

        print('###################################2020XML回复阶段执行完毕！###################################')

        for qid in filter_same_qids:
            # 创建网页
            each_html = []
            # 添加内容
            each_html.append((qid, q2aids_dict[qid]))
            query = qterms_dict[(qid, q2aids_dict[qid])][0]
            each_html.append(query)

            qline = qterms_dict[(qid, q2aids_dict[qid])][1]
            # row标签
            qrowTag = BeautifulSoup(qline, 'lxml').row
            # body属性 一定存在
            qbody = get_tag_param(qrowTag, 'body')
            # <html><body>
            qhtmlSoup = BeautifulSoup(qbody, 'lxml')
            # 保存body部分
            each_html.append(str(qhtmlSoup))

            # 每行数据
            aline = line_dict[q2aids_dict[qid]]
            # row标签
            arowTag = BeautifulSoup(aline, 'lxml').row
            # 评论数
            ccomment = get_tag_param(arowTag, 'commentcount')
            # 打分数
            cscorenum = get_tag_param(arowTag, 'score')
            # 全部属性
            all_terms = qterms_dict[(qid, q2aids_dict[qid])][2] + [int(ccomment), int(cscorenum)]

            # body属性 一定存在
            abody = get_tag_param(arowTag, 'body')
            # <html><body>
            ahtmlSoup = BeautifulSoup(abody, 'lxml')
            # 保存body部分
            each_html.append(str(ahtmlSoup))

            # 抽取代码和上下文 直接用soup（不能转了str)
            block_list = parse_tag_block(ahtmlSoup)

            if block_list != []:
                # 加入属性
                qaterms_dict[qid] = all_terms
                # 加入到html列表
                html_lis.append(each_html)

                # 候选的遍历
                # [(id, index),[[si]，[si+1]] 文本块，[[c]] 代码，[q] 查询, 块长度，标签]
                for index in range(0, len(block_list)):
                    # 创建块组合
                    block_addlis = []
                    # 添加索引
                    block_addlis.append((qid, index))
                    # 添加上下文
                    block_tuple = block_list[index]
                    block_addlis.append([[block_tuple[0]], [block_tuple[-1]]])
                    # 添加代码
                    block_addlis.append([[block_tuple[1]]])
                    block_addlis.append([query])
                    block_addlis.append(len(block_list))
                    # 加入数据
                    corpus_lis.append(block_addlis)

    print('###################################2020XML回复阶段解析完毕！###################################')

    #  存在同一个qid 对不同的acceptedanswerid
    qid2aid_lis = qterms_dict.keys()

    corpus_qids = set([i[0][0] for i in corpus_lis])

    # 帖子id
    qidnum_lis = []

    # 接受id
    accept_lis = []

    # 计数
    count1 = 0
    count2 = 0

    with open(data_folder + '%s_tag_2019.xml' % lang_type, 'r', encoding='utf-8') as f:
        for line in f:
            count1 += 1
            if count1 % 100000 == 0:  # 总计
                print('2019XML查询阶段已经处理--%d条数据' % count1)
            if not contain_non_English(line):
                # row标签
                rowTag = BeautifulSoup(line, 'lxml').row
                # 查询
                query = get_tag_param(rowTag, 'title')
                query = remove_unprint_chars(query)
                # 判断查询是否满足要求
                if query != '-10000':  # 存在
                    accept = get_tag_param(rowTag, 'acceptedanswerid')
                    if accept != '-10000' and lang_type in get_tag_param(rowTag, "tags"):  # 存在
                        count2 += 1
                        if count2 % 100000 == 0:  # 总计
                            print('2019XML查询阶段有效处理--%d条数据' % count2)
                        # 接受id
                        accept_lis.append(int(accept))
                        # 帖子id
                        qidnum = get_tag_param(rowTag, 'id')
                        qidnum_lis.append(int(qidnum))

                        # 字典数据 （更新/添加）
                        qterms_dict[(int(qidnum), int(accept))] = (query,line)

    print('2019XML查询阶段完全遍历处理-%d条数据' % count1)
    print('2019XML查询阶段有效总计处理-%d条数据' % count2)

    print('###################################2019XML查询阶段执行完毕！###################################')

    # accept从小到大的序号排序
    sort_accept_lis = sorted(accept_lis)

    print('2019XML的文件可解析回复qid长度-%s' % len(qidnum_lis))

    ###############################################找寻相关的XML文件#########################################

    # 索引的长度
    accept_num = len(sort_accept_lis)
    print('2019XML回复答案aid需要的解析长度-%d' % accept_num)

    # 来对索引index分割,正则匹配命令太长，必须切割！
    split_index = [sort_accept_lis[i:i + split_num] for i in  range(0, len(sort_accept_lis), split_num)]
    # # 执行每个部分index
    parts_file = []

    for i in range(len(split_index)):
        # linux执行命令
        eping_str = '|'.join(['(%s)' % str(s) for s in split_index[i]])
        # 每份xml命名
        epart_xml = '%s_ans_2019.xml' % lang_type
        epart_xml = epart_xml.replace('ans_2019', 'ans_2019part%d' % (i + 1))
        # 添加到列表
        parts_file.append(data_folder + epart_xml)
        # 命令字符串
        eping_cmd = 'egrep \'<row Id="(%s)"\' %s > %s' % (
        eping_str, data_folder + 'answer_2019.xml', data_folder + epart_xml)
        # 执行命令符
        os.popen(eping_cmd).read()  # 得到part_xml文件
        print('索引的第%d部分检索完毕！' % (i + 1))

    ping_cmd = 'cat %s > %s' % (' '.join(parts_file), data_folder + '%s_ans_2019.xml' % lang_type)

    # 执行命令
    os.popen(ping_cmd).read()  # 得到ans_xml文件
    print('XML整体文件合并完毕!')

    remove_cmd = 'rm %s' % data_folder + '%s_ans_2019part*.xml' % lang_type
    os.popen(remove_cmd).read()
    # print('XML部分文件删除完毕!')

    ###################################解析具有全部accept_lis索引的ans_xml文件#######################################
    # 序号id对应accept的id  qid：accept_id (aid)
    filter_qid2aid_lis = [i for i in qterms_dict.keys() if i not in qid2aid_lis]

    q2aids_dict = {}
    for i in filter_qid2aid_lis:
        q2aids_dict[i[0]] = i[1]

    a2qids_dict = {}
    for i in filter_qid2aid_lis:
        a2qids_dict[i[1]] = i[0]

    # 计数
    count1 = 0
    count2 = 0

    with open(data_folder + '%s_ans_2019.xml' % lang_type, 'r', encoding='utf-8') as f:
        canfind_aids = []
        # 构建词典 aid:line
        line_dict = {}
        for line in f:
            count1 += 1
            if count1 % 100000 == 0:  # 总计
                print('2019XML回复阶段已经处理--%d条数据' % count1)
            if not contain_non_English(line):
                count2 += 1
                if count2 % 100000 == 0:  # 总计
                    print('2019XML回复阶段有效处理--%d条数据' % count2)
                sta = line.find("\"")
                end = line.find("\"", sta + 1)
                # 提取line对应的accept id
                aid = int(line[(sta + 1):end])
                canfind_aids.append(aid)
                # 更新网页
                line_dict[aid] = line

        # 对查询的键值和accept键值进行合并
        same_aids = set(accept_lis) & set(canfind_aids)
        print('2019XML回复答案aid合并的有效长度-%d' % len(same_aids))
        # qterms_dict收集了aid,但a2qids_dict没收录
        common_aids = set(same_aids) & set(a2qids_dict.keys())
        # 对合并后的qid进行排序
        sort_same_qids = sorted([a2qids_dict[aid] for aid in common_aids])
        # 已经重复的不在加入
        filter_same_qids = [i for i in sort_same_qids if i not in corpus_qids]

        print('###################################2019XML回复阶段执行完毕！###################################')

        for qid in filter_same_qids:
            # 创建网页
            each_html = []
            # 添加内容
            each_html.append((qid, q2aids_dict[qid]))
            query = qterms_dict[(qid, q2aids_dict[qid])][0]
            each_html.append(query)

            qline = qterms_dict[(qid, q2aids_dict[qid])][1]
            # row标签
            qrowTag = BeautifulSoup(qline, 'lxml').row
            # body属性 一定存在
            qbody = get_tag_param(qrowTag, 'body')
            # <html><body>
            qhtmlSoup = BeautifulSoup(qbody, 'lxml')
            # 保存body部分
            each_html.append(str(qhtmlSoup))

            # 每行数据
            aline = line_dict[q2aids_dict[qid]]
            # row标签
            arowTag = BeautifulSoup(aline, 'lxml').row
            # 评论数
            ccomment = get_tag_param(arowTag, 'commentcount')
            # 打分数
            cscorenum = get_tag_param(arowTag, 'score')
            # 全部属性
            all_terms = qterms_dict[(qid, q2aids_dict[qid])][2] + [int(ccomment), int(cscorenum)]

            # body属性 一定存在
            abody = get_tag_param(arowTag, 'body')
            # <html><body>
            ahtmlSoup = BeautifulSoup(abody, 'lxml')
            # 保存body部分
            each_html.append(str(ahtmlSoup))

            # 抽取代码和上下文 直接用soup（不能转了str)
            block_list = parse_tag_block(ahtmlSoup)

            if block_list != []:
                # 加入属性
                qaterms_dict[qid] = all_terms
                # 加入到html列表
                html_lis.append(each_html)

                # 候选的遍历
                # [(id, index),[[si]，[si+1]] 文本块，[[c]] 代码，[q] 查询, 块长度，标签]
                for index in range(0, len(block_list)):
                    # 创建块组合
                    block_addlis = []
                    # 添加索引
                    block_addlis.append((qid, index))
                    # 添加上下文
                    block_tuple = block_list[index]
                    block_addlis.append([[block_tuple[0]], [block_tuple[-1]]])
                    # 添加代码
                    block_addlis.append([[block_tuple[1]]])
                    block_addlis.append([query])
                    block_addlis.append(len(block_list))
                    # 加入数据
                    corpus_lis.append(block_addlis)

    print('###################################2019XML回复阶段解析完毕！###################################')

    #  存在同一个qid 对不同的acceptedanswerid
    qid2aid_lis = qterms_dict.keys()

    corpus_qids = set([i[0][0] for i in corpus_lis])

    # 帖子id
    qidnum_lis = []

    # 接受id
    accept_lis = []

    # 计数
    count1 = 0
    count2 = 0

    with open(data_folder + '%s_tag_2018.xml' % lang_type, 'r', encoding='utf-8') as f:
        for line in f:
            count1 += 1
            if count1 % 100000 == 0:  # 总计
                print('2018XML查询阶段已经处理--%d条数据' % count1)
            if not contain_non_English(line):
                # row标签
                rowTag = BeautifulSoup(line, 'lxml').row
                # 查询
                query = get_tag_param(rowTag, 'title')
                query = remove_unprint_chars(query)
                # 判断查询是否满足要求
                if query != '-10000':  # 存在
                    accept = get_tag_param(rowTag, 'acceptedanswerid')
                    if accept != '-10000' and lang_type in get_tag_param(rowTag, "tags"):  # 存在
                        count2 += 1
                        if count2 % 100000 == 0:  # 总计
                            print('2018XML查询阶段有效处理--%d条数据' % count2)
                        # 接受id
                        accept_lis.append(int(accept))
                        # 帖子id
                        qidnum = get_tag_param(rowTag, 'id')
                        qidnum_lis.append(int(qidnum))

                        # 字典数据 （更新/添加）
                        qterms_dict[(int(qidnum), int(accept))] = (query,line)

    print('2018XML查询阶段完全遍历处理-%d条数据' % count1)
    print('2018XML查询阶段有效总计处理-%d条数据' % count2)

    print('###################################2018XML查询阶段执行完毕！###################################')

    # accept从小到大的序号排序
    sort_accept_lis = sorted(accept_lis)

    print('2018XML的文件可解析回复qid长度-%s' % len(qidnum_lis))

    ###############################################找寻相关的XML文件#########################################

    # 索引的长度
    accept_num = len(sort_accept_lis)
    print('2018XML回复答案aid需要的解析长度-%d' % accept_num)

    #来对索引index分割,正则匹配命令太长，必须切割！
    split_index = [sort_accept_lis[i:i + split_num] for i in   range(0, len(sort_accept_lis), split_num)]
    # 执行每个部分index
    parts_file = []

    for i in range(len(split_index)):
        # linux执行命令
        eping_str = '|'.join(['(%s)' % str(s) for s in split_index[i]])
        # 每份xml命名
        epart_xml = '%s_ans_2018.xml' % lang_type
        epart_xml = epart_xml.replace('ans_2018', 'ans_2018part%d' % (i + 1))
        # 添加到列表
        parts_file.append(data_folder + epart_xml)
        # 命令字符串
        eping_cmd = 'egrep \'<row Id="(%s)"\' %s > %s' % (
            eping_str, data_folder + 'answer_2018.xml', data_folder + epart_xml)
        # 执行命令符
        os.popen(eping_cmd).read()  # 得到part_xml文件
        print('索引的第%d部分检索完毕！' % (i + 1))

    ping_cmd = 'cat %s > %s' % (' '.join(parts_file), data_folder + '%s_ans_2018.xml' % lang_type)

    # 执行命令
    os.popen(ping_cmd).read()  # 得到ans_xml文件
    print('XML整体文件合并完毕!')

    remove_cmd = 'rm %s' % data_folder + '%s_ans_2018part*.xml' % lang_type
    os.popen(remove_cmd).read()
    # print('XML部分文件删除完毕!')

    ###################################解析具有全部accept_lis索引的ans_xml文件#######################################
    # 序号id对应accept的id  qid：accept_id (aid)
    filter_qid2aid_lis = [i for i in qterms_dict.keys() if i not in qid2aid_lis]

    q2aids_dict = {}
    for i in filter_qid2aid_lis:
        q2aids_dict[i[0]] = i[1]

    a2qids_dict = {}
    for i in filter_qid2aid_lis:
        a2qids_dict[i[1]] = i[0]

    # 计数
    count1 = 0
    count2 = 0

    with open(data_folder + '%s_ans_2018.xml' % lang_type, 'r', encoding='utf-8') as f:
        canfind_aids = []
        for line in f:
            count1 += 1
            if count1 % 100000 == 0:  # 总计
                print('2018XML回复阶段已经处理--%d条数据' % count1)
            if not contain_non_English(line):
                count2 += 1
                if count2 % 100000 == 0:  # 总计
                    print('2018XML回复阶段有效处理--%d条数据' % count2)
                sta = line.find("\"")
                end = line.find("\"", sta + 1)
                # 提取line对应的accept id
                aid = int(line[(sta + 1):end])
                canfind_aids.append(aid)
                # 更新网页
                line_dict[aid] = line

        # 对查询的键值和accept键值进行合并
        same_aids = set(accept_lis) & set(canfind_aids)
        print('2018XML回复答案aid合并的有效长度-%d' % len(same_aids))
        # qterms_dict收集了aid,但a2qids_dict没收录
        common_aids = set(same_aids) & set(a2qids_dict.keys())
        # 对合并后的qid进行排序
        sort_same_qids = sorted([a2qids_dict[aid] for aid in common_aids])
        # 已经重复的不在加入
        filter_same_qids = [i for i in sort_same_qids if i not in corpus_qids]

        print('###################################2018XML回复阶段执行完毕！###################################')

        for qid in filter_same_qids:
            # 创建网页
            each_html = []
            # 添加内容
            each_html.append((qid, q2aids_dict[qid]))
            query = qterms_dict[(qid, q2aids_dict[qid])][0]
            each_html.append(query)

            qline = qterms_dict[(qid, q2aids_dict[qid])][1]
            # row标签
            qrowTag = BeautifulSoup(qline, 'lxml').row
            # body属性 一定存在
            qbody = get_tag_param(qrowTag, 'body')
            # <html><body>
            qhtmlSoup = BeautifulSoup(qbody, 'lxml')
            # 保存body部分
            each_html.append(str(qhtmlSoup))

            # 每行数据
            aline = line_dict[q2aids_dict[qid]]
            # row标签
            arowTag = BeautifulSoup(aline, 'lxml').row
            # 评论数
            ccomment = get_tag_param(arowTag, 'commentcount')
            # 打分数
            cscorenum = get_tag_param(arowTag, 'score')
            # 全部属性
            all_terms = qterms_dict[(qid, q2aids_dict[qid])][2] + [int(ccomment), int(cscorenum)]

            # body属性 一定存在
            abody = get_tag_param(arowTag, 'body')
            # <html><body>
            ahtmlSoup = BeautifulSoup(abody, 'lxml')
            # 保存body部分
            each_html.append(str(ahtmlSoup))

            # 抽取代码和上下文 直接用soup（不能转了str)
            block_list = parse_tag_block(ahtmlSoup)

            if block_list != []:
                # 加入属性
                qaterms_dict[qid] = all_terms
                # 加入到html列表
                html_lis.append(each_html)

                # 候选的遍历
                # [(id, index),[[si]，[si+1]] 文本块，[[c]] 代码，[q] 查询, 块长度，标签]
                for index in range(0, len(block_list)):
                    # 创建块组合
                    block_addlis = []
                    # 添加索引
                    block_addlis.append((qid, index))
                    # 添加上下文
                    block_tuple = block_list[index]
                    block_addlis.append([[block_tuple[0]], [block_tuple[-1]]])
                    # 添加代码
                    block_addlis.append([[block_tuple[1]]])
                    block_addlis.append([query])
                    block_addlis.append(len(block_list))
                    # 加入数据
                    corpus_lis.append(block_addlis)

    print('###################################2018XML回复阶段解析完毕！###################################')

    #  存在同一个qid 对不同的acceptedanswerid
    qid2aid_lis = qterms_dict.keys()

    corpus_qids = set([i[0][0] for i in corpus_lis])
    # 帖子id
    qidnum_lis = []

    # 接受id
    accept_lis = []

    # 计数
    count1 = 0
    count2 = 0

    with open(data_folder + '%s_tag_2017.xml' % lang_type, 'r', encoding='utf-8') as f:
        for line in f:
            count1 += 1
            if count1 % 100000 == 0:  # 总计
                print('2017XML查询阶段已经处理--%d条数据' % count1)
            if not contain_non_English(line):
                # row标签
                rowTag = BeautifulSoup(line, 'lxml').row
                # 查询
                query = get_tag_param(rowTag, 'title')
                query = remove_unprint_chars(query)
                # 判断查询是否满足要求
                if query != '-10000':  # 存在
                    accept = get_tag_param(rowTag, 'acceptedanswerid')
                    if accept != '-10000' and lang_type in get_tag_param(rowTag, "tags"):  # 存在
                        count2 += 1
                        if count2 % 100000 == 0:  # 总计
                            print('2017XML查询阶段有效处理--%d条数据' % count2)
                        # 接受id
                        accept_lis.append(int(accept))
                        # 帖子id
                        qidnum = get_tag_param(rowTag, 'id')
                        qidnum_lis.append(int(qidnum))

                        # 字典数据 （更新/添加）
                        qterms_dict[(int(qidnum), int(accept))] = (query,line)

    print('2017XML查询阶段完全遍历处理-%d条数据' % count1)
    print('2017XML查询阶段有效总计处理-%d条数据' % count2)

    print('###################################2017XML查询阶段执行完毕！###################################')

    # accept从小到大的序号排序
    new3_new_sort_accept_lis = sorted(accept_lis)

    print('2017XML的文件可解析回复qid长度-%s' % len(qidnum_lis))

    ###############################################找寻相关的XML文件#########################################

    # 索引的长度
    new3_new_accept_num = len(new3_new_sort_accept_lis)
    print('2017XML回复答案aid需要的解析长度-%d' % new3_new_accept_num)

    # 来对索引index分割,正则匹配命令太长，必须切割！
    split_index = [sort_accept_lis[i:i + split_num] for i in range(0, len(sort_accept_lis), split_num)]
    # 执行每个部分index
    parts_file = []

    for i in range(len(split_index)):
        # linux执行命令
        eping_str = '|'.join(['(%s)' % str(s) for s in split_index[i]])
        # 每份xml命名
        epart_xml = '%s_ans_2017.xml' % lang_type
        epart_xml = epart_xml.replace('ans_2017', 'ans_2017part%d' % (i + 1))
        # 添加到列表
        parts_file.append(data_folder + epart_xml)
        # 命令字符串
        eping_cmd = 'egrep \'<row Id="(%s)"\' %s > %s' % (eping_str, data_folder + 'answer_2017.xml', data_folder + epart_xml)
        # 执行命令符
        os.popen(eping_cmd).read()  # 得到part_xml文件
        print('索引的第%d部分检索完毕！' % (i + 1))

    ping_cmd = 'cat %s > %s' % (' '.join(parts_file), data_folder + '%s_ans_2017.xml' % lang_type)

    # 执行命令
    os.popen(ping_cmd).read()  # 得到ans_xml文件
    print('XML整体文件合并完毕!')

    remove_cmd = 'rm %s' % data_folder + '%s_ans_2017part*.xml' % lang_type
    os.popen(remove_cmd).read()
    # print('XML部分文件删除完毕!')

    ###################################解析具有全部accept_lis索引的ans_xml文件#######################################
    filter_qid2aid_lis = [i for i in qterms_dict.keys() if i not in qid2aid_lis]

    q2aids_dict = {}
    for i in filter_qid2aid_lis:
        q2aids_dict[i[0]] = i[1]

    a2qids_dict = {}
    for i in filter_qid2aid_lis:
        a2qids_dict[i[1]] = i[0]

    # 计数
    count1 = 0
    count2 = 0

    with open(data_folder + '%s_ans_2017.xml' % lang_type, 'r', encoding='utf-8') as f:
        canfind_aids = []
        for line in f:
            count1 += 1
            if count1 % 100000 == 0:  # 总计
                print('2017XML回复阶段已经处理--%d条数据' % count1)
            if not contain_non_English(line):
                count2 += 1
                if count2 % 100000 == 0:  # 总计
                    print('2017XML回复阶段有效处理--%d条数据'%count2)
                sta = line.find("\"")
                end = line.find("\"", sta + 1)
                # 提取line对应的accept id
                aid = int(line[(sta + 1):end])
                canfind_aids.append(aid)
                # 更新网页
                line_dict[aid] = line

        # 对查询的键值和accept键值进行合并
        same_aids = set(accept_lis) & set(canfind_aids)
        print('2017XML回复答案aid合并的有效长度-%d' % len(same_aids))
        # qterms_dict收集了aid,但a2qids_dict没收录
        common_aids = set(same_aids) & set(a2qids_dict.keys())
        # 对合并后的qid进行排序
        sort_same_qids = sorted([a2qids_dict[aid] for aid in common_aids])
        # 已经重复的不在加入
        filter_same_qids = [i for i in sort_same_qids if i not in corpus_qids]

        print('###################################2017XML回复阶段执行完毕！###################################')

        for qid in filter_same_qids:
            # 创建网页
            each_html = []
            # 添加内容
            each_html.append((qid, q2aids_dict[qid]))
            query = qterms_dict[(qid, q2aids_dict[qid])][0]
            each_html.append(query)

            qline = qterms_dict[(qid, q2aids_dict[qid])][1]
            # row标签
            qrowTag = BeautifulSoup(qline, 'lxml').row
            # body属性 一定存在
            qbody = get_tag_param(qrowTag, 'body')
            # <html><body>
            qhtmlSoup = BeautifulSoup(qbody, 'lxml')
            # 保存body部分
            each_html.append(str(qhtmlSoup))

            # 每行数据
            aline = line_dict[q2aids_dict[qid]]
            # row标签
            arowTag = BeautifulSoup(aline, 'lxml').row
            # 评论数
            ccomment = get_tag_param(arowTag, 'commentcount')
            # 打分数
            cscorenum = get_tag_param(arowTag, 'score')
            # 全部属性
            all_terms = qterms_dict[(qid, q2aids_dict[qid])][2] + [int(ccomment), int(cscorenum)]

            # body属性 一定存在
            abody = get_tag_param(arowTag, 'body')
            # <html><body>
            ahtmlSoup = BeautifulSoup(abody, 'lxml')
            # 保存body部分
            each_html.append(str(ahtmlSoup))

            # 抽取代码和上下文 直接用soup（不能转了str)
            block_list = parse_tag_block(ahtmlSoup)

            if block_list != []:
                # 加入属性
                qaterms_dict[qid] = all_terms
                # 加入到html列表
                html_lis.append(each_html)

                # 候选的遍历
                # [(id, index),[[si]，[si+1]] 文本块，[[c]] 代码，[q] 查询, 块长度，标签]
                for index in range(0, len(block_list)):
                    # 创建块组合
                    block_addlis = []
                    # 添加索引
                    block_addlis.append((qid, index))
                    # 添加上下文
                    block_tuple = block_list[index]
                    block_addlis.append([[block_tuple[0]], [block_tuple[-1]]])
                    # 添加代码
                    block_addlis.append([[block_tuple[1]]])
                    block_addlis.append([query])
                    block_addlis.append(len(block_list))
                    # 加入数据
                    corpus_lis.append(block_addlis)

    print('###################################2017XML回复阶段解析完毕！###################################')

    # 判断查询为空
    noquery_qids = set([i[0][0] for i in corpus_lis if python_structured.python_query_parse(i[3][0]) == []]) if lang_type == 'python' else \
                  (set([i[0][0] for i in corpus_lis if sqlang_structured.sqlang_query_parse(i[3][0]) == []]) if lang_type == 'sql' else \
                   set([i[0][0] for i in corpus_lis if javang_structured.javang_query_parse(i[3][0]) == []]))

    print('corpus最终查询解析为空的qid索引的长度-%d\n'%len(noquery_qids), noquery_qids)

    # 过滤掉不满足要求
    corpus_lis = [i for i in corpus_lis if i[0][0] not in noquery_qids]

    html_lis = [i for i in html_lis if i[0][0] not in noquery_qids]

    # 删除指定键值
    [qaterms_dict.pop(i,'Key does not exist') for i in noquery_qids]

    # 判断代码结构
    noparse_qids = set([i[0][0] for i in  corpus_lis if python_structured.python_code_parse(i[2][0][0]) == '-1000']) if lang_type == 'python' else \
                   (set([i[0][0] for i in corpus_lis if sqlang_structured.sqlang_code_parse(i[2][0][0]) == '-1000']) if lang_type == 'sql' else \
                    set([i[0][0] for i in corpus_lis if javang_structured.javang_code_parse(i[2][0][0]) == '-1000']))

    print('corpus最终代码无法解析的qid索引的长度-%d\n'% len(noparse_qids), noparse_qids)

    # 过滤掉不满足要求
    corpus_lis = [i for i in corpus_lis if i[0][0] not in noparse_qids]
    corpus_qids = set([i[0][0] for i in corpus_lis])
    print(len(corpus_qids))

    html_lis = [i for i in html_lis if i[0][0] not in noparse_qids]
    html_qids = [i[0][0] for i in html_lis]
    print(len(html_qids))

    # 删除指定键值
    [qaterms_dict.pop(i, 'Key does not exist') for i in noparse_qids]
    print(len(qaterms_dict))

    new_final_qids = set([i[0][0] for i in corpus_lis])

    # 从小到大排序  [(qid,aid), query, qhtmlstr, ahtmlstr]
    sort_html_lis = sorted(html_lis, key=lambda x: (x[0][0], x[0][1]))
    # [(qid, aid),  query,  qhtmlstr, ahtmlstr, soical_terms]
    new_html_lis=[]
    for i in sort_html_lis:
        i.append(qaterms_dict[i[0][0]])
        new_html_lis.append(i)

    # 把所有数据存放到pickle文件夹
    with open(save_folder+'%s_corpus_qid_allterms.pickle'%lang_type, 'wb') as f:
        pickle.dump(new_html_lis, f)

    with open(save_folder+'%s_corpus_qid_qasoup.html'%lang_type, "w") as f:
        # 把内容写到corpus_soup.html
        for i in sort_html_lis:
            f.write('qid & accept id:\n %s \n'% str(i[0]))
            f.write('query:\n %s \n' % i[1])
            f.write('qsoup:\n %s \n' % i[2].replace('\n', '\\n'))
            f.write('qsoup wrap:\n %s \n'% i[2])
            f.write('asoup:\n %s \n' % i[3].replace('\n', '\\n'))
            f.write('asoup wrap:\n %s \n'% i[3])
        f.close()

    # 从小到大排序
    sort_corpus_lis = sorted(corpus_lis, key=lambda x: (x[0][0], x[0][1]))

    with open(save_folder+'%s_like_corpus_qid2index_blocks_unlabeled.pickle'%lang_type, "wb") as f:
        pickle.dump(sort_corpus_lis, f)

    print('corpus最终能对应qid索引长度-%d'%len(new_final_qids))


# 参数配置设置
data_folder = '../corpus_xml/'
save_folder = './corpus/'

split_num = 5000

python_type= 'python'

sql_type= 'sql'

#python
'''

'''


#sql
'''

'''


if __name__ == '__main__':
    # 抽取xml
    # extract_xml(data_folder, 'posts_2017.xml','python_tag_2017.xml','sql_tag_2017.xml','answer_2017.xml')
    # extract_xml(data_folder, 'posts_2018.xml',,'python_tag_2018.xml','sql_tag_2018.xml','answer_2018.xml')
    # extract_xml(data_folder, 'posts_2019.xml','python_tag_2019.xml','sql_tag_2019.xml','answer_2019.xml')
    # extract_xml(data_folder, 'posts_2020.xml',,'python_tag_2020.xml','sql_tag_2020.xml','answer_2020.xml')
    # extract_xml(data_folder, 'posts_2021.xml','python_tag_2021.xml','sql_tag_2021.xml','answer_2021.xml')
    # extract_xml(data_folder, 'posts_2022.xml','python_tag_2022.xml','sql_tag_2022.xml','answer_2022.xml')

    ###############################csqc######################################
    # 处理python
    process_xmldata(data_folder,python_type,split_num,save_folder)
    # 处理sql
    process_xmldata(data_folder,sql_type, split_num, save_folder)









