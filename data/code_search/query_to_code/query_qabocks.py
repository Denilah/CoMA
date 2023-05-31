
import re
import json
import pickle
import itertools
from tqdm import tqdm

#网页转义
import html
#解析数据
from bs4 import BeautifulSoup

# 替换标签
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
    return ptaglist


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

    p_pcode1 = re.compile(r'(<pre><code>[\s\S]*</code></pre>)')
    p_pcode2 = re.compile(r'(<pre class=[\s\S]*><code>[\s\S]*</code></pre>)')

    text_blocks = []

    code_blocks = []

    # 遍历每个block
    for i in range(0, len(all_blocklis)):

        # print('########################Block %s#############################\n' % str(i), all_blocklis[i])

        if i == 0:
            n_f = match_str_ptag(p_context, match_str_block(p_forward, all_blocklis[i])[0]) if p_context.findall(
                match_str_block(p_forward, all_blocklis[i])[0]) != [] else '-10000'
            text_blocks.append(n_f)
            # print('--------forward nl description-------\n', n_f)
        else:
            n_f = match_str_ptag(p_context, match_str_block(p_backward, all_blocklis[i - 1])[0]) if p_context.findall(
                match_str_block(p_backward, all_blocklis[i-1])[0]) != [] else '-10000'
            text_blocks.append(n_f)
            # print('--------forward nl description-------\n', n_f)

        code = match_tag_code(p_hcode, p_pcode, p_pcode1, p_pcode2, match_str_block(p_code, all_blocklis[i])[0])
        # 对网页标签转义
        c = html.unescape(code)
        code_blocks.append(c)
        # print('-------------context code----------\n', c)

        if i == len(all_blocklis) - 1:
            n_b = match_str_ptag(p_context, match_str_block(p_backward, all_blocklis[i])[0]) if p_context.findall(
                match_str_block(p_backward, all_blocklis[i])[0]) != [] else '-10000'
            text_blocks.append(n_b)
            # print('-----------backward nl description--------\n', n_b)
        else:
            n_b = match_str_ptag(p_context, match_str_block(p_backward, all_blocklis[i])[0]) if p_context.findall(
                match_str_block(p_backward, all_blocklis[i])[0]) != [] else '-10000'
            text_blocks.append(n_b)
            # print('-----------backward nl description--------\n', n_b)

        # print('########################Block %s#############################\n' % str(i))

    text_blocks = [i for i in text_blocks if i != '-10000']
    # 多个列表变一个列表
    text_blocks = list(set(itertools.chain.from_iterable(text_blocks)))

    block_list = text_blocks + code_blocks

    return block_list


def main(lang_type):

    # [(qid, aid),  query,  qhtmlstr, ahtmlstr, soical_terms]
    posts_terms = pickle.load(open('./corpus/%s_corpus_qid_allterms.pickle'%lang_type, "rb"))
    # python 862710  sql 668544
    print('%s的query语料共计%s条'%(lang_type, len(posts_terms)))

    qablock_dict = {}

    for  c  in tqdm(posts_terms):
        # qid
        qid  = c[0][0]
        # query
        query = c[1]
        # <html><body>
        qhtmlSoup = BeautifulSoup(c[2], 'lxml')
        # <html><body>
        ahtmlSoup = BeautifulSoup(c[3], 'lxml')

        # 抽取代码和上下文 直接用soup（不能转了str)
        qblock_list = parse_tag_block(qhtmlSoup)

        if qblock_list != []:
            question_blocks = qblock_list
        else:
            question_blocks = [str(qhtmlSoup)]

        # 抽取代码和上下文 直接用soup（不能转了str)
        ablock_list = parse_tag_block(ahtmlSoup)

        if ablock_list != []:
            answer_blocks = ablock_list
        else:
            answer_blocks = [str(ahtmlSoup)]

        # 加入数据
        qablock_dict[qid] = (query, answer_blocks, question_blocks)

    with open('./corpus/%s_qid_query_to_qablocks.json' % lang_type, 'w') as f:
        json.dump(qablock_dict, f)


python_type = 'python'
sql_type = 'sql'

if __name__ == '__main__':
    main(python_type)
    #main(sql_type)