import os
import io
import json
import torch

from PIL import Image
from flask import Flask, jsonify, request, render_template
from flask_cors import CORS


app = Flask(__name__)
CORS(app)  # 解决跨域问题

# weights_path = "./MobileNetV2(flower).pth"
# class_json_path = "./class_indices.json"

# # select device
# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# print(device)
# # create model
# model = eval(num_classes=5)
# # load model weights
# model.load_state_dict(torch.load(weights_path, map_location=device))
# model.to(device)
# model.eval()

# # load class info
# json_file = open(class_json_path, 'rb')
# class_indict = json.load(json_file)


def get_prediction(image_bytes):
    try:
        tensor = ""
        outputs = torch.softmax(model.forward(tensor).squeeze(), dim=0)
        prediction = outputs.detach().cpu().numpy()
        template = "class:{:<15} probability:{:.3f}"
        index_pre = [(class_indict[str(index)], float(p)) for index, p in enumerate(prediction)]
        # sort probability
        index_pre.sort(key=lambda x: x[1], reverse=True)
        text = [template.format(k, v) for k, v in index_pre]
        return_info = {"result": text}
    except Exception as e:
        return_info = {"result": [str(e)]}
    return return_info


@app.route("/predict", methods=["POST"])
@torch.no_grad()
def predict():
    image = request.files["file"]
    img_bytes = image.read()
    info = get_prediction(image_bytes=img_bytes)
    return jsonify(info)









#!/usr/bin/env python
# -*- coding:utf-8 -*-
from collections import defaultdict
import re

__all__ = ['NaiveFilter', 'BSFilter', 'DFAFilter']
__author__ = 'observer'
__date__ = '2012.01.05'


class NaiveFilter():

    '''Filter Messages from keywords
    very simple filter implementation
    >>> f = NaiveFilter()
    >>> f.add("sexy")
    >>> f.filter("hello sexy baby")
    hello **** baby
    '''

    def __init__(self):
        self.keywords = set([])

    def parse(self, path):
        for keyword in open(path):
            self.keywords.add(keyword.strip().decode('utf-8').lower())

    def filter(self, message, repl="*"):
        message = unicode(message).lower()
        for kw in self.keywords:
            message = message.replace(kw, repl)
        return message


class BSFilter:

    '''Filter Messages from keywords
    Use Back Sorted Mapping to reduce replacement times
    >>> f = BSFilter()
    >>> f.add("sexy")
    >>> f.filter("hello sexy baby")
    hello **** baby
    '''

    def __init__(self):
        self.keywords = []
        self.kwsets = set([])
        self.bsdict = defaultdict(set)
        self.pat_en = re.compile(r'^[0-9a-zA-Z]+$')  # english phrase or not

    def add(self, keyword):
        if not isinstance(keyword, unicode):
            keyword = keyword.decode('utf-8')
        keyword = keyword.lower()
        if keyword not in self.kwsets:
            self.keywords.append(keyword)
            self.kwsets.add(keyword)
            index = len(self.keywords) - 1
            for word in keyword.split():
                if self.pat_en.search(word):
                    self.bsdict[word].add(index)
                else:
                    for char in word:
                        self.bsdict[char].add(index)

    def parse(self, path):
        with open(path, "r") as f:
            for keyword in f:
                self.add(keyword.strip())

    def filter(self, message, repl="*"):
        if not isinstance(message, unicode):
            message = message.decode('utf-8')
        message = message.lower()
        for word in message.split():
            if self.pat_en.search(word):
                for index in self.bsdict[word]:
                    message = message.replace(self.keywords[index], repl)
            else:
                for char in word:
                    for index in self.bsdict[char]:
                        message = message.replace(self.keywords[index], repl)
        return message


class DFAFilter():

    '''Filter Messages from keywords
    Use DFA to keep algorithm perform constantly
    >>> f = DFAFilter()
    >>> f.add("sexy")
    >>> f.filter("hello sexy baby")
    hello **** baby
    '''

    def __init__(self):
        self.keyword_chains = {}
        self.delimit = '\x00'  #这是一个不可见字符,看起来跟''一样,但是实际上不一样! 他作为结尾符很适合!

    def add(self, keyword):
        if not isinstance(keyword, str):
            keyword = keyword.decode('utf-8')
        keyword = keyword.lower()
        chars = keyword.strip()
        if not chars:
            return
        level = self.keyword_chains
        for i in range(len(chars)): #对字符串里面每一个字符建立trie树.
            if chars[i] in level: #如果当前这个汉子存在,那么就level进入下一层.
                level = level[chars[i]]
            else:
                if not isinstance(level, dict):#走到头了.说明已经存在这个level了.
                    break
                for j in range(i, len(chars)):#建立新的子字典.
                    level[chars[j]] = {}
                    last_level, last_char = level, chars[j]
                    level = level[chars[j]]
                last_level[last_char] = {self.delimit: 0} #然后写入结束符.
                break
        if i == len(chars) - 1: # 说明已经有过这个字符串,那么我们就写入结束符即可. 比如先add 啊啊啊啊, 再 add 啊, 那么这一样代码就会触发!
            level[self.delimit] = 0

    def parse(self, path):
        with open(path,encoding='utf-8') as f:
            for keyword in f:
                self.add(keyword.strip())

    def filter(self, message, repl="*"):
        if not isinstance(message, str):
            message = message.decode('utf-8')
        message = message.lower()
        ret = []
        start = 0
        while start < len(message):
            level = self.keyword_chains
            step_ins = 0
            for char in message[start:]:
                if char in level:
                    step_ins += 1
                    if self.delimit not in level[char]:
                        level = level[char]
                    else:
                        ret.append(repl * step_ins)
                        start += step_ins - 1
                        break
                else:
                    ret.append(message[start])
                    break
            else:
                ret.append(message[start])
            start += 1

        return ''.join(ret)




#==============这个函数,输入message,返回 跟trie树中匹配上的所有字符串的start end索引,组成的二维数组.
#==============这个函数,输入message,返回 跟trie树中匹配上的所有字符串的start end索引,组成的二维数组. 这里面end是按照python规范来!==============这个是最短匹配,从头到尾找子串,只要匹配成功就跳过这个成功的. 找后面匹配的部分!
    #==========现在我们把这个最短匹配当做默认情况,因为这个算法是最快的.一般也足够用了!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    def pipei_shortest(self, message, repl="*"):
        if not isinstance(message, str):
            message = message.decode('utf-8')
        out=[]
        out_debug=[]
        message = message.lower()
        ret = []
        start = 0
        while start < len(message):
            level = self.keyword_chains
            step_ins = 0
            for index,char in enumerate( message[start:]):
                if char in level:
                    step_ins += 1
                    if self.delimit not in level[char]:#无脑进入.
                        level = level[char]
                    else:#===============匹配成功了!#===========这个地方好像有问题, 如果字典里面有a 也有ab,那么运行到a就停了,不会继续找更长的!!!!!!!!!!!!!!!!
                        ret.append(repl * step_ins)
                        old_start=start
                        start += step_ins - 1
                        out.append([old_start,start+1])
                        out_debug.append(message[old_start:start+1])
                        break
                else:#如果char不存在,

                    break
            else: # for else: 上面的break都没触发,就走这个else. 说明一直进入到了最后一层.并且里面一直都没有结束符!!!!!说明当前位置字符串只是一个前缀,不能成为单词.所以不是我们要的.
                pass

            start += 1#=========这里也是可以直接跳过.
        print(out_debug)
        return out



#最长匹配, 尽量找跟字典中最长的匹配, 尽可能让找到的字符串最长!!!!!!!!!!!!!!!!!!!性能会比上面的低很多!
    def pipei_longest(self, message, repl="*"):
        if not isinstance(message, str):
            message = message.decode('utf-8')
        out=[]
        message = message.lower()
        ret = []
        start = 0

        while start < len(message):
            level = self.keyword_chains
            step_ins = 0#用来记录当前遍历到字典的第几层.
            start2 = None
            for index,char in enumerate( message[start:]):
                if char in level:
                    step_ins += 1
                    if self.delimit not in level[char]:#无脑进入.
                        level = level[char]
                    else:#===============匹配成功了!#===========这个地方好像有问题, 如果字典里面有a 也有ab,那么运行到a就停了,不会继续找更长的!!!!!!!!!!!!!!!!
                        level = level[char]
                        old_start=start
                        start2 =start+ step_ins - 1 #保证找到的不会重叠.


                else:#如果char不存在,
                    # ret.append(message[start])
                    break

            #=================遍历玩了当前字符为起始字符的全排列.
            if start2!=None:
                out.append([start, start2 + 1])
            if start2!=None:
                start=start2+1 #因为已经匹配最长了,直接跳过即可!!!!!!!!!!
            else:
                start += 1

        return out











#全匹配,也是最浪费性能的!!!!!!!!!!!!!!!!!!!!!!!!!!!!性能会比上面的低很多!
    def pipei_all(self, message, repl="*"):
        if not isinstance(message, str):
            message = message.decode('utf-8')
        out=[]
        message = message.lower()
        ret = []
        start = 0
        while start < len(message):
            level = self.keyword_chains
            step_ins = 0#用来记录当前遍历到字典的第几层.
            for index,char in enumerate( message[start:]):
                if char in level:
                    step_ins += 1
                    if self.delimit not in level[char]:#无脑进入.
                        level = level[char]
                    else:#===============匹配成功了!#===========这个地方好像有问题, 如果字典里面有a 也有ab,那么运行到a就停了,不会继续找更长的!!!!!!!!!!!!!!!!
                        level = level[char]
                        old_start=start
                        start2 =start+ step_ins - 1 #保证找到的不会重叠.
                        out.append([old_start,start2+1])

                else:#如果char不存在,
                    # ret.append(message[start])
                    break
            else: # for else: 上面的break都没触发,就走这个else. 说明一直进入到了最后一层.并且里面一直都没有结束符!!!!!说明当前位置字符串只是一个前缀,不能成为单词.所以不是我们要的.
                # ret.append(message[start])
                pass
            start += 1

        return out















def test_first_character():
    gfw = DFAFilter()
    gfw.add("1989年")
    assert gfw.filter("1989", "*") == "1989"




#==================================sb服务.




with open('1.txt' ,'w') as f:
    f.write("""
问：招待外部专家的费用标准？
答：邀请外部专家为公司提供劳务等产生的由公司负担的差旅费、餐费等费用，原则上按照不超过公司职级 P9 的标准执行。

问：对于已收到入职通知书但未入职的员工按照公司要求产生的费用是否可以报销。
答：对于已经接到入职通知的新员工，在接到入职通知后按照公司要求参与公司安排的各项活动所发生的差旅费，依据公司差旅费管理办法予以报销。

问：差旅费的定义。
答：差旅费是指工作人员临时到常驻地以外地区（北京除外）公务出差所发生的城市间交通费、住宿费、伙食费、市内交通费和外埠交通费等。

问：出差期间产生的招待费、邮寄费、复印费等费用是否可以报销？
答：出差期间发生的招待费、复印费、邮寄费等费用，均在差旅费之外，按相 
应的费用管理规定报销。

问：出差过程中发生一些特殊的事项，公司是否有特殊报销标准。
答：如遇特殊事项，相关差旅费报销事宜可按公司领导批准的方式和标准执行。

问：出差的交通工具标准？
答：公司领导，火车软席（软座、软卧）、高铁/动车一等座、动车一等卧（动卧）；全列软席列车一等软座。轮船二等舱，飞机经济舱，其它凭据报销。P8或P9，火车硬席（硬座、硬卧），高铁/动车一等座，动车二等卧（动卧），全列软席列车二等软座。轮船二等舱，飞机经济舱，其它凭据报销。P7及P7以下，火车硬席（硬座、硬卧），动高铁/动车二等座，二等卧（动卧），全列软席列车二等软座。轮船二等舱，飞机经济舱，其它凭据报销。

问：出差过程中，使用交通工具是否可以购买交通意外保险。
答：乘坐飞机、火车、轮船等交通工具的，每人次可以购买交通意外保险一份。

问：出差过程中，住宿标准是什么。
答：公司领导，一类地区800，二类地区750，三类地区700，P8-P9，一类地区650，二类地区600，三类地区550，P7及以下，一类地区500，二类地区450，三类地区400。

问：出差一类地区都包含哪些？
答：一类地区包括北京、上海、广州、深圳，大连（7-9 月）、哈尔滨（7-9 月）、青岛（7-9 月）、海口（11-2 月）、拉萨（6-9 月）、西宁（6-9 月）、张家口（7-9 月、11-3 月）、秦皇岛（7-8 月）、海拉尔、满洲里、阿尔山（7-9 月）、 额济纳旗（9-10 月）、洛阳（4-5 月上旬）、三亚（10-4 月）14 个城市在旅游旺季可按 一类地区标准报销。

问：出差二类地区都包含哪些？
答：二类地区包括各省会城市（广州除外）、 天津、重庆、大连、厦门、青岛。

问：出差三类地区都包含哪些？
答：除一类地区、二类地区以外的其它区域属于三类地区。

问：无锡属于几类地区？
答：无锡属于三类地区。

问：出差住酒店报销需要有什么凭证？
答：住宿费报销时应取得增值税专用发票及盖有入住宾馆印章的费用明细单（盖章的原件或复印件皆可）。宾馆费用明细单如无印章则需要提供说明材料,由分管领导审批；未取得增值税专用发票的，如酒店可提供说明则尽量酒店提供，否则报销人出具说明材料，由分管领导签批

问：增值税发票报销抬头。
答：抬头：未来电视有限公司，税号：911 2011 6586 419887T，地址：天津市空港经济区环河北路80号商务园东区E5，开户行：中国银行天津融合广场支行，账号：2687 6632 7455

问：出差餐补报销标准。
答：国内出差餐补为每人每天 150 元标准，随同差旅费报销款项一并发放。国内出差津贴（餐补）以出差申请表中注明的实际住宿过夜天数为准进 行计算，当天往返的情况可享受一天的出差餐补。返程车票或飞机票的出发时间晚于 当天 18:00 的，可报销不超过 50 元的返程日当天晚餐费用，报销时需提供当地餐费发 票。若员工报销业务招待费或加班餐费，则相应扣除当天餐补，每招待一餐或报销加班一餐，扣除当天餐补的 1/3。

问：对于多人出差招待餐时报销注意事项有哪些？
答：对于公司内多人一起出差一起参加招待用餐的，在各自报销差旅费时应按照招待用餐次数计算差旅补助扣除金额，未按规定填报的一经风控查处按公司规定处理。

问：出差期间市内交通和外埠交通费可以报销吗？
答：市内交通费是指工作人员因公出差期间发生的往返机场、车站的天津市 内交通费用。外埠交通费是指工作人员因公出差期间在出差地区发生的交通费用，不 包括城市间交通费用。市内交通费和外埠交通费应注明事由，并凭相关票据据实报销。 发票连号、乘坐同一车辆或使用定额发票的，需提供支付记录

问：出差使用企业滴滴有什么注意事项？
答：除接送重要客户外，不得使用滴滴 专车，只能通过滴滴快车出行。对于无特别事由而使用滴滴专车的情形将处以三倍滴 滴出行费用的罚款。使用“滴滴企业版 APP”叫车时，请务必在“补充说明（选填）” 中注明具体事由，相关“用车备注-补充说明”应按照“拜访具体目的地+拜访具体事 由”的方式填写清楚，而非单纯简单填写“商务出行、出差、拜访客户”等简略方式， 未按标准填写不予报销。

问：国外出差注意事项？
答：出国人员在出国前应提交书面计划书，在 OA 系统内填写“出国（境）差旅审批”（附件一），经部门总监、分管副总经理审核，报总经理批准后方可办理出国手续。出国费用专项预算包括以下内容：国际旅费、国外城市间交通费、住宿 费、伙 食费、公杂费和其他费用（主要是指出国签证费用、必需的保险费用、防疫费用、国际会议注册 费用等）。

问：国外出差住宿费用标准是什么？
答：出国人员应当严格按照不超过财政部、外交部《各国家和地区住宿费、伙食费、公 杂费开支标准表》(财行【2013】516 号)和《关于调整因公临时出国住宿费标准等有关 事项的通知》（财行【2017】434 号）规定的标准(具体标准详见附件二)安排住宿。 参加国际会议等的出国人员，原则上应当按照不超过附件二的住宿费标准执行。如 对方组织单位指定或推荐酒店，应当严格把关，通过询价方式从紧安排，超出费用标准的，需说明情况事先报经主管副总经理和总经理批准。经批准，住宿费可据实报销。

问：国外出差伙食费和公杂费规则是什么？
答：伙食费、公杂费，按照财政部、外交部《各国家和地区住宿费、伙食费、公杂费开 支标准表》（财行【2013】516 号）和《关于调整因公临时出国住宿费标准等有关事项 的通知》（财行【2017】434 号）规定的标准(具体标准详见附件二)发放个人包干使用。 包干天数按离、抵我国国境之日计算。不宜个人包干的出访团组，可由团组统一使用。 如果出国团组申请了租车费或外方为我以出访团组提供交通接待的，出国人员可按 标准的 40%领取公杂费。

问：国外出差报销凭证的注意事项有哪些？
答：出国人员回国报销费用时，各种报销凭证需用中文注明开支内容、日期、 
数量、金额等，并由经办人签字。

问：出差前流程由哪些？
答：出差人员在出差前填写“出差审批单”，经部门总监核准后报分管副总经理审批。出差人员将“出差审批单”送到行政部，由行政部统一预定机票。已经预定的机票，确因工作需要，需转签、退票或退订，需经分管领导根据工作需要进行审批确认后方可向行政申请办理退、改、签。对于回程日期不确定的情况，回程机票事宜由出差人员直接联系行政部按规定订票即可，不用再行审批。火车票由出差人员自行预定。

问：出差的机票怎么定？
答：出差人员将“出差审批单”送到行政部，由行政部统一预定机票。已经预定的机票，确因工作需要，需转签、退票或退订，需经分管领导根据工作需要进行审批确认后方可向行政申请办理退、改、签。对于回程日期不确定的情况，回程机票事宜由出差人员直接联系行政部按规定订票即可，不用再行审批。

问：出差报销需要提供什么材料？
答：报销差旅费时需填写《差旅费报销单》，随《出差审批单》、酒店打印的住宿水单和有关票据一并交财务管理中心审核。时间要求按照《日常费用报销管理办法》第四条规定执行：
1.城市间交通费按乘坐交通工具的类别凭票据报销。交通意外保险费、经批准发生 
的签转或退票费凭票据报销。 
2.自 2016 年 7 月 1 日开始，住宿费在标准限额之内凭“增值税专用发票”据实报 
销，仅报销住宿产生的房费，不报销额外产生的费用，如额外收费电视节目、迷你吧 
消费等。若未取得“增值税专用发票”，报销人需提供相关说明。 
3.出差期间发生的业务招待费报销原则上不接受超市、商场购物发票，因特殊情 
况在超市、商场购买食品的需提供购物小票和经主管副总审批的说明。 
4.市内交通费和外埠交通费凭相关票据据实报销，其中外埠交通费应当注明详细 
事由以及出发地和目的地信息。 
5.出差期间航班的商务舱和头等舱不予报销，发生的超重的行李（由非公司公物 
所导致的超重）费用不予报销，其他与工作相关的费用（如机场打包费等）凭相关票 
据据实报销。其中，票据未注明费用内容的，需提供书面说明。在现行人工订购出差 
的情况下，机票报销单据由行政部提交，需包括行程单、经办人签字的行程单明细表、 登机牌复印件、经办人签字的发票等单据；行政部须在报销单提交 OA 系统办理之日起 一个月内完成登机牌复印件的归集并作为附件粘贴至报销单后；具体报销办法由财务 管理中心在日常工作中予以明确。后期使用商旅平台后，具体报销办法由财务管理中 心在日常工作予以明确。 
6.未按规定开支差旅费的，超支部分由个人自理。

问：出差报销的截止时间要求？
答：出差人员应于出差结束后 2 个月内，填写《差旅费报销单》，并按规定流程申请报销。超出报销时限的，应附情况说明，报请分管领导签批。

问：正向激励的奖金在什么时间有具体体现？
答：正向激励资金应在作出相应激励决定的次月工资中发放。

问：负向激励的奖金在什么时间有具体体现？
答：负向激励资金应在作出相应激励决定的次月工资中执行。

问：正向激励分类以及奖金。
答：四级正激励公司内邮件通报；100-300 元，三级正激励公司内邮件通报；300-1000 元，二级正激励公司内邮件通报； 1000-2000 元，一级正激励公司内邮件通报； 2000-10000 元，公司大奖在公司内部邮件通报：提升一个职级，加薪 5-50%，可视重要程度进行调整。

问：创造收益或节约开支是否有额外奖励？
答：给公司创造收益或者节约开支的，除获得上述正激励外，可按照一定比例 
获得创造收益或节约开支的提成，具体按公司提成制度执行。

问：负向激励分类以及惩罚标准是什么。
答：四级负激励公司内邮件通报；负向激励系数为 1‰-2%，三级负激励公司内邮件通报；负向激励系数为 2‰-10%，二级负激励公司内邮件通报；负向激励系数10‰-20%，一级负激励公司内邮件通报；负向激励系数为 20%-100%；降低一个职级， 降薪幅度 5-30%；具体情况可视严重程度进行调整。终极负激励解除劳动合同。
实发业绩工资=业绩工资标准×绩效考核系数×（1-负向激励系数）。
    
    
    """.replace(':',"："))

with open('1.txt'  ) as f:

    tmp=f.read()
print(1)
tmp=tmp.split('\n\n')
print(1)
all_question=[i[:i.find('答：')] for i in tmp]
all_answer=[i[i.find('答：'):] for i in tmp]

print(1)

# Prompt-based MLM fine-tuning
from transformers import BertForMaskedLM, BertTokenizer
import torch










# Prompt-based Sentence Similarity
# To extract sentence representations.
from transformers import BertForMaskedLM, BertTokenizer
import torch

# Loading models
tokenizer=BertTokenizer.from_pretrained("IDEA-CCNL/Erlangshen-TCBert-330M-Sentence-Embedding-Chinese")
model=BertForMaskedLM.from_pretrained("IDEA-CCNL/Erlangshen-TCBert-330M-Sentence-Embedding-Chinese")

# Cosine similarity function
cos = torch.nn.CosineSimilarity(dim=0, eps=1e-8)
all_vec=[]
for dex,i in enumerate(all_question):
    
    with torch.no_grad():

        # To extract sentence representations for training data
        t = tokenizer(i, return_tensors="pt")
        print(t)
        training_outputs = model(**t, output_hidden_states=True)
        training_representation = torch.mean(training_outputs.hidden_states[-1].squeeze(), dim=0)
        print(training_representation.shape)
      

    all_vec.append(training_representation)
# all_vec=torch.vstack(all_vec)
# print(all_vec.shape)
































@app.route("/", methods=["GET", "POST"])
def root():
    
    with torch.no_grad():

            # To extract sentence representations for training data
            t = tokenizer('招待外部人员的费用', return_tensors="pt")
            training_outputs = model(**t, output_hidden_states=True)
            t2 = torch.mean(training_outputs.hidden_states[-1].squeeze(), dim=0)

        
    # Calculate similarity scores


    mini=-float('inf')
    dex=-1
    for i in range(len(all_vec)):
        t=cos(all_vec[i], t2)
        print('当前相似度',t)
        if t>mini:
            mini=t
            dex=i

    print('最相似的是',dex,mini)
    #==========大于0.6就返回答案.
    ans=all_answer[dex][2:]
    print('你的答案是',all_answer[dex][2:])
    return jsonify(ans)

    
if __name__ == "__main__":
    # gfw = NaiveFilter()
    # gfw = BSFilter()
    import time

    

    app.run(host="0.0.0.0", port=7862)