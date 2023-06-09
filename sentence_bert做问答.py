with open('1.txt' ,'w') as f:
    f.write("""
问:招待外部专家的费用标准？
答：邀请外部专家为公司提供劳务等产生的由公司负担的差旅费、餐费等费用，原则上按照不超过公司职级 P9 的标准执行。

问：对于已收到入职通知书但未入职的员工按照公司要求产生的费用是否可以报销。
答：对于已经接到入职通知的新员工，在接到入职通知后按照公司要求参与公司安排的各项活动所发生的差旅费，依据公司差旅费管理办法予以报销。

问：差旅费的定义。
答：差旅费是指工作人员临时到常驻地以外地区（北京除外）公务出差所发生的城市间交通费、住宿费、伙食费、市内交通费和外埠交通费等。
    
    
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
print('你的答案是',all_answer[dex][2:])