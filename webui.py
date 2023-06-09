import gradio as gr
import os
import shutil

from chains.local_doc_qa import LocalDocQA
from configs.model_config import *
import nltk
from models.base import (BaseAnswer,
                         AnswerResult)
import models.shared as shared
from models.loader.args import parser
from models.loader import LoaderCheckPoint

nltk.data.path = [NLTK_DATA_PATH] + nltk.data.path
# 2023-06-06,23点10  ,添加本地知识库的智齿.
print('打印路径变量.!!!!!!!!!')
print(VS_ROOT_PATH,VS_ROOT_PATH)
def get_vs_list(): #vector_store 叫vs #读取本地知识库, 然后把列表返回.
    lst_default = ["新建知识库"]
    if not os.path.exists(VS_ROOT_PATH):
        return lst_default
    lst = os.listdir(VS_ROOT_PATH)
    if not lst:
        return lst_default
    lst.sort()
    return lst_default + lst


embedding_model_dict_list = list(embedding_model_dict.keys()) # 3个向量化模型.

llm_model_dict_list = list(llm_model_dict.keys()) #所有的llm模型.

local_doc_qa = LocalDocQA()

flag_csv_logger = gr.CSVLogger()


def get_answer(query, vs_path, history, mode, score_threshold=VECTOR_SEARCH_SCORE_THRESHOLD,
               vector_search_top_k=VECTOR_SEARCH_TOP_K, chunk_conent: bool = True,
               chunk_size=CHUNK_SIZE, streaming: bool = STREAMING):
    print('打印get_answer的入参',query, vs_path, history, mode, )
    print('打印',os.path.exists(vs_path))
    print('打印',"index.faiss" in os.listdir(
            vs_path))
    if mode == "Bing搜索问答":
        for resp, history in local_doc_qa.get_search_result_based_answer(
                query=query, chat_history=history, streaming=streaming):
            source = "\n\n"
            source += "".join(
                [
                    f"""<details> <summary>出处 [{i + 1}] <a href="{doc.metadata["source"]}" target="_blank">{doc.metadata["source"]}</a> </summary>\n"""
                    f"""{doc.page_content}\n"""
                    f"""</details>"""
                    for i, doc in
                    enumerate(resp["source_documents"])])
            history[-1][-1] += source
            yield history, ""
    elif mode == "知识库问答" and vs_path is not None and os.path.exists(vs_path) and "index.faiss" in os.listdir(
            vs_path):
        for resp, history in local_doc_qa.get_knowledge_based_answer(
                query=query, vs_path=vs_path, chat_history=history, streaming=streaming):
            source = "\n\n"
            source += "".join(
                [f"""<details> <summary>出处 [{i + 1}] {os.path.split(doc.metadata["source"])[-1]}</summary>\n"""
                 f"""{doc.page_content}\n"""
                 f"""</details>"""
                 for i, doc in
                 enumerate(resp["source_documents"])])
            history[-1][-1] += source
            yield history, ""
    elif mode == "知识库测试":
        if os.path.exists(vs_path):
            resp, prompt = local_doc_qa.get_knowledge_based_conent_test(query=query, vs_path=vs_path,
                                                                        score_threshold=score_threshold,
                                                                        vector_search_top_k=vector_search_top_k,
                                                                        chunk_conent=chunk_conent,
                                                                        chunk_size=chunk_size)
            if not resp["source_documents"]:
                yield history + [[query,
                                  "根据您的设定，没有匹配到任何内容，请确认您设置的知识相关度 Score 阈值是否过小或其他参数是否正确。"]], ""
            else:
                source = "\n".join(
                    [
                        f"""<details open> <summary>【知识相关度 Score】：{doc.metadata["score"]} - 【出处{i + 1}】：  {os.path.split(doc.metadata["source"])[-1]} </summary>\n"""
                        f"""{doc.page_content}\n"""
                        f"""</details>"""
                        for i, doc in
                        enumerate(resp["source_documents"])])
                history.append([query, "以下内容为知识库中满足设置条件的匹配结果：\n\n" + source])
                yield history, ""
        else:
            yield history + [[query,
                              "请选择知识库后进行测试，当前未选择知识库。"]], ""
    else:
        for answer_result in local_doc_qa.llm.generatorAnswer(prompt=query, history=history[:-1],
                                                              streaming=streaming):
            resp = answer_result.llm_output["answer"]
            history = answer_result.history
            history[-1][-1] = resp
            yield history, ""
    logger.info(f"flagging: username={FLAG_USER_NAME},query={query},vs_path={vs_path},mode={mode},history={history}")
    flag_csv_logger.flag([query, vs_path, history, mode], username=FLAG_USER_NAME)


def init_model():
    args = parser.parse_args()

    args_dict = vars(args)
    shared.loaderCheckPoint = LoaderCheckPoint(args_dict)
    llm_model_ins = shared.loaderLLM()
    llm_model_ins.set_history_len(LLM_HISTORY_LEN)
    try:
        local_doc_qa.init_cfg(llm_model=llm_model_ins)
        generator = local_doc_qa.llm.generatorAnswer("你好")
        for answer_result in generator:
            print(answer_result.llm_output)
        reply = """模型已成功加载，可以开始对话，或从右侧选择模式后开始对话"""
        logger.info(reply)
        return reply
    except Exception as e:
        logger.error(e)
        reply = """模型未成功加载，请到页面左上角"模型配置"选项卡中重新选择后点击"加载模型"按钮"""
        if str(e) == "Unknown platform: darwin":
            logger.info("该报错可能因为您使用的是 macOS 操作系统，需先下载模型至本地后执行 Web UI，具体方法请参考项目 README 中本地部署方法及常见问题："
                        " https://github.com/imClumsyPanda/langchain-ChatGLM")
        else:
            logger.info(reply)
        return reply


def reinit_model(llm_model, embedding_model, llm_history_len, no_remote_model, use_ptuning_v2, use_lora, top_k,
                 history):
    try:
        llm_model_ins = shared.loaderLLM(llm_model, no_remote_model, use_ptuning_v2)
        llm_model_ins.history_len = llm_history_len
        local_doc_qa.init_cfg(llm_model=llm_model_ins,
                              embedding_model=embedding_model,
                              top_k=top_k)
        model_status = """模型已成功重新加载，可以开始对话，或从右侧选择模式后开始对话"""
        logger.info(model_status)
    except Exception as e:
        logger.error(e)
        model_status = """模型未成功重新加载，请到页面左上角"模型配置"选项卡中重新选择后点击"加载模型"按钮"""
        logger.info(model_status)
    return history + [[None, model_status]]


def get_vector_store(vs_id, files, sentence_size, history, one_conent, one_content_segmentation):
    vs_path = os.path.join(VS_ROOT_PATH, vs_id)
    filelist = []
    if local_doc_qa.llm and local_doc_qa.embeddings:
        if isinstance(files, list):
            for file in files:
                filename = os.path.split(file.name)[-1]
                shutil.move(file.name, os.path.join(UPLOAD_ROOT_PATH, vs_id, filename))
                filelist.append(os.path.join(UPLOAD_ROOT_PATH, vs_id, filename))
            vs_path, loaded_files = local_doc_qa.init_knowledge_vector_store(filelist, vs_path, sentence_size)
        else:
            vs_path, loaded_files = local_doc_qa.one_knowledge_add(vs_path, files, one_conent, one_content_segmentation,
                                                                   sentence_size)
        if len(loaded_files):
            file_status = f"已添加 {'、'.join([os.path.split(i)[-1] for i in loaded_files if i])} 内容至知识库，并已加载知识库，请开始提问"
        else:
            file_status = "文件未成功加载，请重新上传文件"
    else:
        file_status = "模型未完成加载，请先在加载模型后再导入文件"
        vs_path = None
    logger.info(file_status)
    return vs_path, None, history + [[None, file_status]]

#geshiahua shangchuan
def get_vector_store2(vs_id, files, sentence_size, history, one_conent, one_content_segmentation):
    vs_path = os.path.join(VS_ROOT_PATH, vs_id)
    filelist = []
    # if local_doc_qa.llm and local_doc_qa.embeddings:
    if 1:
        if isinstance(files, list):
            for file in files:
                filename = os.path.split(file.name)[-1]
                shutil.move(file.name, os.path.join(UPLOAD_ROOT_PATH, vs_id, filename))
                filelist.append(os.path.join(UPLOAD_ROOT_PATH, vs_id, filename))
            vs_path, loaded_files = local_doc_qa.init_knowledge_vector_store(filelist, vs_path, sentence_size)
        else: #============只添加一条知识.
            vs_path, loaded_files = local_doc_qa.one_knowledge_add(vs_path, files, one_conent, one_content_segmentation,
                                                                   sentence_size)
        if len(loaded_files):
            file_status = f"已添加 {'、'.join([os.path.split(i)[-1] for i in loaded_files if i])} 内容至知识库，并已加载知识库，请开始提问"
        else:
            file_status = "文件未成功加载，请重新上传文件"
    else:
        file_status = "模型未完成加载，请先在加载模型后再导入文件"
        vs_path = None
    logger.info(file_status)
    return vs_path, None, history + [[None, file_status]]




#========根据选项卡, 改变页面.
def change_vs_name_input(vs_id, history):
    if vs_id == "新建知识库": #如果新建知识库,
        return gr.update(visible=True), gr.update(visible=True), gr.update(visible=False), None, history
    else:
        vs_path = os.path.join(VS_ROOT_PATH, vs_id)
        if "index.faiss" in os.listdir(vs_path): #如果里面有faiss文件,说明知识库里面已经做好了向量化.
            file_status = f"已加载知识库{vs_id}，请开始提问"
        else: # 否则提示失败.
            file_status = f"已选择知识库{vs_id}，当前知识库中未上传文件，请先上传文件后，再开始提问"
        return gr.update(visible=False), gr.update(visible=False), gr.update(visible=True), \
               vs_path, history + [[None, file_status]]










knowledge_base_test_mode_info = ("【注意】\n\n"
                                 "1. 您已进入知识库测试模式，您输入的任何对话内容都将用于进行知识库查询，"
                                 "并仅输出知识库匹配出的内容及相似度分值和及输入的文本源路径，查询的内容并不会进入模型查询。\n\n"
                                 "2. 知识相关度 Score 经测试，建议设置为 500 或更低，具体设置情况请结合实际使用调整。"
                                 """3. 使用"添加单条数据"添加文本至知识库时，内容如未分段，则内容越多越会稀释各查询内容与之关联的score阈值。\n\n"""
                                 "4. 单条内容长度建议设置在100-150左右。\n\n"
                                 "5. 本界面用于知识入库及知识匹配相关参数设定，但当前版本中，"
                                 "本界面中修改的参数并不会直接修改对话界面中参数，仍需前往`configs/model_config.py`修改后生效。"
                                 "相关参数将在后续版本中支持本界面直接修改。")

# gr.update() 里面写一个语句, 那么触发之后就运行这个语句.
def change_mode(mode, history):
    if mode == "知识库问答":
        return gr.update(visible=True), gr.update(visible=True), history
        # + [[None, "【注意】：您已进入知识库问答模式，您输入的任何查询都将进行知识库查询，然后会自动整理知识库关联内容进入模型查询！！！"]]
    elif mode == "知识库测试":
        return gr.update(visible=True), gr.update(visible=True), [[None,
                                                                   knowledge_base_test_mode_info]]
    else:
        return gr.update(visible=False), gr.update(visible=False), history


def change_chunk_conent(mode, label_conent, history):
    conent = ""
    if "chunk_conent" in label_conent:
        conent = "搜索结果上下文关联"
    elif "one_content_segmentation" in label_conent:  # 这里没用上，可以先留着
        conent = "内容分段入库"

    if mode:
        return gr.update(visible=True), history + [[None, f"【已开启{conent}】"]]
    else:
        return gr.update(visible=False), history + [[None, f"【已关闭{conent}】"]]


def add_vs_name(vs_name, chatbot):
    if vs_name in get_vs_list():
        vs_status = "与已有知识库名称冲突，请重新选择其他名称后提交"
        chatbot = chatbot + [[None, vs_status]] #添加一句回答.然后下一句,更新chatbox内容.
        return gr.update(visible=True), gr.update(visible=True), gr.update(visible=True), gr.update(
            visible=False), chatbot
    else:
        # 新建上传文件存储路径
        if not os.path.exists(os.path.join(UPLOAD_ROOT_PATH, vs_name)):
            os.makedirs(os.path.join(UPLOAD_ROOT_PATH, vs_name))
        # 新建向量库存储路径
        if not os.path.exists(os.path.join(VS_ROOT_PATH, vs_name)):
            os.makedirs(os.path.join(VS_ROOT_PATH, vs_name))
        vs_status = f"""已新增知识库"{vs_name}",将在上传文件并载入成功后进行存储。请在开始对话前，先完成文件上传。 """
        chatbot = chatbot + [[None, vs_status]]
        return gr.update(visible=True, choices=get_vs_list(), value=vs_name), gr.update(
            visible=False), gr.update(visible=False), gr.update(visible=True), chatbot














# 自动化加载固定文件间中文件
def reinit_vector_store(vs_id, history):
    try:
        shutil.rmtree(VS_ROOT_PATH)
        vs_path = os.path.join(VS_ROOT_PATH, vs_id)
        sentence_size = gr.Number(value=SENTENCE_SIZE, precision=0,
                                  label="文本入库分句长度限制",
                                  interactive=True, visible=True)
        vs_path, loaded_files = local_doc_qa.init_knowledge_vector_store(UPLOAD_ROOT_PATH, vs_path, sentence_size)
        model_status = """知识库构建成功"""
    except Exception as e:
        logger.error(e)
        model_status = """知识库构建未成功"""
        logger.info(model_status)
    return history + [[None, model_status]]


def refresh_vs_list():
    return gr.update(choices=get_vs_list()), gr.update(choices=get_vs_list())


block_css = """.importantButton {
    background: linear-gradient(45deg, #7e0570,#5d1c99, #6e00ff) !important;
    border: none !important;
}
.importantButton:hover {
    background: linear-gradient(45deg, #ff00e0,#8500ff, #6e00ff) !important;
    border: none !important;
}"""







#===============================页面设置代码.


webui_title = """
# langchain-ChatGLM 

"""
default_vs = get_vs_list()[0] if len(get_vs_list()) > 1 else "为空"
init_message = f"""
"""

# 初始化消息
try:
    model_status = init_model()
except:
        model_status=''
model_status=''
#设置主题样式.
default_theme_args = dict( 
    font=["Source Sans Pro", 'ui-sans-serif', 'system-ui', 'sans-serif'],
    font_mono=['IBM Plex Mono', 'ui-monospace', 'Consolas', 'monospace'],
)




#界面编写.
with gr.Blocks(css=block_css, theme=gr.themes.Default(**default_theme_args)) as demo:


    # gr.State是 session 状态里面的变量. 他会在每次刷新浏览器的时候重新计算.

    vs_path, file_status, model_status = gr.State(
        os.path.join(VS_ROOT_PATH, get_vs_list()[0]) if len(get_vs_list()) > 1 else ""), gr.State(""), gr.State(
        model_status)





    gr.Markdown(webui_title)
    #=========整体是3个tab页面.
    #2023-06-07,12点44 llm对话删了. 
    with gr.Tab("对话"):
        with gr.Row():
            with gr.Column(scale=10):
                chatbot = gr.Chatbot([[None, init_message], [None, model_status.value]],
                                     elem_id="chat-box",
                                     show_label=False).style(height=750)
                query = gr.Textbox(show_label=False,
                                   placeholder="请输入提问内容，按回车进行提交").style(container=False)
            with gr.Column(scale=5):
                mode = gr.Radio(["LLM 对话", "知识库问答", "Bing搜索问答"],
                                label="请选择使用模式",
                                value="知识库问答", )
                #=======设置2个手风琴菜单.
                knowledge_set = gr.Accordion("知识库设定", visible=True) # =======这个地方应该设置为开.
                vs_setting = gr.Accordion("配置知识库")
                mode.change(fn=change_mode,
                            inputs=[mode, chatbot],
                            outputs=[vs_setting, knowledge_set, chatbot])

                with knowledge_set:
                    score_threshold = gr.Number(value=VECTOR_SEARCH_SCORE_THRESHOLD,
                                                label="知识相关度 Score 阈值，分值越低匹配度越高",
                                                precision=0,
                                                interactive=True)
                    vector_search_top_k = gr.Number(value=VECTOR_SEARCH_TOP_K, precision=0,
                                                    label="获取知识库内容条数", interactive=True)
                    chunk_conent = gr.Checkbox(value=False,
                                               label="是否启用上下文关联",
                                               interactive=True)
                    chunk_sizes = gr.Number(value=CHUNK_SIZE, precision=0,
                                            label="匹配单段内容的连接上下文后最大长度",
                                            interactive=True, visible=False)
                    chunk_conent.change(fn=change_chunk_conent,
                                        inputs=[chunk_conent, gr.Textbox(value="chunk_conent", visible=False), chatbot],
                                        outputs=[chunk_sizes, chatbot])




                with vs_setting:
                    #===先写完全部的按钮,
                    vs_refresh = gr.Button("更新已有知识库选项")
                    select_vs = gr.Dropdown(get_vs_list(),
                                            label="请选择要加载的知识库",
                                            interactive=True,
                                            value=get_vs_list()[0] if len(get_vs_list()) > 0 else None
                                            ) # value是dropdown下拉框的默认值.
                    vs_name = gr.Textbox(label="请输入新建知识库名称，当前知识库命名暂不支持中文",
                                         lines=1,
                                         interactive=True,
                                         visible=True)
                    vs_add = gr.Button(value="添加至知识库选项", visible=True)
                    file2vs = gr.Column(visible=False)
                    with file2vs:
                        # load_vs = gr.Button("加载知识库")
                        gr.Markdown("向知识库中添加文件")
                        sentence_size = gr.Number(value=SENTENCE_SIZE, precision=0,
                                                  label="文本入库分句长度限制",
                                                  interactive=True, visible=True)
                        with gr.Tab("上传文件"):
                            files = gr.File(label="添加文件",
                                            file_types=['.txt', '.md', '.docx', '.pdf', '.png', '.jpg'],
                                            file_count="multiple",
                                            show_label=False)
                            load_file_button = gr.Button("上传文件并加载知识库")
                        with gr.Tab("上传格式化文件"):
                            files = gr.File(label="添加格式化文件,每条用/r/n切割",
                                            file_types=['.txt', '.md', '.docx', '.pdf', '.png', '.jpg'],
                                            file_count="multiple",
                                            show_label=False)
                            load_fileformat_button = gr.Button("上传格式化文件并加载知识库")






                        with gr.Tab("上传文件夹"):
                            folder_files = gr.File(label="添加文件",
                                                   file_count="directory",
                                                   show_label=False)
                            load_folder_button = gr.Button("上传文件夹并加载知识库")




#再写所有按钮的click函数.
                    vs_refresh.click(fn=refresh_vs_list,
                                     inputs=[],
                                     outputs=select_vs)
#添加一个知识库触发的函数.
                    vs_add.click(fn=add_vs_name,
                                 inputs=[vs_name, chatbot],
                                 outputs=[select_vs, vs_name, vs_add, file2vs, chatbot])
                    select_vs.change(fn=change_vs_name_input,
                                     inputs=[select_vs, chatbot],
                                     outputs=[vs_name, vs_add, file2vs, vs_path, chatbot])
#下面这2个是核心函数, 把文件给向量化.
                    load_file_button.click(get_vector_store,
                                           show_progress=True,
                                           inputs=[select_vs, files, sentence_size, chatbot, vs_add, vs_add],
                                           outputs=[vs_path, files, chatbot], )

                    load_fileformat_button.click(get_vector_store2,
                                           show_progress=True,
                                           inputs=[select_vs, files, sentence_size, chatbot, vs_add, vs_add],
                                           outputs=[vs_path, files, chatbot], )



                    load_folder_button.click(get_vector_store,
                                             show_progress=True,
                                             inputs=[select_vs, folder_files, sentence_size, chatbot, vs_add,
                                                     vs_add],
                                             outputs=[vs_path, folder_files, chatbot], )
                    flag_csv_logger.setup([query, vs_path, chatbot, mode], "flagged")
#==========下面这个函数就用来触发返回答案功能!!!!!!最核心函数.

                    stream=True
                    # query.submit(get_answer,
                    #              [query, vs_path, chatbot, mode, score_threshold, vector_search_top_k, chunk_conent,
                    #               chunk_sizes],
                    #              [chatbot, query],api_name='tiwen')




                    query.submit(get_answer,
                                 [query, vs_path, chatbot, mode],
                                 [chatbot, query],api_name='tiwen')














    with gr.Tab("知识库测试 Beta"):
        with gr.Row():
            with gr.Column(scale=10):
                chatbot = gr.Chatbot([[None, knowledge_base_test_mode_info]],
                                     elem_id="chat-box",
                                     show_label=False).style(height=750)
                query = gr.Textbox(show_label=False,
                                   placeholder="请输入提问内容，按回车进行提交").style(container=False)
            with gr.Column(scale=5):
                mode = gr.Radio(["知识库测试"],  # "知识库问答",
                                label="请选择使用模式",
                                value="知识库测试",
                                visible=False)
                knowledge_set = gr.Accordion("知识库设定", visible=True)
                vs_setting = gr.Accordion("配置知识库", visible=True)
                mode.change(fn=change_mode,
                            inputs=[mode, chatbot],
                            outputs=[vs_setting, knowledge_set, chatbot])
                with knowledge_set:
                    score_threshold = gr.Number(value=VECTOR_SEARCH_SCORE_THRESHOLD,
                                                label="知识检索内容相关度 Score, 数值范围约为0-1100，如果为0，则不生效，经测试设置为小于500时，匹配结果更精准",
                                                precision=0,
                                                interactive=True)
                    vector_search_top_k = gr.Number(value=VECTOR_SEARCH_TOP_K, precision=0,
                                                    label="获取知识库内容条数", interactive=True)
                    chunk_conent = gr.Checkbox(value=False,
                                               label="是否启用上下文关联",
                                               interactive=True)
                    chunk_sizes = gr.Number(value=CHUNK_SIZE, precision=0,
                                            label="匹配单段内容的连接上下文后最大长度",
                                            interactive=True, visible=False)
                    chunk_conent.change(fn=change_chunk_conent,
                                        inputs=[chunk_conent, gr.Textbox(value="chunk_conent", visible=False), chatbot],
                                        outputs=[chunk_sizes, chatbot])
                with vs_setting:
                    vs_refresh = gr.Button("更新已有知识库选项")
                    select_vs_test = gr.Dropdown(get_vs_list(),
                                            label="请选择要加载的知识库",
                                            interactive=True,
                                            value=get_vs_list()[0] if len(get_vs_list()) > 0 else None)
                    vs_name = gr.Textbox(label="请输入新建知识库名称，当前知识库命名暂不支持中文",
                                         lines=1,
                                         interactive=True,
                                         visible=True)
                    vs_add = gr.Button(value="添加至知识库选项", visible=True)
                    file2vs = gr.Column(visible=False)
                    with file2vs:
                        # load_vs = gr.Button("加载知识库")
                        gr.Markdown("向知识库中添加单条内容或文件")
                        sentence_size = gr.Number(value=SENTENCE_SIZE, precision=0,
                                                  label="文本入库分句长度限制",
                                                  interactive=True, visible=True)
                        with gr.Tab("上传文件"):
                            files = gr.File(label="添加文件",
                                            file_types=['.txt', '.md', '.docx', '.pdf'],
                                            file_count="multiple",
                                            show_label=False
                                            )
                            load_file_button = gr.Button("上传文件并加载知识库")
                        with gr.Tab("上传文件夹"):
                            folder_files = gr.File(label="添加文件",
                                                   # file_types=['.txt', '.md', '.docx', '.pdf'],
                                                   file_count="directory",
                                                   show_label=False)
                            load_folder_button = gr.Button("上传文件夹并加载知识库")
                        with gr.Tab("添加单条内容"):
                            one_title = gr.Textbox(label="标题", placeholder="请输入要添加单条段落的标题", lines=1)
                            one_conent = gr.Textbox(label="内容", placeholder="请输入要添加单条段落的内容", lines=5)
                            one_content_segmentation = gr.Checkbox(value=True, label="禁止内容分句入库",
                                                                   interactive=True)
                            load_conent_button = gr.Button("添加内容并加载知识库")
                    # 将上传的文件保存到content文件夹下,并更新下拉框
                    vs_refresh.click(fn=refresh_vs_list,
                                     inputs=[],
                                     outputs=select_vs_test)
                    vs_add.click(fn=add_vs_name,
                                 inputs=[vs_name, chatbot],
                                 outputs=[select_vs_test, vs_name, vs_add, file2vs, chatbot])
                    select_vs_test.change(fn=change_vs_name_input,
                                     inputs=[select_vs_test, chatbot],
                                     outputs=[vs_name, vs_add, file2vs, vs_path, chatbot])
                    load_file_button.click(get_vector_store,
                                           show_progress=True,
                                           inputs=[select_vs_test, files, sentence_size, chatbot, vs_add, vs_add],
                                           outputs=[vs_path, files, chatbot], )
                    load_folder_button.click(get_vector_store,
                                             show_progress=True,
                                             inputs=[select_vs_test, folder_files, sentence_size, chatbot, vs_add,
                                                     vs_add],
                                             outputs=[vs_path, folder_files, chatbot], )
                    load_conent_button.click(get_vector_store,
                                             show_progress=True,
                                             inputs=[select_vs_test, one_title, sentence_size, chatbot,
                                                     one_conent, one_content_segmentation],
                                             outputs=[vs_path, files, chatbot], )
                    flag_csv_logger.setup([query, vs_path, chatbot, mode], "flagged")
                    query.submit(get_answer,
                                 [query, vs_path, chatbot, mode, score_threshold, vector_search_top_k, chunk_conent,
                                  chunk_sizes],
                                 [chatbot, query])
    if 0:
     with gr.Tab("模型配置"):
        llm_model = gr.Radio(llm_model_dict_list,
                             label="LLM 模型",
                             value=LLM_MODEL,
                             interactive=True)
        no_remote_model = gr.Checkbox(shared.LoaderCheckPoint.no_remote_model,
                                      label="加载本地模型",
                                      interactive=True)

        llm_history_len = gr.Slider(0, 10,
                                    value=LLM_HISTORY_LEN,
                                    step=1,
                                    label="LLM 对话轮数",
                                    interactive=True)
        use_ptuning_v2 = gr.Checkbox(USE_PTUNING_V2,
                                     label="使用p-tuning-v2微调过的模型",
                                     interactive=True)
        use_lora = gr.Checkbox(USE_LORA,
                               label="使用lora微调的权重",
                               interactive=True)
        embedding_model = gr.Radio(embedding_model_dict_list,
                                   label="Embedding 模型",
                                   value=EMBEDDING_MODEL,
                                   interactive=True)
        top_k = gr.Slider(1, 20, value=VECTOR_SEARCH_TOP_K, step=1,
                          label="向量匹配 top k", interactive=True)
        load_model_button = gr.Button("重新加载模型")
        load_model_button.click(reinit_model, show_progress=True,
                                inputs=[llm_model, embedding_model, llm_history_len, no_remote_model, use_ptuning_v2,
                                        use_lora, top_k, chatbot], outputs=chatbot)
        load_knowlege_button = gr.Button("重新构建知识库")
        load_knowlege_button.click(reinit_vector_store, show_progress=True,
                                   inputs=[select_vs, chatbot], outputs=chatbot)
    demo.load(
        fn=refresh_vs_list,
        inputs=None,
        outputs=[select_vs, select_vs_test],
        queue=True,
        show_progress=False,
    )
if __name__ == "__main__":
    (demo
    .queue(concurrency_count=3)
    .launch(server_name='0.0.0.0',
            server_port=7862,
            show_api=True,
            share=False,
            inbrowser=False))
