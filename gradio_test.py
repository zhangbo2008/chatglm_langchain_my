import gradio as gr

demo = gr.Blocks(css="""#btn {color: red} .abc {font-family: "Comic Sans MS", "Comic Sans", cursive !important}""")
a=333 #这里面加上保证每次刷新页面都是从这个开始.
with demo:
    default_json = {"a": "a"}

    num = gr.State(value=a)
    squared = gr.Number(value=a*a)
    btn = gr.Button("Next Square", elem_id="btn", elem_classes=["abc", "def"])
    btn = gr.Button("Next Square2", elem_id="btn", elem_classes=["abc", "def"])

    stats = gr.State(value=default_json)
    table = gr.JSON()

    def increase(var, stats_history):
        var += 1
        stats_history[str(var)] = var**2
        return var, var**2, stats_history, stats_history

    btn.click(increase, [num, stats], [num, squared, stats, table],api_name="addition")
    btn.click(increase, [num, stats], [num, squared, stats, table],)

if __name__ == "__main__":
    demo.launch()
