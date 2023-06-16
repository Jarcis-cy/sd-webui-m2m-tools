import gradio as gr
from tag_tools import process_files
from tag_tools import replace_words

def on_ui_tabs():
    with gr.Blocks(analytics_enabled=False) as pro_interface:
        with gr.Row().style(equal_height=False):
            with gr.Column(variant='panel'):
                with gr.Tabs():
                    with gr.TabItem(label='Word Statistics'):
                        folder_input = gr.Textbox(label='Folder Path', lines=1)
                        frequency_input = gr.Slider(minimum=1, maximum=1000, step=1, label='Frequency', value=10)
                        search_input = gr.Textbox(label='Search Word', lines=1)
                        greater_than_checkbox = gr.Checkbox(label="Greater Than Frequency")
                        less_than_checkbox = gr.Checkbox(label="Less Than Frequency")
                        word_stats_button = gr.Button(value="Calculate Word Statistics")
                        freq_output = gr.Textbox(label="Word Frequency", interactive=False, visible=True, placeholder="Word frequencies will be shown here")
                        search_output = gr.Textbox(label="Search Results", interactive=False, visible=True, placeholder="Search results will be shown here")
                        word_stats_button.click(process_files.process, inputs=[folder_input, frequency_input, search_input, greater_than_checkbox, less_than_checkbox], outputs=[freq_output, search_output])

                    with gr.TabItem(label='Replace Words'):
                        folder_input = gr.Textbox(label='Folder Path', lines=1)
                        old_word_input = gr.Textbox(label='Old Word', lines=1)
                        new_word_input = gr.Textbox(label='New Word', lines=1)
                        global_replace_checkbox = gr.Checkbox(label="Global Replace")  # 新增复选框
                        replace_button = gr.Button(value="Replace Words")
                        replace_output = gr.Textbox(label="Replace Status", interactive=False, visible=True, placeholder="Words replaced successfully")
                        # 新增复选框作为一个输入
                        replace_button.click(replace_words.replace, inputs=[folder_input, old_word_input, new_word_input, global_replace_checkbox], outputs=replace_output)

    return [(pro_interface, "Tag tools", "Tag tools")]

if __name__ == "__main__":
    on_ui_tabs()
