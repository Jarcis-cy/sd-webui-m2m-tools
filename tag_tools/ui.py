import gradio as gr
from tag_tools import process_files, replace_words, delete_words

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
                        delete_words_input = gr.Textbox(label='Delete Words', lines=1, placeholder="Input words to delete, separated by comma")
                        delete_words_button = gr.Button(value="Delete Words")
                        delete_words_button.click(delete_words.delete_ws, inputs=[folder_input, delete_words_input])
                        word_stats_button.click(process_files.process, inputs=[folder_input, frequency_input, search_input, greater_than_checkbox, less_than_checkbox], outputs=[freq_output, search_output])

                    with gr.TabItem(label='Replace Words'):
                        folder_input = gr.Textbox(label='Folder Path', lines=1)
                        old_word_input = gr.Textbox(label='Old Word', lines=1)
                        new_word_input = gr.Textbox(label='New Word', lines=1)
                        global_replace_checkbox = gr.Checkbox(label="Global Replace")  
                        replace_button = gr.Button(value="Replace Words")
                        replace_output = gr.Textbox(label="Replace Status", interactive=False, visible=True, placeholder="Words replaced successfully")
                        replace_button.click(replace_words.replace, inputs=[folder_input, old_word_input, new_word_input, global_replace_checkbox], outputs=replace_output)

    return [(pro_interface, "Tag tools", "Tag tools")]

if __name__ == "__main__":
    on_ui_tabs()
