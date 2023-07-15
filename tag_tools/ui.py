import gradio as gr
from tag_tools import process_files, replace_words, delete_words
from tag_tools.frame_splitter import split_frames


def on_ui_tabs():
    with gr.Blocks(analytics_enabled=False) as pro_interface:
        with gr.Row().style(equal_height=False):
            with gr.Column(variant='panel'):
                with gr.Tabs():
                    with gr.TabItem(label='M2M'):
                        gr.Markdown("### Stage 1")
                        with gr.Row(variant='panel'):
                            project_input = gr.Textbox(label='Project Path', lines=1)
                            movie_input = gr.Textbox(label='Movie Path', lines=1)
                        with gr.Row(variant='panel'):
                            aim_fps_checkbox = gr.Checkbox(label="启用输出帧率控制")
                            aim_fps = gr.Slider(
                                minimum=1,
                                maximum=60,
                                step=1,
                                label='输出帧率',
                                value=30, interactive=True)
                        btn = gr.Button(value="gene_frame")

                        def split_frames_with_fps_control(project_path, movie_path, fps_checkbox, fps):
                            # If aim_fps_checkbox is True, pass aim_fps to split_frames
                            # Otherwise, pass None to split_frames
                            split_frames(project_path, movie_path, fps if fps_checkbox else None)

                        btn.click(split_frames_with_fps_control,
                                  inputs=[project_input, movie_input, aim_fps_checkbox, aim_fps])
                    with gr.TabItem(label='Word Statistics'):
                        folder_input = gr.Textbox(label='Folder Path', lines=1)
                        frequency_input = gr.Slider(minimum=1, maximum=1000, step=1, label='Frequency', value=10)
                        with gr.Row():
                            search_input = gr.Textbox(label='Search Word', lines=1)

                            options = ['None', 'Body', 'Face', 'Color', 'Sex']
                            option_dropdown = gr.Dropdown(choices=options, label="Auto-fill Options")
                            option_dropdown.change(auto_fill_search_words, inputs=[option_dropdown, search_input],
                                                   outputs=search_input)

                        greater_than_checkbox = gr.Checkbox(label="Greater Than Frequency")
                        less_than_checkbox = gr.Checkbox(label="Less Than Frequency")
                        word_stats_button = gr.Button(value="Calculate Word Statistics")
                        search_output = gr.Textbox(label="Search Results", interactive=False, visible=True,
                                                   placeholder="Search results will be shown here")
                        with gr.Row():
                            freq_output = gr.Textbox(label="Word Frequency", interactive=False, visible=True,
                                                     placeholder="Word frequencies will be shown here")
                            delete_words_input = gr.Textbox(label='Delete Words', lines=1,
                                                            placeholder="Input words to delete, separated by comma")  # define delete_words_input here
                            freq_to_delete_button = gr.Button(value="Process Frequency to Delete Words")
                            freq_to_delete_button.click(process_frequency_to_words, inputs=freq_output,
                                                        outputs=delete_words_input)
                        delete_words_button = gr.Button(value="Delete Words")
                        delete_words_button.click(delete_words.delete_ws, inputs=[folder_input, delete_words_input])
                        word_stats_button.click(process_files.process,
                                                inputs=[folder_input, frequency_input, search_input,
                                                        greater_than_checkbox, less_than_checkbox],
                                                outputs=[freq_output, search_output])
                    with gr.TabItem(label='Replace Words'):
                        folder_input = gr.Textbox(label='Folder Path', lines=1)
                        old_word_input = gr.Textbox(label='Old Word', lines=1,
                                                    placeholder="If left empty, the New Word will be added to the end of all files.")
                        new_word_input = gr.Textbox(label='New Word', lines=1)
                        global_replace_checkbox = gr.Checkbox(label="Global Replace")
                        replace_button = gr.Button(value="Replace Words")
                        replace_output = gr.Textbox(label="Replace Status", interactive=False, visible=True,
                                                    placeholder="Words replaced successfully")
                        replace_button.click(replace_words.replace,
                                             inputs=[folder_input, old_word_input, new_word_input,
                                                     global_replace_checkbox], outputs=replace_output)

    return [(pro_interface, "Tag tools", "Tag tools")]


def process_frequency_to_words(freq_str):
    lines = freq_str.strip().split('\n')
    words = [line.split(':')[0].strip().replace('"', '') for line in lines]
    return ','.join(words)


def auto_fill_search_words(option, search_content):
    body_words = 'arm,leg,head,body,hand,foot,finger,toe,chest,back,shoulder'
    face_words = 'eye,nose,mouth,ear,cheek,chin,forehead,eyebrow,lip,teeth'
    color_words = 'red,blue,green,yellow,black,white,pink,purple,orange,brown,grey'
    sex_words = 'male,female,man,woman,boy,girl,he,she'

    # Determine which words to add based on the selected option
    if option == 'Body':
        words_to_add = body_words
    elif option == 'Face':
        words_to_add = face_words
    elif option == 'Color':
        words_to_add = color_words
    elif option == 'Sex':
        words_to_add = sex_words
    else:
        words_to_add = ''

    # If there's already content in the search box, add a comma before adding the new words
    if search_content:
        if not search_content.endswith(','):
            search_content += ','
        search_content += words_to_add
    else:
        search_content = words_to_add

    return search_content


if __name__ == "__main__":
    on_ui_tabs()
