import os.path

import gradio as gr
from tag_tools import process_files, replace_words, delete_words
from tag_tools.frame_splitter import split_frames, detach_img, reconfiguration, frames_to_video, superposition, \
    exact_match


def on_ui_tabs():
    with gr.Blocks(analytics_enabled=False) as pro_interface:
        with gr.Row().style(equal_height=False):
            with gr.Column(variant='panel'):
                with gr.Tabs():
                    with gr.TabItem(label='M2M'):
                        gr.Markdown("""
                        ## Stage 1
                        输入原视频地址以及项目文件夹地址，可以选择是否按照原视频帧率拆帧，如果不自定义拆帧帧率，则按照原视频帧率拆帧。

                        会创建frame文件夹并把每帧都放入frame文件夹中。
                        """)
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

                        out = gr.Textbox(label="log info", interactive=False, visible=True, placeholder="output log")
                        btn.click(split_frames_with_fps_control,
                                  inputs=[project_input, movie_input, aim_fps_checkbox, aim_fps], outputs=out)

                        gr.Markdown("""
                        ## Stage 2
                        这一步是为使用segment anything做批量抠图设计的便捷功能
                        
                        在使用这一步之前，请确保已经做完以下操作：
                        1. 开启GroundingDINO，源目录填写 项目地址/frame 目标目录填写 项目地址/Out 
                        2. 单独输出每张图像一般选择3，这样可以选择更合适的抠图效果
                        3. 选择保存蒙版后图像和保存蒙版
                        4. 开始批量处理
                        
                        然后在下方选择一个合适的抠图效果
                        """)
                        calcmode = gr.Radio(label="请选择你喜欢的蒙版", choices=["0", "1", "2"], value="2")
                        btn1 = gr.Button(value="detach_img")
                        out1 = gr.Textbox(label="log info", interactive=False, visible=True, placeholder="output log")
                        btn1.click(detach_img, inputs=[project_input, calcmode], outputs=out1)
                        gr.Markdown("""
                        ## Stage 3
                        流程分为开启精准匹配和非精准匹配两种，精准匹配可以将图完整的融合进原视频中，但没有竖屏视频的生成，非精准匹配可以生成竖屏视频
                        
                        #### 精准匹配模式
                        
                        如果开启精准匹配模式，程序会尝试匹配出蒙版中的主体，然后裁切并记录裁切大小和具体坐标。这样裁切出来的内容大小是不一致的
                        点击运行，程序会检查是否存在video_key文件夹，如果存在，则会执行精准裁切模式，并在裁切完成后，创建exact_img2img_key文件夹，请将图生图后的结果放入其中。如果图生图出的尺寸不是原尺寸，则应该先放大回原尺寸再放入其中
                        如果不存在，则会创建好video_frame、video_mask文件夹，并将frame和mask文件夹中的内容拷贝进去等待执行ebs的执行stage 2生成key帧
                        
                        **所以有三个注意点：**
                        1. 在图生图的时候，不能选择固定的像素大小，而是应该选择倍率。
                        2. 流程：抠图 -> Stage 2 -> ebs填写源视频地址和项目目录，并执行stage 2生成key帧 -> Stage 3(精准匹配) -> 图生图 -> 返回ebs的stage5生成ebs文件 -> 风格迁移 -> 生成新视频
                        3. 当选择精准匹配模式后，程序会创建exact_frame文件夹和exact_mask文件夹，并将裁切好的蒙版和帧序列文件存入其中。位置信息将会被保存到exact_crop_info.json这个文件中
                        
                        #### 非精准匹配模式
                        点击运行，会创建好video_frame、video_mask文件夹，然后执行重构序列的操作，并将过程中记录的位置信息保存到crop_info.json这个文件中，
                        
                        将重构好的frame保存到video_frame文件夹中，重构好的mask保存到video_mask文件夹中。
                        
                        流程：抠图 -> Stage 2 -> Stage 3(非精准匹配) -> Stage 4 -> Stage 5 ->Stage 4
                        
                        调整step将会调整扫描框的移动步长，会加快生成效率，但有可能会降低生成质量
                        
                        smooth_factor为平滑过度参数，默认0.9，不建议调整，数值越小，人物在图片中的占比会越大，越居中，但可能导致前后两帧变化幅度过大
                        """)
                        with gr.Row(variant='panel'):
                            exact_match_cb = gr.Checkbox(label="开启精准匹配模式")
                            ebs_cb = gr.Checkbox(label="是否使用ebs", value=True)
                        with gr.Row(variant='panel'):
                            step_input = gr.Slider(minimum=1, maximum=100, step=1, label='step size', value=5)
                            width = gr.Slider(minimum=1, maximum=4000, step=1, label='sequence width', lines=1,
                                              value=810)
                            smooth_factor = gr.Slider(minimum=0, maximum=1, step=0.1, label='smooth factor', value=0.9)
                        btn2 = gr.Button(value="reconfiguration")
                        out2 = gr.Textbox(label="log info", interactive=False, visible=True, placeholder="output log")
                        if exact_match_cb:
                            btn2.click(exact_match, inputs=[project_input, ebs_cb], outputs=out2)
                        else:
                            btn2.click(reconfiguration,
                                       inputs=[project_input, width, movie_input, step_input, smooth_factor],
                                       outputs=out2)
                        gr.Markdown("""
                        ## Stage 4
                        重新合并帧序列，将重构好的帧重新生成一个新的视频，便于ebs处理

                        直接点击运行，即可参考原视频的帧率进行合成视频，生成一个tmp.mp4（可修改）保存到项目文件夹中
                        
                        也可以在文本框中输入一个帧序列地址，然后指定帧率进行合成，不指定则参考原视频的帧率
                        """)
                        with gr.Row(variant='panel'):
                            fps_cbox = gr.Checkbox(label="启用输出帧率控制")
                            fps = gr.Slider(minimum=1, maximum=60, step=1, label='FPS', value=30)
                        frame_input_dir = gr.Textbox(label='图片输入地址', lines=1,
                                                     placeholder='input folder, default: video_frame')
                        video_output_dir = gr.Textbox(label='视频输出地址/名称', lines=1,
                                                      placeholder='output folder, default: tmp.mp4')
                        btn3 = gr.Button(value="gene_video")
                        out3 = gr.Textbox(label="log info", interactive=False, visible=True, placeholder="output log")

                        def fps_control(project_path, frame_path, video_path, movie_path, cfps, fps_checkbox):
                            return frames_to_video(project_path, frame_path, video_path, movie_path,
                                                   cfps if fps_checkbox else None)

                        btn3.click(fps_control,
                                   inputs=[project_input, frame_input_dir, video_output_dir, movie_input, fps,
                                           fps_cbox],
                                   outputs=out3)
                        gr.Markdown("""
                        ## Stage 5
                        
                        #### 精准匹配模式：
                        
                        程序会读取exact_crop_info.json中的信息，然后查看exact_img2img_key中的图片信息，会先确认大小与文件中记录的大小是否一致，如果不一致，则会提示先放大回原图再生成
                        
                        如果信息无误，则会开始拼接工作，并将拼接后的结果放入img2img_key文件夹中，便于ebs进行操作。
                        
                        #### 非精准匹配模式：
                        前往ebs的流程操作，进行生成key帧->图生图->图片放大还原->生成ebs文件->风格迁移->生成新视频（也就是ebs插件执行完stage 7）

                        执行完成后，应该会有一个文件夹crossfade_tmp，这个文件夹中存放着所有的风格迁移好的文件，此时点击运行，插件会先创建refactor_frame文件夹，然后会读取crossfade_tmp中的所有png文件，
                        
                        然后根据crop_info.json的内容，将这个文件夹中的同名文件叠加到frame文件夹的同名文件上，并保存到refactor_frame中，这个文件夹中的内容即是变回横板后的图片。
                        
                        此时可以回到stage 4，生成视频。
                        """)
                        img2img_input_dir = gr.Textbox(label='图片输入地址', lines=1,
                                                       placeholder='input folder, default: if exact：exact_img2img_key '
                                                                   'else: crossfade_tmp')
                        btn4 = gr.Button(value="superposition")
                        out4 = gr.Textbox(label="log info", interactive=False, visible=True, placeholder="output log")
                        if exact_match:
                            pass
                        else:
                            btn4.click(superposition,
                                       inputs=[project_input, img2img_input_dir],
                                       outputs=out4)

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
                                                    placeholder="If left empty, the New Word will be added to the end "
                                                                "of all files.")
                        new_word_input = gr.Textbox(label='New Word', lines=1)
                        global_replace_checkbox = gr.Checkbox(label="Global Replace")
                        replace_button = gr.Button(value="Replace Words")
                        replace_output = gr.Textbox(label="Replace Status", interactive=False, visible=True,
                                                    placeholder="Words replaced successfully")
                        replace_button.click(replace_words.replace,
                                             inputs=[folder_input, old_word_input, new_word_input,
                                                     global_replace_checkbox], outputs=replace_output)

    return [(pro_interface, "M2M tools", "M2M tools")]


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
