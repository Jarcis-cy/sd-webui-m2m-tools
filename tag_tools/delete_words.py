import os

def delete_ws(folder_path, delete_words):
    delete_word_list = delete_words.split(',')
    delete_word_list = [word.strip() for word in delete_word_list]

    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.endswith('.txt'):
                with open(os.path.join(root, file), 'r') as f:
                    content = f.read()

                # 将文件内容转为单词数组
                word_list = content.split(',')

                # 生成一个新的单词数组，其中不包含待删除的单词
                new_word_list = [word for word in word_list if word.strip() not in delete_word_list]

                # 将新的单词数组连接为字符串，单词间以逗号分隔
                new_content = ','.join(new_word_list)

                # 将修改后的内容写回文件
                with open(os.path.join(root, file), 'w') as f:
                    f.write(new_content)
