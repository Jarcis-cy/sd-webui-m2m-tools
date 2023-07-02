import os

def delete_ws(folder_path, delete_words):
    delete_word_list = delete_words.split(',')
    delete_word_list = sorted([word.strip() for word in delete_word_list], key=len, reverse=True)

    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.endswith('.txt'):
                with open(os.path.join(root, file), 'r') as f:
                    content = f.read()

                # 先处理长度较长的项
                for word_to_delete in delete_word_list:
                    word_list = content.split(',')
                    new_word_list = [word for word in word_list if word.strip() != word_to_delete]
                    content = ','.join(new_word_list)

                # 将修改后的内容写回文件
                with open(os.path.join(root, file), 'w') as f:
                    f.write(content)
