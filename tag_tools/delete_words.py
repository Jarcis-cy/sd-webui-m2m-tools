import os

def delete_ws(folder_path, delete_words):
    delete_word_list = delete_words.split(',')
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.endswith('.txt'):
                with open(os.path.join(root, file), 'r') as f:
                    content = f.read()
                # 删除内容中的单词
                for word in delete_word_list:
                    # 当单词后面有逗号时，同时删除单词和逗号
                    content = content.replace(word.strip() + ',', '')
                    # 当单词后面没有逗号时，只删除单词
                    content = content.replace(word.strip(), '')
                # 将修改后的内容写回文件
                with open(os.path.join(root, file), 'w') as f:
                    f.write(content)