import os
from collections import Counter

def process(folder_path, frequency, search_word, greater_than, less_than, delete_words):
    freq_counter = Counter()
    search_counter = Counter() if search_word else None

    delete_word_list = delete_words.split(',') if delete_words else []

    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.endswith('.txt'):
                with open(os.path.join(root, file), 'r') as f:
                    content = f.read()
                    words = [word.strip() for word in content.split(',') if word not in delete_word_list]  # 添加 strip 方法来删除前导和尾随空白，同时删除不想要的单词
                    freq_counter.update(words)
                    if search_word:
                        search_counter.update(w for w in words if search_word in w)

    freq_result = '\n'.join([f'"{word}": {count}' for word, count in freq_counter.items() if (greater_than and count > frequency) or (less_than and count < frequency) or count == frequency])
    search_result = '\n'.join([f'"{word}": {count}' for word, count in search_counter.items()]) if search_word else ''

    return freq_result, search_result
