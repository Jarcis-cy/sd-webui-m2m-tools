import os
from collections import Counter


def process(folder_path, frequency, search_word, greater_than, less_than, delete_words):
    freq_counter = Counter()
    delete_word_list = delete_words.split(',') if delete_words else []

    search_words = search_word.split(',') if search_word else []
    search_counters = {word.strip(): Counter() for word in search_words}  # 为每个搜索词创建一个 Counter

    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.endswith('.txt'):
                with open(os.path.join(root, file), 'r') as f:
                    content = f.read()
                    words = [word.strip() for word in content.split(',') if
                             word not in delete_word_list]  # 添加 strip 方法来删除前导和尾随空白，同时删除不想要的单词
                    freq_counter.update(words)
                    for search_word, search_counter in search_counters.items():
                        search_counter.update(w for w in words if search_word in w)

    freq_result = '\n'.join([f'"{word}": {count}' for word, count in freq_counter.items() if
                             (greater_than and count > frequency) or (
                                         less_than and count < frequency) or count == frequency])

    # 美化搜索结果的输出
    # 美化搜索结果的输出
    search_result = '\n\n'.join(
        [f'{word}:\n' + '\n'.join([f'  {found_word}: {count}' for found_word, count in counter.items()]) for
         word, counter in search_counters.items() if sum(counter.values()) > 0]) if search_words else ''


    return freq_result, search_result
