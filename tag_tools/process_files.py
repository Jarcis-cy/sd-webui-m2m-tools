import os
from collections import Counter

def process(folder_path, frequency, search_word, greater_than, less_than):
    freq_counter = Counter()
    search_counter = Counter() if search_word else None

    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.endswith('.txt'):
                with open(os.path.join(root, file), 'r') as f:
                    content = f.read()
                    words = [word.strip() for word in content.split(',')]  # 添加 strip 方法来删除前导和尾随空白
                    freq_counter.update(words)
                    if search_word:
                        search_counter.update(w for w in words if search_word in w)

    freq_result = '\n'.join([f'"{word}": {count}' for word, count in freq_counter.items() if (greater_than and count > frequency) or (less_than and count < frequency) or count == frequency])
    search_result = '\n'.join([f'"{word}": {count}' for word, count in search_counter.items()]) if search_word else ''

    return freq_result, search_result
