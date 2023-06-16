import os

def replace(folder_path, old_word, new_word, global_replace):
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.endswith('.txt'):
                with open(os.path.join(root, file), 'r') as f:
                    content = f.read().strip()

                if old_word == "":
                    if not content.endswith(",") and not new_word.startswith(","):
                        new_word = "," + new_word
                    if content.endswith(",") and not new_word.startswith(","):
                        new_word = new_word[1:]
                    new_content = content + new_word
                elif global_replace:
                    new_content = content.replace(old_word, new_word)
                else:
                    words = content.split(',')
                    new_words = [new_word if word.strip() == old_word else word for word in words]
                    new_content = ','.join(new_words)

                with open(os.path.join(root, file), 'w') as f:
                    f.write(new_content)

    return "Words replaced successfully"
