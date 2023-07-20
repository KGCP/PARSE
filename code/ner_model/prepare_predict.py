
import re

if __name__ == '__main__':
    text_path = "./dataset/askg/test.txt"
    with open(text_path, 'r') as f:
        line = f.readline()
        line = re.split("[ ]", line)
        for item in line:
            if "\\n" in item:
                item0 = item.split('\\n')
                i=1

