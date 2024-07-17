import os
import re

def extract_number(filename):
    """从文件名中提取数字部分"""
    match = re.search(r'_(\d+)\.png', filename)
    return int(match.group(1)) if match else float('inf')

def generate_labels(img_folder, label_type):
    """生成标签文件"""
    img_files = [f for f in os.listdir(img_folder) if f.endswith('.png')]
    img_files_sorted = sorted(img_files, key=extract_number)

    label_file = f'{label_type}.txt'
    with open(label_file, 'w') as f:
        for img_file in img_files_sorted:
            f.write(f'{img_file} {label_type}\n')
    print(f"标签文件 {label_file} 已生成。")

# 获取用户输入选择
user_input = input("输入0执行scam部分，输入1执行benign部分：")

if user_input == '0':
    img_folder = input("输入图片文件夹路径：")
    generate_labels(img_folder, 'scam')
elif user_input == '1':
    img_folder = input("输入图片文件夹路径：")
    generate_labels(img_folder, 'benign')
else:
    print("输入无效，请输入0或1。")
