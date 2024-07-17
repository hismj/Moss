import os
import shutil

# 指定目录路径
directory = 'apk_scam'

# 获取目录中的所有文件名
files = os.listdir(directory)

# 遍历每个文件，并给文件添加 .apk_scam 扩展名
for filename in files:
    if not filename.endswith('.apk'):
        # 构造旧的文件路径和新的文件路径
        old_filepath = os.path.join(directory, filename)
        new_filepath = old_filepath + '.apk'

        # 重命名文件，添加 .apk_scam 扩展名
        shutil.move(old_filepath, new_filepath)

print('扩展名添加完成。')
