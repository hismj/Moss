import os

# 获取当前工作目录
current_dir = 'apk_scam'

# 遍历当前目录下的所有文件
for filename in os.listdir(current_dir):
    # 检查文件名是否以.zip.apk结尾
    if filename.endswith(".zip"):
        # 构造新文件名，去掉.apk后缀
        new_filename = filename.replace(".zip", ".apk")

        # 重命名文件
        os.rename(os.path.join(current_dir, filename), os.path.join(current_dir, new_filename))

        print(f"重命名文件: {filename} -> {new_filename}")

print("完成重命名操作。")
