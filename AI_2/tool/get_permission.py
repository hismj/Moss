from androguard.core.apk import APK

# 加载APK文件
apk_path = 'data/train/base11.apk'
apk = APK(apk_path)

# 获取权限信息
permissions = apk.get_permissions()

# 打印权限信息
for permission in permissions:
    print(permission)
