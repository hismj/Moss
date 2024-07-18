## Moss 涉诈APP智能分析识别系统
***版本号Version：1.0.0*
作者Author：宋明键、孟琦**
![Moss](https://raw.githubusercontent.com/hismj/Moss/master/mobsf/static/img/moss_logo.png)
Moss涉诈APP智能分析识别系统是一个专注于Android移动应用安全的智能分析平台。该系统集成了静态分析、动态分析功能，并且能够基于机器学习及深度学习方法进行AI分析。该系统通过处理APK源代码、分析应用权限请求、解包APK文件的静态信息等方法，对涉诈内容及类似应用集群进行有效识别，可满足移动应用安全、反诈识别、恶意软件分析和隐私分析等多种用途。动态分析模块支持Genymotion，Android Studio等安卓模拟器，基于Frida框架提供交互式的测试环境，支持运行时数据和网络流量分析。Moss涉诈APP智能分析识别系统通过REST API和CLI工具与DevSecOps或CI/CD管道无缝集成，为安全工作流程提供了便利和增强。系统特点主要结合深度学习、机器学习

Github链接：[hismj/Moss at master (github.com)](https://github.com/hismj/Moss/tree/master)

 **1. 安装环境依赖**

 - 安装 Git（必要）
 - 安装 Python 3.10+（必要）
 - 安装 JDK 8+（必要）
 - 安装 Microsoft Visual C++ Build Tools（必要）
 - 安装 OpenSSL（non-light）（必要）
 - 安装Genymotion或Android studio等模拟器，用于运行动态分析（必要）
 - 安装 wkhtmltopdf，并将包含 wkhtmltopdf 的二进制文件路径添加到环境变量 PATH 里（非必要）
 - keras==2.15.0
 - tensorflow==2.15.0
 - pyzbar
 - torch
 - androguard==4.1.2
详细其他包见requirements.txt

**2. 下载获取Moss**

从github上获取
```
git clone https://github.com/hismj/Moss.git
```
安装并初始化
```
setup.bat
```
**3. 运行**

在本地或服务器上运行启动
```
run.bat 127.0.0.1:8000
```
在浏览器上进入
```
127.0.0.1:8000
```
**4. 登录注册**

管理员账户
用户名moss
密码moss
点击注册可以创建新账户
输入相关信息即可

**5. AI分析**

- 点击home页面顶部菜单栏中的AI分析即可进入AI分析界面
- 点击“选择APK文件”可以选择一个或多个文件上传（不选择会提示错误）
- 点击“选择模型”可以选择目前我们已预训练好的两个模型（不选择会提示错误）
- 在文本框中输入apk下载的url，点击“下载apk”可以从url中下载apk并直接上传
- 点击“上传二维码图片”可以从二维码中提取url并下载apk上传

**6. 简单静态分析**

- 点击页面顶部菜单栏中的简单静态分析即可进入简单静态分析界面，也就是最初页面
- 可以上传apk文件
- 上传完毕会自动进入分析页面
- 可以点击页面顶部菜单栏中的最近检测，查看之前分析过的文件

**7. 动态分析**

- 动态分析需要下载Genymotion或Android studio等模拟器，Genymotion或Android studio下载配置好，创建虚拟机后会自动连接到系统
- 注意：请先启动模拟器再运行
- 点击中心“安卓动态分析”按钮，连接模拟器，此时如果报错可能是模拟器未能正常建立连接。

**注意：如果模拟器无法正常连接，请遵循以下步骤**
1. 请下载  [Android Studio Emulator](https://developer.android.com/studio) 模拟器
2. 创建**Android 9.0、API 28、非Google Play**的AVD
3. **使用emulator从命令行运行 AVD**
将 Android SDK 模拟器目录追加到环境变量`PATH`。
一些示例位置
苹果电脑 `/Users/<user>/Library/Android/sdk/emulator`
Linux `/home/<user>/Android/Sdk/emulator`
Windows `C:\Users\<user>\AppData\Local\Android\Sdk\emulator`
4. **列出可用的 Android 虚拟设备 （AVD）名称**
```
$ emulator -list-avds
```
5. **运行 Android 虚拟设备 （AVD）**
```
$ emulator -avd <your_avd_name> -writable-system -no-snapshot
```
将<your_avd_name>替换为4.步列出的可用的 Android 虚拟设备 （AVD）名称
- 若无法联网，请尝试配置网络，参考[Android Studio模拟器无法连接网络_android studio模拟器无法联网-CSDN博客](https://blog.csdn.net/qq_51802315/article/details/124852026?ops_request_misc=&request_id=&biz_id=102&utm_term=Android%20StudioAVD%E6%A8%A1%E6%8B%9F%E5%99%A8%E9%85%8D%E7%BD%AE%E7%BD%91%E7%BB%9C&utm_medium=distribute.pc_search_result.none-task-blog-2~all~sobaiduweb~default-1-124852026.142^v100^pc_search_result_base4&spm=1018.2226.3001.4449)
- 页面中出现简单静态分析扫描过的apk，可选择将其下载的模拟器中运行，然后选择对何种行为进行动态分析

本项目参考MobSF移动安全测试框架：
[https://github.com/MobSF/Mobile-Security-Framework-MobSF](https://github.com/MobSF/Mobile-Security-Framework-MobSF)





<!--stackedit_data:
eyJoaXN0b3J5IjpbLTQ2NzIzMjI1LDE5MjQyNzI2NTEsLTE5Mz
M2NTE5ODUsLTEwMTQxOTE2NTYsMTk5Mzg3MzE4MSwtODU3NTAx
NTgzLC0xNjgyOTk5NjMzLC02MDE0NzgyNDYsMTcxNTMxODEwNC
wxMDA4NTI3NDgwLC0xMjE4MTgxMzEwLDMzMDQzMjA0Nl19
-->