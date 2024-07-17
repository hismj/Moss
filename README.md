## Moss 涉诈APP智能分析识别系统
*版本号Version：1.0.0*
作者Author：宋明键、孟琦
![Moss](https://raw.githubusercontent.com/hismj/Moss/master/mobsf/static/img/moss_logo.png)
Moss涉诈APP智能分析识别系统是一个专注于Android移动应用安全的智能分析平台。该系统集成了静态分析、动态分析功能，并且能够基于机器学习及深度学习方法进行AI分析。该系统通过处理APK源代码、分析应用权限请求、解包APK文件的静态信息等方法，对涉诈内容及类似应用集群进行有效识别，可满足移动应用安全、反诈识别、恶意软件分析和隐私分析等多种用途。动态分析模块支持Genymotion，Android Studio等安卓模拟器，基于Frida框架提供交互式的测试环境，支持运行时数据和网络流量分析。Moss涉诈APP智能分析识别系统通过REST API和CLI工具与DevSecOps或CI/CD管道无缝集成，为安全工作流程提供了便利和增强。系统特点主要结合深度学习、机器学习

1. 下载获取Moss
```
git clone https://github.com/hismj/Moss.git
```
2. 安装依赖

 - python==3.10
 - keras==2.15.0
 - tensorflow==2.15.0
 - pyzbar
 - torch
 - androguard==4.1.2
详细其他包见requirements.txt




<!--stackedit_data:
eyJoaXN0b3J5IjpbLTU0NzM1Njg3NywxNzE1MzE4MTA0LDEwMD
g1Mjc0ODAsLTEyMTgxODEzMTAsMzMwNDMyMDQ2XX0=
-->