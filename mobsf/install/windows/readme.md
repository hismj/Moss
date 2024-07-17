# Windows App Analysis

If you are running MobSF in Windows host, you do not have to configure anything, apart from interacting with the automated installation script for the first time when you run MobSF. However, if you are using a different host OS, you need to configure a Windows VM. Sadly [binskim](https://www.nuget.org/packages/Microsoft.CodeAnalysis.BinSkim/) is only available on Windows. So even for static analysis, a Windows VM is required.

## Steps on the Windows-VM
1. Install the following requirements on the VM
  * [Python 3](https://www.python.org/downloads/)
  * rsa (via `python -m pip install rsa`)
2. There is some manual interaction, but if there are no errors, everything is good and the RPC-Server should be running
3. Do the steps of the next section for Moss

## Video: Configuring Windows VM and Moss for Windows App Static Analysis
[![Configuring Windows VM and Moss for Windows App Static Analysis](https://img.youtube.com/vi/17ilENuMj58/0.jpg)](https://www.youtube.com/watch?v=17ilENuMj58)


## Caution
Use separate Windows-VM for Moss and *don't* expose it to a network range where an attack might be coming from. The best solution is to set it to host-only mode.


## Steps for Moss 
To integrate a Windows-VM into Moss, please follow these steps.

* Get the IP of your Windows VM and set `WINDOWS_VM_IP` in `<user_home_dir>/.Moss/config.py`.
* (If not yet done:) Copy the private rsa key from the VM to Moss

NOTE: These steps are not required, if you are running Moss in a Windows Host.

## FAQ

* If you see errors like this

```
Unhandled Exception: System.NotSupportedException: The requested security protocol is not supported.
   at System.Net.ServicePointManager.set_SecurityProtocol(SecurityProtocolType value)
   at NuGet.CommandLine.Program.MainCore(String workingDirectory, String[] args)
   at NuGet.CommandLine.Program.Main(String[] args)
```
Install [.NET Framework 4.6](https://www.microsoft.com/en-in/download/confirmation.aspx?id=48130)

*  Error: **AttributeError: ConfigParser instance has no attribute 'getitem'**

Moss
setup script assume that your VM or host Windows box have a C Drive and you have all the permissions to perform read/write operations in `C:\Moss`. This error occurs if you don't have proper read/write permissions.
