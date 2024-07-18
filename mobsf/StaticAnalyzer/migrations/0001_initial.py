# Generated by Django 5.0.7 on 2024-07-14 15:20

import datetime
from django.db import migrations, models


class Migration(migrations.Migration):

    initial = True

    dependencies = []

    operations = [
        migrations.CreateModel(
            name="RecentScansDB",
            fields=[
                ("ANALYZER", models.CharField(default="", max_length=50)),
                ("SCAN_TYPE", models.CharField(default="", max_length=10)),
                ("FILE_NAME", models.CharField(default="", max_length=260)),
                ("APP_NAME", models.CharField(default="", max_length=260)),
                ("PACKAGE_NAME", models.CharField(default="", max_length=260)),
                ("VERSION_NAME", models.CharField(default="", max_length=50)),
                (
                    "MD5",
                    models.CharField(
                        default="", max_length=32, primary_key=True, serialize=False
                    ),
                ),
                ("TIMESTAMP", models.DateTimeField(default=datetime.datetime.now)),
            ],
            options={
                "permissions": (
                    ("can_delete", "Delete Scans"),
                    ("can_scan", "Scan Files"),
                ),
            },
        ),
        migrations.CreateModel(
            name="StaticAnalyzerAndroid",
            fields=[
                ("FILE_NAME", models.CharField(default="", max_length=260)),
                ("APP_NAME", models.CharField(default="", max_length=255)),
                ("APP_TYPE", models.CharField(default="", max_length=20)),
                ("SIZE", models.CharField(default="", max_length=50)),
                (
                    "MD5",
                    models.CharField(
                        default="", max_length=32, primary_key=True, serialize=False
                    ),
                ),
                ("SHA1", models.CharField(default="", max_length=40)),
                ("SHA256", models.CharField(default="", max_length=64)),
                ("PACKAGE_NAME", models.TextField(default="")),
                ("MAIN_ACTIVITY", models.TextField(default="")),
                ("EXPORTED_ACTIVITIES", models.TextField(default="")),
                ("BROWSABLE_ACTIVITIES", models.TextField(default={})),
                ("ACTIVITIES", models.TextField(default=[])),
                ("RECEIVERS", models.TextField(default=[])),
                ("PROVIDERS", models.TextField(default=[])),
                ("SERVICES", models.TextField(default=[])),
                ("LIBRARIES", models.TextField(default=[])),
                ("TARGET_SDK", models.CharField(default="", max_length=50)),
                ("MAX_SDK", models.CharField(default="", max_length=50)),
                ("MIN_SDK", models.CharField(default="", max_length=50)),
                ("VERSION_NAME", models.CharField(default="", max_length=100)),
                ("VERSION_CODE", models.CharField(default="", max_length=50)),
                ("ICON_PATH", models.TextField(default="")),
                ("PERMISSIONS", models.TextField(default={})),
                ("MALWARE_PERMISSIONS", models.TextField(default={})),
                ("CERTIFICATE_ANALYSIS", models.TextField(default={})),
                ("MANIFEST_ANALYSIS", models.TextField(default=[])),
                ("BINARY_ANALYSIS", models.TextField(default=[])),
                ("FILE_ANALYSIS", models.TextField(default=[])),
                ("ANDROID_API", models.TextField(default={})),
                ("CODE_ANALYSIS", models.TextField(default={})),
                ("NIAP_ANALYSIS", models.TextField(default={})),
                ("PERMISSION_MAPPING", models.TextField(default={})),
                ("URLS", models.TextField(default=[])),
                ("DOMAINS", models.TextField(default={})),
                ("EMAILS", models.TextField(default=[])),
                ("STRINGS", models.TextField(default={})),
                ("FIREBASE_URLS", models.TextField(default=[])),
                ("FILES", models.TextField(default=[])),
                ("EXPORTED_COUNT", models.TextField(default={})),
                ("APKID", models.TextField(default={})),
                ("QUARK", models.TextField(default=[])),
                ("TRACKERS", models.TextField(default={})),
                ("PLAYSTORE_DETAILS", models.TextField(default={})),
                ("NETWORK_SECURITY", models.TextField(default=[])),
                ("SECRETS", models.TextField(default=[])),
            ],
            options={
                "permissions": (
                    ("can_delete", "Delete Scans"),
                    ("can_scan", "Scan Files"),
                ),
            },
        ),
        migrations.CreateModel(
            name="StaticAnalyzerIOS",
            fields=[
                ("FILE_NAME", models.CharField(default="", max_length=255)),
                ("APP_NAME", models.CharField(default="", max_length=255)),
                ("APP_TYPE", models.CharField(default="", max_length=20)),
                ("SIZE", models.CharField(default="", max_length=50)),
                (
                    "MD5",
                    models.CharField(
                        default="", max_length=32, primary_key=True, serialize=False
                    ),
                ),
                ("SHA1", models.CharField(default="", max_length=40)),
                ("SHA256", models.CharField(default="", max_length=64)),
                ("BUILD", models.TextField(default="")),
                ("APP_VERSION", models.CharField(default="", max_length=100)),
                ("SDK_NAME", models.CharField(default="", max_length=50)),
                ("PLATFORM", models.CharField(default="", max_length=50)),
                ("MIN_OS_VERSION", models.CharField(default="", max_length=50)),
                ("BUNDLE_ID", models.TextField(default="")),
                ("BUNDLE_URL_TYPES", models.TextField(default=[])),
                (
                    "BUNDLE_SUPPORTED_PLATFORMS",
                    models.CharField(default=[], max_length=50),
                ),
                ("ICON_PATH", models.TextField(default="")),
                ("INFO_PLIST", models.TextField(default="")),
                ("BINARY_INFO", models.TextField(default={})),
                ("PERMISSIONS", models.TextField(default={})),
                ("ATS_ANALYSIS", models.TextField(default=[])),
                ("BINARY_ANALYSIS", models.TextField(default=[])),
                ("MACHO_ANALYSIS", models.TextField(default={})),
                ("DYLIB_ANALYSIS", models.TextField(default=[])),
                ("FRAMEWORK_ANALYSIS", models.TextField(default=[])),
                ("IOS_API", models.TextField(default={})),
                ("CODE_ANALYSIS", models.TextField(default={})),
                ("FILE_ANALYSIS", models.TextField(default=[])),
                ("LIBRARIES", models.TextField(default=[])),
                ("FILES", models.TextField(default=[])),
                ("URLS", models.TextField(default=[])),
                ("DOMAINS", models.TextField(default={})),
                ("EMAILS", models.TextField(default=[])),
                ("STRINGS", models.TextField(default=[])),
                ("FIREBASE_URLS", models.TextField(default=[])),
                ("APPSTORE_DETAILS", models.TextField(default={})),
                ("SECRETS", models.TextField(default=[])),
                ("TRACKERS", models.TextField(default={})),
            ],
            options={
                "permissions": (
                    ("can_delete", "Delete Scans"),
                    ("can_scan", "Scan Files"),
                ),
            },
        ),
        migrations.CreateModel(
            name="StaticAnalyzerWindows",
            fields=[
                ("FILE_NAME", models.CharField(default="", max_length=260)),
                ("APP_NAME", models.CharField(default="", max_length=260)),
                ("PUBLISHER_NAME", models.TextField(default="")),
                ("SIZE", models.CharField(default="", max_length=50)),
                (
                    "MD5",
                    models.CharField(
                        default="", max_length=32, primary_key=True, serialize=False
                    ),
                ),
                ("SHA1", models.CharField(default="", max_length=40)),
                ("SHA256", models.CharField(default="", max_length=64)),
                ("APP_VERSION", models.TextField(default="")),
                ("ARCHITECTURE", models.TextField(default="")),
                ("COMPILER_VERSION", models.TextField(default="")),
                ("VISUAL_STUDIO_VERSION", models.TextField(default="")),
                ("VISUAL_STUDIO_EDITION", models.TextField(default="")),
                ("TARGET_OS", models.TextField(default="")),
                ("APPX_DLL_VERSION", models.TextField(default="")),
                ("PROJ_GUID", models.TextField(default="")),
                ("OPTI_TOOL", models.TextField(default="")),
                ("TARGET_RUN", models.TextField(default="")),
                ("FILES", models.TextField(default=[])),
                ("STRINGS", models.TextField(default=[])),
                ("BINARY_ANALYSIS", models.TextField(default=[])),
                ("BINARY_WARNINGS", models.TextField(default=[])),
            ],
            options={
                "permissions": (
                    ("can_delete", "Delete Scans"),
                    ("can_scan", "Scan Files"),
                ),
            },
        ),
        migrations.CreateModel(
            name="SuppressFindings",
            fields=[
                (
                    "id",
                    models.AutoField(
                        auto_created=True,
                        primary_key=True,
                        serialize=False,
                        verbose_name="ID",
                    ),
                ),
                ("PACKAGE_NAME", models.CharField(default="", max_length=260)),
                ("SUPPRESS_RULE_ID", models.TextField(default=[])),
                ("SUPPRESS_FILES", models.TextField(default={})),
                ("SUPPRESS_TYPE", models.TextField(default="")),
            ],
            options={
                "permissions": (
                    ("can_delete", "Delete Scans"),
                    ("can_scan", "Scan Files"),
                    ("can_suppress", "Suppress Findings"),
                ),
            },
        ),
    ]