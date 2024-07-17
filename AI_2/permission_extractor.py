import csv
import os

from androguard.core.apk import APK

# APK文件路径
apk_folder_path = 'data/train'
csv_file = "pred_data.csv"
# 指定的权限列表
def process_data(apk_filename):
    permissions_to_find = [
        "android",
        "android.app.cts.permission.TEST_GRANTED",
        "android.intent.category.MASTER_CLEAR.permission.C2D_MESSAGE",
        "android.os.cts.permission.TEST_GRANTED",
        "android.permission.ACCESS_ALL_DOWNLOADS",
        "android.permission.ACCESS_ALL_EXTERNAL_STORAGE",
        "android.permission.ACCESS_BLUETOOTH_SHARE",
        "android.permission.ACCESS_CACHE_FILESYSTEM",
        "android.permission.ACCESS_CHECKIN_PROPERTIES",
        "android.permission.ACCESS_COARSE_LOCATION",
        "android.permission.ACCESS_CONTENT_PROVIDERS_EXTERNALLY",
        "android.permission.ACCESS_DOWNLOAD_MANAGER",
        "android.permission.ACCESS_DOWNLOAD_MANAGER_ADVANCED",
        "android.permission.ACCESS_DRM_CERTIFICATES",
        "android.permission.ACCESS_FINE_LOCATION",
        "android.permission.ACCESS_FM_RADIO",
        "android.permission.ACCESS_INPUT_FLINGER",
        "android.permission.ACCESS_KEYGUARD_SECURE_STORAGE",
        "android.permission.ACCESS_LOCATION_EXTRA_COMMANDS",
        "android.permission.ACCESS_MOCK_LOCATION",
        "android.permission.ACCESS_MTP",
        "android.permission.ACCESS_NETWORK_CONDITIONS",
        "android.permission.ACCESS_NETWORK_STATE",
        "android.permission.ACCESS_NOTIFICATIONS",
        "android.permission.ACCESS_PDB_STATE",
        "android.permission.ACCESS_SURFACE_FLINGER",
        "android.permission.ACCESS_WIFI_STATE",
        "android.permission.ACCESS_WIMAX_STATE",
        "android.permission.ACCOUNT_MANAGER",
        "android.permission.ALLOW_ANY_CODEC_FOR_PLAYBACK",
        "android.permission.ASEC_ACCESS",
        "android.permission.ASEC_CREATE",
        "android.permission.ASEC_DESTROY",
        "android.permission.ASEC_MOUNT_UNMOUNT",
        "android.permission.ASEC_RENAME",
        "android.permission.AUTHENTICATE_ACCOUNTS",
        "android.permission.BACKUP",
        "android.permission.BATTERY_STATS",
        "android.permission.BIND_ACCESSIBILITY_SERVICE",
        "android.permission.BIND_APPWIDGET",
        "android.permission.BIND_CARRIER_MESSAGING_SERVICE",
        "android.permission.BIND_CONDITION_PROVIDER_SERVICE",
        "android.permission.BIND_CONNECTION_SERVICE",
        "android.permission.BIND_DEVICE_ADMIN",
        "android.permission.BIND_DIRECTORY_SEARCH",
        "android.permission.BIND_DREAM_SERVICE",
        "android.permission.BIND_INCALL_SERVICE",
        "android.permission.BIND_INPUT_METHOD",
        "android.permission.BIND_JOB_SERVICE",
        "android.permission.BIND_KEYGUARD_APPWIDGET",
        "android.permission.BIND_NFC_SERVICE",
        "android.permission.BIND_NOTIFICATION_LISTENER_SERVICE",
        "android.permission.BIND_PACKAGE_VERIFIER",
        "android.permission.BIND_PRINT_SERVICE",
        "android.permission.BIND_PRINT_SPOOLER_SERVICE",
        "android.permission.BIND_REMOTEVIEWS",
        "android.permission.BIND_REMOTE_DISPLAY",
        "android.permission.BIND_TEXT_SERVICE",
        "android.permission.BIND_TRUST_AGENT",
        "android.permission.BIND_TV_INPUT",
        "android.permission.BIND_VOICE_INTERACTION",
        "android.permission.BIND_VPN_SERVICE",
        "android.permission.BIND_WALLPAPER",
        "android.permission.BLUETOOTH",
        "android.permission.BLUETOOTH_ADMIN",
        "android.permission.BLUETOOTH_MAP",
        "android.permission.BLUETOOTH_PRIVILEGED",
        "android.permission.BLUETOOTH_STACK",
        "android.permission.BODY_SENSORS",
        "android.permission.BRICK",
        "android.permission.BROADCAST_CALLLOG_INFO",
        "android.permission.BROADCAST_NETWORK_PRIVILEGED",
        "android.permission.BROADCAST_PACKAGE_REMOVED",
        "android.permission.BROADCAST_SMS",
        "android.permission.BROADCAST_STICKY",
        "android.permission.BROADCAST_WAP_PUSH",
        "android.permission.CALL_PHONE",
        "android.permission.CALL_PRIVILEGED",
        "android.permission.CAMERA",
        "android.permission.CAMERA_DISABLE_TRANSMIT_LED",
        "android.permission.CAPTURE_AUDIO_HOTWORD",
        "android.permission.CAPTURE_AUDIO_OUTPUT",
        "android.permission.CAPTURE_SECURE_VIDEO_OUTPUT",
        "android.permission.CAPTURE_TV_INPUT",
        "android.permission.CAPTURE_VIDEO_OUTPUT",
        "android.permission.CARRIER_FILTER_SMS",
        "android.permission.CHANGE_BACKGROUND_DATA_SETTING",
        "android.permission.CHANGE_COMPONENT_ENABLED_STATE",
        "android.permission.CHANGE_CONFIGURATION",
        "android.permission.CHANGE_NETWORK_STATE",
        "android.permission.CHANGE_WIFI_MULTICAST_STATE",
        "android.permission.CHANGE_WIFI_STATE",
        "android.permission.CHANGE_WIMAX_STATE",
        "android.permission.CLEAR_APP_CACHE",
        "android.permission.CLEAR_APP_USER_DATA",
        "android.permission.CONFIGURE_WIFI_DISPLAY",
        "android.permission.CONFIRM_FULL_BACKUP",
        "android.permission.CONNECTIVITY_INTERNAL",
        "android.permission.CONTROL_INCALL_EXPERIENCE",
        "android.permission.CONTROL_KEYGUARD",
        "android.permission.CONTROL_LOCATION_UPDATES",
        "android.permission.CONTROL_VPN",
        "android.permission.CONTROL_WIFI_DISPLAY",
        "android.permission.COPY_PROTECTED_DATA",
        "android.permission.CRYPT_KEEPER",
        "android.permission.DELETE_CACHE_FILES",
        "android.permission.DELETE_PACKAGES",
        "android.permission.DEVICE_POWER",
        "android.permission.DIAGNOSTIC",
        "android.permission.DISABLE_KEYGUARD",
        "android.permission.DOWNLOAD_CACHE_NON_PURGEABLE",
        "android.permission.DOWNLOAD_WITHOUT_NOTIFICATION",
        "android.permission.DUMP",
        "android.permission.EXPAND_STATUS_BAR",
        "android.permission.FACTORY_TEST",
        "android.permission.FILTER_EVENTS",
        "android.permission.FLASHLIGHT",
        "android.permission.FORCE_BACK",
        "android.permission.FORCE_STOP_PACKAGES",
        "android.permission.FRAME_STATS",
        "android.permission.FREEZE_SCREEN",
        "android.permission.GET_ACCOUNTS",
        "android.permission.GET_APP_OPS_STATS",
        "android.permission.GET_DETAILED_TASKS",
        "android.permission.GET_PACKAGE_SIZE",
        "android.permission.GET_TASKS",
        "android.permission.GET_TOP_ACTIVITY_INFO",
        "android.permission.GLOBAL_SEARCH",
        "android.permission.GLOBAL_SEARCH_CONTROL",
        "android.permission.GRANT_REVOKE_PERMISSIONS",
        "android.permission.HARDWARE_TEST",
        "android.permission.HDMI_CEC",
        "android.permission.INJECT_EVENTS",
        "android.permission.INSTALL_LOCATION_PROVIDER",
        "android.permission.INSTALL_PACKAGES",
        "android.permission.INTERACT_ACROSS_USERS",
        "android.permission.INTERACT_ACROSS_USERS_FULL",
        "android.permission.INTERNAL_SYSTEM_WINDOW",
        "android.permission.INTERNET",
        "android.permission.INVOKE_CARRIER_SETUP",
        "android.permission.KILL_BACKGROUND_PROCESSES",
        "android.permission.LAUNCH_TRUST_AGENT_SETTINGS",
        "android.permission.LOCATION_HARDWARE",
        "android.permission.LOOP_RADIO",
        "android.permission.MANAGE_ACCOUNTS",
        "android.permission.MANAGE_ACTIVITY_STACKS",
        "android.permission.MANAGE_APP_TOKENS",
        "android.permission.MANAGE_CA_CERTIFICATES",
        "android.permission.MANAGE_DEVICE_ADMINS",
        "android.permission.MANAGE_DOCUMENTS",
        "android.permission.MANAGE_MEDIA_PROJECTION",
        "android.permission.MANAGE_NETWORK_POLICY",
        "android.permission.MANAGE_USB",
        "android.permission.MANAGE_USERS",
        "android.permission.MANAGE_VOICE_KEYPHRASES",
        "android.permission.MASTER_CLEAR",
        "android.permission.MEDIA_CONTENT_CONTROL",
        "android.permission.MMS_SEND_OUTBOX_MSG",
        "android.permission.MODIFY_APPWIDGET_BIND_PERMISSIONS",
        "android.permission.MODIFY_AUDIO_ROUTING",
        "android.permission.MODIFY_AUDIO_SETTINGS",
        "android.permission.MODIFY_NETWORK_ACCOUNTING",
        "android.permission.MODIFY_PARENTAL_CONTROLS",
        "android.permission.MODIFY_PHONE_STATE",
        "android.permission.MOUNT_FORMAT_FILESYSTEMS",
        "android.permission.MOUNT_UNMOUNT_FILESYSTEMS",
        "android.permission.MOVE_PACKAGE",
        "android.permission.NET_ADMIN",
        "android.permission.NET_TUNNELING",
        "android.permission.NFC",
        "android.permission.NFC_HANDOVER_STATUS",
        "android.permission.OEM_UNLOCK_STATE",
        "android.permission.PACKAGE_USAGE_STATS",
        "android.permission.PACKAGE_VERIFICATION_AGENT",
        "android.permission.PERFORM_CDMA_PROVISIONING",
        "android.permission.PERSISTENT_ACTIVITY",
        "android.permission.PROCESS_CALLLOG_INFO",
        "android.permission.PROCESS_OUTGOING_CALLS",
        "android.permission.PROVIDE_TRUST_AGENT",
        "android.permission.READ_CALENDAR",
        "android.permission.READ_CALL_LOG",
        "android.permission.READ_CELL_BROADCASTS",
        "android.permission.READ_CONTACTS",
        "android.permission.READ_DREAM_STATE",
        "android.permission.READ_EXTERNAL_STORAGE",
        "android.permission.READ_FRAME_BUFFER",
        "android.permission.READ_INPUT_STATE",
        "android.permission.READ_INSTALL_SESSIONS",
        "android.permission.READ_LOGS",
        "android.permission.READ_NETWORK_USAGE_HISTORY",
        "android.permission.READ_PHONE_STATE",
        "android.permission.READ_PRECISE_PHONE_STATE",
        "android.permission.READ_PRIVILEGED_PHONE_STATE",
        "android.permission.READ_PROFILE",
        "android.permission.READ_SEARCH_INDEXABLES",
        "android.permission.READ_SMS",
        "android.permission.READ_SOCIAL_STREAM",
        "android.permission.READ_SYNC_SETTINGS",
        "android.permission.READ_SYNC_STATS",
        "android.permission.READ_USER_DICTIONARY",
        "android.permission.READ_WIFI_CREDENTIAL",
        "android.permission.REAL_GET_TASKS",
        "android.permission.REBOOT",
        "android.permission.RECEIVE_BLUETOOTH_MAP",
        "android.permission.RECEIVE_BOOT_COMPLETED",
        "android.permission.RECEIVE_DATA_ACTIVITY_CHANGE",
        "android.permission.RECEIVE_EMERGENCY_BROADCAST",
        "android.permission.RECEIVE_MMS",
        "android.permission.RECEIVE_SMS",
        "android.permission.RECEIVE_WAP_PUSH",
        "android.permission.RECORD_AUDIO",
        "android.permission.RECOVERY",
        "android.permission.REGISTER_CALL_PROVIDER",
        "android.permission.REGISTER_CONNECTION_MANAGER",
        "android.permission.REGISTER_SIM_SUBSCRIPTION",
        "android.permission.REMOTE_AUDIO_PLAYBACK",
        "android.permission.REMOVE_DRM_CERTIFICATES",
        "android.permission.REMOVE_TASKS",
        "android.permission.REORDER_TASKS",
        "android.permission.RESTART_PACKAGES",
        "android.permission.RETRIEVE_WINDOW_CONTENT",
        "android.permission.RETRIEVE_WINDOW_TOKEN",
        "android.permission.SCORE_NETWORKS",
        "android.permission.SEND_DOWNLOAD_COMPLETED_INTENTS", "android.permission.SEND_RESPOND_VIA_MESSAGE",
        "android.permission.SEND_SMS",
        "android.permission.SERIAL_PORT",
        "android.permission.SET_ACTIVITY_WATCHER",
        "android.permission.SET_ALWAYS_FINISH",
        "android.permission.SET_ANIMATION_SCALE",
        "android.permission.SET_DEBUG_APP",
        "android.permission.SET_INPUT_CALIBRATION",
        "android.permission.SET_KEYBOARD_LAYOUT",
        "android.permission.SET_ORIENTATION",
        "android.permission.SET_POINTER_SPEED",
        "android.permission.SET_PREFERRED_APPLICATIONS",
        "android.permission.SET_PROCESS_LIMIT",
        "android.permission.SET_SCREEN_COMPATIBILITY",
        "android.permission.SET_TIME",
        "android.permission.SET_TIME_ZONE",
        "android.permission.SET_WALLPAPER",
        "android.permission.SET_WALLPAPER_COMPONENT",
        "android.permission.SET_WALLPAPER_HINTS",
        "android.permission.SHUTDOWN",
        "android.permission.SIGNAL_PERSISTENT_PROCESSES",
        "android.permission.START_ANY_ACTIVITY",
        "android.permission.START_PRINT_SERVICE_CONFIG_ACTIVITY",
        "android.permission.START_TASKS_FROM_RECENTS",
        "android.permission.STATUS_BAR",
        "android.permission.STATUS_BAR_SERVICE",
        "android.permission.STOP_APP_SWITCHES",
        "android.permission.SUBSCRIBED_FEEDS_READ",
        "android.permission.SUBSCRIBED_FEEDS_WRITE",
        "android.permission.SYSTEM_ALERT_WINDOW",
        "android.permission.TEMPORARY_ENABLE_ACCESSIBILITY",
        "android.permission.TRANSMIT_IR",
        "android.permission.TRUST_LISTENER",
        "android.permission.TV_INPUT_HARDWARE",
        "android.permission.UPDATE_APP_OPS_STATS",
        "android.permission.UPDATE_DEVICE_STATS",
        "android.permission.UPDATE_LOCK",
        "android.permission.USER_ACTIVITY",
        "android.permission.USE_CREDENTIALS",
        "android.permission.USE_SIP",
        "android.permission.VIBRATE",
        "android.permission.WAKE_LOCK",
        "android.permission.WRITE_APN_SETTINGS",
        "android.permission.WRITE_CALENDAR",
        "android.permission.WRITE_CALL_LOG",
        "android.permission.WRITE_CONTACTS",
        "android.permission.WRITE_DREAM_STATE",
        "android.permission.WRITE_EXTERNAL_STORAGE",
        "android.permission.WRITE_GSERVICES",
        "android.permission.WRITE_MEDIA_STORAGE",
        "android.permission.WRITE_PROFILE",
        "android.permission.WRITE_SECURE_SETTINGS",
        "android.permission.WRITE_SETTINGS",
        "android.permission.WRITE_SMS",
        "android.permission.WRITE_SOCIAL_STREAM",
        "android.permission.WRITE_SYNC_SETTINGS",
        "android.permission.WRITE_USER_DICTIONARY",
        "com.android.alarm.permission.SET_ALARM",
        "com.android.browser.permission.PRELOAD",
        "com.android.browser.permission.READ_HISTORY_BOOKMARKS",
        "com.android.browser.permission.WRITE_HISTORY_BOOKMARKS",
        "com.android.certinstaller.INSTALL_AS_USER",
        "com.android.cts.intent.sender.permission.SAMPLE",
        "com.android.cts.keysets_permdef.keysets_perm",
        "com.android.cts.permissionAllowedWithSignature",
        "com.android.cts.permissionNormal",
        "com.android.cts.permissionNotUsedWithSignature",
        "com.android.cts.permissionWithSignature",
        "com.android.email.permission.ACCESS_PROVIDER",
        "com.android.email.permission.READ_ATTACHMENT",
        "com.android.frameworks.coretests.DANGEROUS",
        "com.android.frameworks.coretests.NORMAL",
        "com.android.frameworks.coretests.SIGNATURE",
        "com.android.frameworks.coretests.keysets_permdef.keyset_perm",
        "com.android.frameworks.coretests.permission.TEST_DENIED",
        "com.android.frameworks.coretests.permission.TEST_GRANTED",
        "com.android.gallery3d.filtershow.permission.READ",
        "com.android.gallery3d.filtershow.permission.WRITE",
        "com.android.gallery3d.permission.GALLERY_PROVIDER",
        "com.android.launcher.permission.INSTALL_SHORTCUT",
        "com.android.launcher.permission.PRELOAD_WORKSPACE",
        "com.android.launcher.permission.READ_SETTINGS",
        "com.android.launcher.permission.UNINSTALL_SHORTCUT",
        "com.android.launcher.permission.WRITE_SETTINGS",
        "com.android.launcher3.permission.READ_SETTINGS",
        "com.android.launcher3.permission.RECEIVE_FIRST_LOAD_BROADCAST",
        "com.android.launcher3.permission.RECEIVE_LAUNCH_BROADCASTS",
        "com.android.launcher3.permission.WRITE_SETTINGS",
        "com.android.permission.WHITELIST_BLUETOOTH_DEVICE",
        "com.android.printspooler.permission.ACCESS_ALL_PRINT_JOBS",
        "com.android.providers.tv.permission.ACCESS_ALL_EPG_DATA",
        "com.android.providers.tv.permission.ACCESS_WATCHED_PROGRAMS",
        "com.android.providers.tv.permission.READ_EPG_DATA",
        "com.android.providers.tv.permission.WRITE_EPG_DATA",
        "com.android.smspush.WAPPUSH_MANAGER_BIND",
        "com.android.voicemail.permission.ADD_VOICEMAIL",
        "com.android.voicemail.permission.READ_VOICEMAIL",
        "com.android.voicemail.permission.WRITE_VOICEMAIL",
        "com.foo.mypermission",
        "com.foo.mypermission2",
        "org.chromium.chrome.shell.permission.C2D_MESSAGE",
        "org.chromium.chrome.shell.permission.DEBUG",
        "org.chromium.chrome.shell.permission.SANDBOX",
        "org.chromium.chromecast.shell.permission.SANDBOX",
        "org.chromium.content_shell.permission.SANDBOX",
        "test_permission",
    ]

    # 打开CSV文件准备写入
    with open(csv_file, mode='w', newline='') as file:
        writer = csv.writer(file)

        # 写入表头
        writer.writerow(
            ["android;android.app.cts.permission.TEST_GRANTED;android.intent.category.MASTER_CLEAR.permission"
             ".C2D_MESSAGE;android.os.cts.permission.TEST_GRANTED;android.permission.ACCESS_ALL_DOWNLOADS"
             ";android.permission.ACCESS_ALL_EXTERNAL_STORAGE;android.permission.ACCESS_BLUETOOTH_SHARE"
             ";android.permission.ACCESS_CACHE_FILESYSTEM;android.permission.ACCESS_CHECKIN_PROPERTIES"
             ";android.permission.ACCESS_COARSE_LOCATION;android.permission"
             ".ACCESS_CONTENT_PROVIDERS_EXTERNALLY;android.permission.ACCESS_DOWNLOAD_MANAGER;android"
             ".permission.ACCESS_DOWNLOAD_MANAGER_ADVANCED;android.permission.ACCESS_DRM_CERTIFICATES;android"
             ".permission.ACCESS_FINE_LOCATION;android.permission.ACCESS_FM_RADIO;android.permission"
             ".ACCESS_INPUT_FLINGER;android.permission.ACCESS_KEYGUARD_SECURE_STORAGE;android.permission"
             ".ACCESS_LOCATION_EXTRA_COMMANDS;android.permission.ACCESS_MOCK_LOCATION;android.permission"
             ".ACCESS_MTP;android.permission.ACCESS_NETWORK_CONDITIONS;android.permission"
             ".ACCESS_NETWORK_STATE;android.permission.ACCESS_NOTIFICATIONS;android.permission"
             ".ACCESS_PDB_STATE;android.permission.ACCESS_SURFACE_FLINGER;android.permission"
             ".ACCESS_WIFI_STATE;android.permission.ACCESS_WIMAX_STATE;android.permission.ACCOUNT_MANAGER"
             ";android.permission.ALLOW_ANY_CODEC_FOR_PLAYBACK;android.permission.ASEC_ACCESS;android"
             ".permission.ASEC_CREATE;android.permission.ASEC_DESTROY;android.permission.ASEC_MOUNT_UNMOUNT"
             ";android.permission.ASEC_RENAME;android.permission.AUTHENTICATE_ACCOUNTS;android.permission"
             ".BACKUP;android.permission.BATTERY_STATS;android.permission.BIND_ACCESSIBILITY_SERVICE;android"
             ".permission.BIND_APPWIDGET;android.permission.BIND_CARRIER_MESSAGING_SERVICE;android.permission"
             ".BIND_CONDITION_PROVIDER_SERVICE;android.permission.BIND_CONNECTION_SERVICE;android.permission"
             ".BIND_DEVICE_ADMIN;android.permission.BIND_DIRECTORY_SEARCH;android.permission"
             ".BIND_DREAM_SERVICE;android.permission.BIND_INCALL_SERVICE;android.permission.BIND_INPUT_METHOD"
             ";android.permission.BIND_JOB_SERVICE;android.permission.BIND_KEYGUARD_APPWIDGET;android"
             ".permission.BIND_NFC_SERVICE;android.permission.BIND_NOTIFICATION_LISTENER_SERVICE;android"
             ".permission.BIND_PACKAGE_VERIFIER;android.permission.BIND_PRINT_SERVICE;android.permission"
             ".BIND_PRINT_SPOOLER_SERVICE;android.permission.BIND_REMOTEVIEWS;android.permission"
             ".BIND_REMOTE_DISPLAY;android.permission.BIND_TEXT_SERVICE;android.permission.BIND_TRUST_AGENT"
             ";android.permission.BIND_TV_INPUT;android.permission.BIND_VOICE_INTERACTION;android.permission"
             ".BIND_VPN_SERVICE;android.permission.BIND_WALLPAPER;android.permission.BLUETOOTH;android"
             ".permission.BLUETOOTH_ADMIN;android.permission.BLUETOOTH_MAP;android.permission"
             ".BLUETOOTH_PRIVILEGED;android.permission.BLUETOOTH_STACK;android.permission.BODY_SENSORS"
             ";android.permission.BRICK;android.permission.BROADCAST_CALLLOG_INFO;android.permission"
             ".BROADCAST_NETWORK_PRIVILEGED;android.permission.BROADCAST_PACKAGE_REMOVED;android.permission"
             ".BROADCAST_SMS;android.permission.BROADCAST_STICKY;android.permission.BROADCAST_WAP_PUSH"
             ";android.permission.CALL_PHONE;android.permission.CALL_PRIVILEGED;android.permission.CAMERA"
             ";android.permission.CAMERA_DISABLE_TRANSMIT_LED;android.permission.CAPTURE_AUDIO_HOTWORD"
             ";android.permission.CAPTURE_AUDIO_OUTPUT;android.permission.CAPTURE_SECURE_VIDEO_OUTPUT;android"
             ".permission.CAPTURE_TV_INPUT;android.permission.CAPTURE_VIDEO_OUTPUT;android.permission"
             ".CARRIER_FILTER_SMS;android.permission.CHANGE_BACKGROUND_DATA_SETTING;android.permission"
             ".CHANGE_COMPONENT_ENABLED_STATE;android.permission.CHANGE_CONFIGURATION;android.permission"
             ".CHANGE_NETWORK_STATE;android.permission.CHANGE_WIFI_MULTICAST_STATE;android.permission"
             ".CHANGE_WIFI_STATE;android.permission.CHANGE_WIMAX_STATE;android.permission.CLEAR_APP_CACHE"
             ";android.permission.CLEAR_APP_USER_DATA;android.permission.CONFIGURE_WIFI_DISPLAY;android"
             ".permission.CONFIRM_FULL_BACKUP;android.permission.CONNECTIVITY_INTERNAL;android.permission"
             ".CONTROL_INCALL_EXPERIENCE;android.permission.CONTROL_KEYGUARD;android.permission"
             ".CONTROL_LOCATION_UPDATES;android.permission.CONTROL_VPN;android.permission"
             ".CONTROL_WIFI_DISPLAY;android.permission.COPY_PROTECTED_DATA;android.permission.CRYPT_KEEPER"
             ";android.permission.DELETE_CACHE_FILES;android.permission.DELETE_PACKAGES;android.permission"
             ".DEVICE_POWER;android.permission.DIAGNOSTIC;android.permission.DISABLE_KEYGUARD;android"
             ".permission.DOWNLOAD_CACHE_NON_PURGEABLE;android.permission.DOWNLOAD_WITHOUT_NOTIFICATION"
             ";android.permission.DUMP;android.permission.EXPAND_STATUS_BAR;android.permission.FACTORY_TEST"
             ";android.permission.FILTER_EVENTS;android.permission.FLASHLIGHT;android.permission.FORCE_BACK"
             ";android.permission.FORCE_STOP_PACKAGES;android.permission.FRAME_STATS;android.permission"
             ".FREEZE_SCREEN;android.permission.GET_ACCOUNTS;android.permission.GET_APP_OPS_STATS;android"
             ".permission.GET_DETAILED_TASKS;android.permission.GET_PACKAGE_SIZE;android.permission.GET_TASKS"
             ";android.permission.GET_TOP_ACTIVITY_INFO;android.permission.GLOBAL_SEARCH;android.permission"
             ".GLOBAL_SEARCH_CONTROL;android.permission.GRANT_REVOKE_PERMISSIONS;android.permission"
             ".HARDWARE_TEST;android.permission.HDMI_CEC;android.permission.INJECT_EVENTS;android.permission"
             ".INSTALL_LOCATION_PROVIDER;android.permission.INSTALL_PACKAGES;android.permission"
             ".INTERACT_ACROSS_USERS;android.permission.INTERACT_ACROSS_USERS_FULL;android.permission"
             ".INTERNAL_SYSTEM_WINDOW;android.permission.INTERNET;android.permission.INVOKE_CARRIER_SETUP"
             ";android.permission.KILL_BACKGROUND_PROCESSES;android.permission.LAUNCH_TRUST_AGENT_SETTINGS"
             ";android.permission.LOCATION_HARDWARE;android.permission.LOOP_RADIO;android.permission"
             ".MANAGE_ACCOUNTS;android.permission.MANAGE_ACTIVITY_STACKS;android.permission.MANAGE_APP_TOKENS"
             ";android.permission.MANAGE_CA_CERTIFICATES;android.permission.MANAGE_DEVICE_ADMINS;android"
             ".permission.MANAGE_DOCUMENTS;android.permission.MANAGE_MEDIA_PROJECTION;android.permission"
             ".MANAGE_NETWORK_POLICY;android.permission.MANAGE_USB;android.permission.MANAGE_USERS;android"
             ".permission.MANAGE_VOICE_KEYPHRASES;android.permission.MASTER_CLEAR;android.permission"
             ".MEDIA_CONTENT_CONTROL;android.permission.MMS_SEND_OUTBOX_MSG;android.permission"
             ".MODIFY_APPWIDGET_BIND_PERMISSIONS;android.permission.MODIFY_AUDIO_ROUTING;android.permission"
             ".MODIFY_AUDIO_SETTINGS;android.permission.MODIFY_NETWORK_ACCOUNTING;android.permission"
             ".MODIFY_PARENTAL_CONTROLS;android.permission.MODIFY_PHONE_STATE;android.permission"
             ".MOUNT_FORMAT_FILESYSTEMS;android.permission.MOUNT_UNMOUNT_FILESYSTEMS;android.permission"
             ".MOVE_PACKAGE;android.permission.NET_ADMIN;android.permission.NET_TUNNELING;android.permission"
             ".NFC;android.permission.NFC_HANDOVER_STATUS;android.permission.OEM_UNLOCK_STATE;android"
             ".permission.PACKAGE_USAGE_STATS;android.permission.PACKAGE_VERIFICATION_AGENT;android"
             ".permission.PERFORM_CDMA_PROVISIONING;android.permission.PERSISTENT_ACTIVITY;android.permission"
             ".PROCESS_CALLLOG_INFO;android.permission.PROCESS_OUTGOING_CALLS;android.permission"
             ".PROVIDE_TRUST_AGENT;android.permission.READ_CALENDAR;android.permission.READ_CALL_LOG;android"
             ".permission.READ_CELL_BROADCASTS;android.permission.READ_CONTACTS;android.permission"
             ".READ_DREAM_STATE;android.permission.READ_EXTERNAL_STORAGE;android.permission.READ_FRAME_BUFFER"
             ";android.permission.READ_INPUT_STATE;android.permission.READ_INSTALL_SESSIONS;android"
             ".permission.READ_LOGS;android.permission.READ_NETWORK_USAGE_HISTORY;android.permission"
             ".READ_PHONE_STATE;android.permission.READ_PRECISE_PHONE_STATE;android.permission"
             ".READ_PRIVILEGED_PHONE_STATE;android.permission.READ_PROFILE;android.permission"
             ".READ_SEARCH_INDEXABLES;android.permission.READ_SMS;android.permission.READ_SOCIAL_STREAM"
             ";android.permission.READ_SYNC_SETTINGS;android.permission.READ_SYNC_STATS;android.permission"
             ".READ_USER_DICTIONARY;android.permission.READ_WIFI_CREDENTIAL;android.permission.REAL_GET_TASKS"
             ";android.permission.REBOOT;android.permission.RECEIVE_BLUETOOTH_MAP;android.permission"
             ".RECEIVE_BOOT_COMPLETED;android.permission.RECEIVE_DATA_ACTIVITY_CHANGE;android.permission"
             ".RECEIVE_EMERGENCY_BROADCAST;android.permission.RECEIVE_MMS;android.permission.RECEIVE_SMS"
             ";android.permission.RECEIVE_WAP_PUSH;android.permission.RECORD_AUDIO;android.permission"
             ".RECOVERY;android.permission.REGISTER_CALL_PROVIDER;android.permission"
             ".REGISTER_CONNECTION_MANAGER;android.permission.REGISTER_SIM_SUBSCRIPTION;android.permission"
             ".REMOTE_AUDIO_PLAYBACK;android.permission.REMOVE_DRM_CERTIFICATES;android.permission"
             ".REMOVE_TASKS;android.permission.REORDER_TASKS;android.permission.RESTART_PACKAGES;android"
             ".permission.RETRIEVE_WINDOW_CONTENT;android.permission.RETRIEVE_WINDOW_TOKEN;android.permission"
             ".SCORE_NETWORKS;android.permission.SEND_DOWNLOAD_COMPLETED_INTENTS;android.permission"
             ".SEND_RESPOND_VIA_MESSAGE;android.permission.SEND_SMS;android.permission.SERIAL_PORT;android"
             ".permission.SET_ACTIVITY_WATCHER;android.permission.SET_ALWAYS_FINISH;android.permission"
             ".SET_ANIMATION_SCALE;android.permission.SET_DEBUG_APP;android.permission.SET_INPUT_CALIBRATION"
             ";android.permission.SET_KEYBOARD_LAYOUT;android.permission.SET_ORIENTATION;android.permission"
             ".SET_POINTER_SPEED;android.permission.SET_PREFERRED_APPLICATIONS;android.permission"
             ".SET_PROCESS_LIMIT;android.permission.SET_SCREEN_COMPATIBILITY;android.permission.SET_TIME"
             ";android.permission.SET_TIME_ZONE;android.permission.SET_WALLPAPER;android.permission"
             ".SET_WALLPAPER_COMPONENT;android.permission.SET_WALLPAPER_HINTS;android.permission.SHUTDOWN"
             ";android.permission.SIGNAL_PERSISTENT_PROCESSES;android.permission.START_ANY_ACTIVITY;android"
             ".permission.START_PRINT_SERVICE_CONFIG_ACTIVITY;android.permission.START_TASKS_FROM_RECENTS"
             ";android.permission.STATUS_BAR;android.permission.STATUS_BAR_SERVICE;android.permission"
             ".STOP_APP_SWITCHES;android.permission.SUBSCRIBED_FEEDS_READ;android.permission"
             ".SUBSCRIBED_FEEDS_WRITE;android.permission.SYSTEM_ALERT_WINDOW;android.permission"
             ".TEMPORARY_ENABLE_ACCESSIBILITY;android.permission.TRANSMIT_IR;android.permission"
             ".TRUST_LISTENER;android.permission.TV_INPUT_HARDWARE;android.permission.UPDATE_APP_OPS_STATS"
             ";android.permission.UPDATE_DEVICE_STATS;android.permission.UPDATE_LOCK;android.permission"
             ".USER_ACTIVITY;android.permission.USE_CREDENTIALS;android.permission.USE_SIP;android.permission"
             ".VIBRATE;android.permission.WAKE_LOCK;android.permission.WRITE_APN_SETTINGS;android.permission"
             ".WRITE_CALENDAR;android.permission.WRITE_CALL_LOG;android.permission.WRITE_CONTACTS;android"
             ".permission.WRITE_DREAM_STATE;android.permission.WRITE_EXTERNAL_STORAGE;android.permission"
             ".WRITE_GSERVICES;android.permission.WRITE_MEDIA_STORAGE;android.permission.WRITE_PROFILE"
             ";android.permission.WRITE_SECURE_SETTINGS;android.permission.WRITE_SETTINGS;android.permission"
             ".WRITE_SMS;android.permission.WRITE_SOCIAL_STREAM;android.permission.WRITE_SYNC_SETTINGS"
             ";android.permission.WRITE_USER_DICTIONARY;com.android.alarm.permission.SET_ALARM;com.android"
             ".browser.permission.PRELOAD;com.android.browser.permission.READ_HISTORY_BOOKMARKS;com.android"
             ".browser.permission.WRITE_HISTORY_BOOKMARKS;com.android.certinstaller.INSTALL_AS_USER;com"
             ".android.cts.intent.sender.permission.SAMPLE;com.android.cts.keysets_permdef.keysets_perm;com"
             ".android.cts.permissionAllowedWithSignature;com.android.cts.permissionNormal;com.android.cts"
             ".permissionNotUsedWithSignature;com.android.cts.permissionWithSignature;com.android.email"
             ".permission.ACCESS_PROVIDER;com.android.email.permission.READ_ATTACHMENT;com.android.frameworks"
             ".coretests.DANGEROUS;com.android.frameworks.coretests.NORMAL;com.android.frameworks.coretests"
             ".SIGNATURE;com.android.frameworks.coretests.keysets_permdef.keyset_perm;com.android.frameworks"
             ".coretests.permission.TEST_DENIED;com.android.frameworks.coretests.permission.TEST_GRANTED;com"
             ".android.gallery3d.filtershow.permission.READ;com.android.gallery3d.filtershow.permission.WRITE"
             ";com.android.gallery3d.permission.GALLERY_PROVIDER;com.android.launcher.permission"
             ".INSTALL_SHORTCUT;com.android.launcher.permission.PRELOAD_WORKSPACE;com.android.launcher"
             ".permission.READ_SETTINGS;com.android.launcher.permission.UNINSTALL_SHORTCUT;com.android"
             ".launcher.permission.WRITE_SETTINGS;com.android.launcher3.permission.READ_SETTINGS;com.android"
             ".launcher3.permission.RECEIVE_FIRST_LOAD_BROADCAST;com.android.launcher3.permission"
             ".RECEIVE_LAUNCH_BROADCASTS;com.android.launcher3.permission.WRITE_SETTINGS;com.android"
             ".permission.WHITELIST_BLUETOOTH_DEVICE;com.android.printspooler.permission"
             ".ACCESS_ALL_PRINT_JOBS;com.android.providers.tv.permission.ACCESS_ALL_EPG_DATA;com.android"
             ".providers.tv.permission.ACCESS_WATCHED_PROGRAMS;com.android.providers.tv.permission"
             ".READ_EPG_DATA;com.android.providers.tv.permission.WRITE_EPG_DATA;com.android.smspush"
             ".WAPPUSH_MANAGER_BIND;com.android.voicemail.permission.ADD_VOICEMAIL;com.android.voicemail"
             ".permission.READ_VOICEMAIL;com.android.voicemail.permission.WRITE_VOICEMAIL;com.foo"
             ".mypermission;com.foo.mypermission2;org.chromium.chrome.shell.permission.C2D_MESSAGE;org"
             ".chromium.chrome.shell.permission.DEBUG;org.chromium.chrome.shell.permission.SANDBOX;org"
             ".chromium.chromecast.shell.permission.SANDBOX;org.chromium.content_shell.permission.SANDBOX"
             ";test_permission;type"])

        # 遍历文件夹中的所有APK文件
        '''for apk_filename in os.listdir(apk_folder_path):
            if apk_filename.endswith(".apk"):
                apk_path = os.path.join(apk_folder_path, apk_filename)

                try:
                    # 加载APK文件
                    apk = APK(apk_path)

                    # 获取APK文件中的所有权限
                    all_permissions = apk.get_permissions()

                    # 准备存储每个权限结果的列表
                    results = []

                    # 检查每个权限是否在APK中
                    for permission in permissions_to_find:
                        if permission in all_permissions:
                            results.append("1")
                        else:
                            results.append("0")

                    # 将结果列表转换为分号分隔的字符串
                    results_str = ";".join(results)

                    # 将结果行写入CSV文件，将apk_filename替换为"0"
                    writer.writerow([results_str])

                except Exception as e:
                    print(f"Failed to process {apk_filename}: {e}")'''
        if apk_filename.endswith(".apk"):
            apk_path = apk_filename

            try:
                # 加载APK文件
                apk = APK(apk_path)

                # 获取APK文件中的所有权限
                all_permissions = apk.get_permissions()

                # 准备存储每个权限结果的列表
                results = []

                # 检查每个权限是否在APK中
                for permission in permissions_to_find:
                    if permission in all_permissions:
                        results.append("1")
                    else:
                        results.append("0")

                # 将结果列表转换为分号分隔的字符串
                results_str = ";".join(results)

                # 将结果行写入CSV文件，将apk_filename替换为"0"
                writer.writerow([results_str])

            except Exception as e:
                print(f"Failed to process {apk_filename}: {e}")
    return csv_file
    print(f"权限结果已保存到 {csv_file}")

