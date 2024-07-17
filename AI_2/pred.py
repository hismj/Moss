import pickle
import tempfile
import pandas as pd
from keras.models import load_model
from AI_2.permission_extractor import process_data


def predict(apk_path):
    # 加载特征选择信息
    with open('AI_2/features.pkl', 'rb') as f:
        important_features = pickle.load(f)

        # 获得预测数据路径
        apk_path_pred = process_data(apk_path)

        # 加载待预测的未处理数据
        new_data = pd.read_csv(apk_path_pred, sep=';')

        # 选取与训练时相同的重要特征
        X_pred = new_data[important_features].values

        # 加载训练好的模型
        model = load_model('AI_2/train.keras')

        # 对数据进行预测
        y_pred = model.predict(X_pred)
        y_pred = (y_pred > 0.6).astype(int)  # 假设阈值为 0.6

        # 返回预测结果的第一个值（假设每个APK文件只返回一个预测结果）
    return int(y_pred[0][0])
