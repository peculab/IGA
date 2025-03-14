from flask import Flask, request, jsonify
import pandas as pd
import joblib

# 載入僅包含分類器的模型（不含前處理步驟）
clf = joblib.load('clf_model.pkl')  # 請確保檔案名稱與路徑正確

app = Flask(__name__)

@app.route('/')
def index():
    return "Flask 伺服器運行中。請使用 /predict 來獲取預測結果！"

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # 從 POST 請求中讀取 JSON 資料
        data = request.get_json(force=True)
        # 如果傳入的是單一字典，包裝成列表
        if isinstance(data, dict):
            data = [data]
        # 將 JSON 轉換成 DataFrame
        # 注意：此 DataFrame 必須已經是數值化且欄位順序與訓練時一致
        df = pd.DataFrame(data)
        
        # 使用裸分類器進行預測（此處不會自動進行前處理）
        predictions = clf.predict(df)
        anomaly_scores = clf.predict_proba(df)[:, 1] * 100  # 將異常概率轉為百分比
        
        # 將預測結果構造成 JSON 回傳
        results = []
        for i in range(len(df)):
            results.append({
                "is_anomaly": int(predictions[i]),
                "anomaly_score": float(anomaly_scores[i])
            })
        return jsonify(results)
    
    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
