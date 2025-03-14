import requests

# API URL
url = "http://127.0.0.1:5000/predict"

# 假設你已經將原始資料數值化，並確保欄位順序如下：
# user_action, user_ip, user_account, IP, user_id, role_id, action, hour, minute, second, dayofweek, day_type
payload = [
    {
        "user_action": 263,   # 數值化後的編碼
        "user_ip": 80998,
        "user_account": 91,
        "IP": 13902,
        "user_id": 83,
        "role_id": 0,
        "action": 0,
        "hour": 9,
        "minute": 11,
        "second": 55,
        "dayofweek": 6,
        "day_type": 1  # 假設 1 表示 weekday, 0 表示 weekend/holiday
    },
    {
        "user_action": 198,   # 數值化後的編碼
        "user_ip": 59623,
        "user_account": 69,
        "IP": 13014,
        "user_id": 63,
        "role_id": 6,
        "action": 1,
        "hour": 0,
        "minute": 40,
        "second": 55,
        "dayofweek": 2,
        "day_type": 0  # 假設 1 表示 weekday, 0 表示 weekend/holiday
    },
    {
        "user_action": 11,   # 數值化後的編碼
        "user_ip": 1034,
        "user_account": 5,
        "IP": 10436,
        "user_id": 104,
        "role_id": 6,
        "action": 0,
        "hour": 2,
        "minute": 19,
        "second": 55,
        "dayofweek": 0,
        "day_type": 0  # 假設 1 表示 weekday, 0 表示 weekend/holiday
    },
    # 如果有多筆資料，可以繼續在這個列表中加入其他字典
]

# 發送 POST 請求
response = requests.post(url, json=payload)
print("預測結果：", response.json())
