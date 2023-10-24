import os
import requests
import json
from bs4 import BeautifulSoup


keywords = ['纯海洋背景', '纯树林背景', '纯丛林背景', '纯下雨背景', '纯色块背景']
base_url = "https://image.baidu.com/search/acjson?tn=resultjson_com&ipn=rj&ct=201326592&is=&fp=result&queryWord={}&cl=2&lm=-1&ie=utf-8&oe=utf-8&adpicid=&st=&z=&ic=&hd=&latest=&copyright=&word={}&s=&se=&tab=&width=&height=&face=&istype=&qc=&nc=&fr=&expermode=&force=&pn={}&rn=30"


folder_path = './background_image'
if not os.path.exists(folder_path):
    os.makedirs(folder_path)


count = 0

for keyword in keywords:
    for page in range(10):  # 搜索前10页的结果
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3"
        }

        response = requests.get(base_url.format(keyword, keyword, page * 30), headers=headers)
        # data = json.loads(response.text)
        try:
            data = response.json()
        except json.JSONDecodeError as e:
            print(f"Failed to decode JSON: {e}")
        # print(data)  # 打印整个 data 对象
        # print(data.keys())  # 打印所有的键值

        # 查找页面中的所有图片
        for img in data['data']:

            link = img.get('thumbURL')

            if link is not None:
                # 限制图片数量
                if count >= 100:
                    break

                img_data = requests.get(link).content
                with open(os.path.join(folder_path, 'background' + str(count+500) + '.jpg'), 'wb') as handler:
                    handler.write(img_data)

                count += 1

print(f"Downloaded {count} images.")