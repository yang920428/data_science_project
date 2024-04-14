# from selenium import webdriver
# from selenium.webdriver.chrome.service import Service
# from selenium.webdriver.chrome.options import Options
# from selenium.webdriver.common.by import By
# from selenium.webdriver.support.ui import Select
# import time

# # 設定下載檔案的路徑
# folder = "GaoXiong"
# download_path = "D:/code_sets/ds_bigproject/data_set/raw_data/" + folder

# # 設定 Chrome WebDriver 選項
# options = Options()
# options.add_experimental_option("prefs", {
#   "download.default_directory": download_path,
#   "download.prompt_for_download": False,
#   "download.directory_upgrade": True,
#   "safebrowsing.enabled": True
# })

# # 初始化 WebDriver in origin
# # service = Service("D:/user/chromedriver_win32/chromedriver.exe")
# # driver = webdriver.Chrome(service=service)
# # driver.get("https://codis.cwa.gov.tw/StationData")
# # element = driver.find_element(By.XPATH, '//*[@id="auto_C0"]')
# # element.click()


# # # 初始化 WebDriver by changed
# service = Service("D:/user/chromedriver_win32/chromedriver.exe")
# driver = webdriver.Chrome(service=service, options=options)
# driver.get("https://codis.cwa.gov.tw/StationData")
# time.sleep(1)
# element = driver.find_element(By.XPATH, '//*[@id="auto_C0"]')
# element.click()



# # 定位下拉選單元素
# select_element = driver.find_element(By.XPATH, '//*[@id="station_area"]')
# # 創建Select對象
# select = Select(select_element)
# # 選擇下拉選單的一個選項，這裡以選擇第一個選項為例
# select.select_by_index(15)  # 通過索引選擇

# # 定位到 <input> 元素
# input_element = driver.find_element(By.XPATH, '/html/body/div/main/div/div/div/div/aside/div/div[1]/div/div/section/ul/li[5]/div/div[2]/div/input')
# # 向 <input> 元素輸入文本
# input_element.send_keys("高雄 (467441)")

# # 定位到按鈕元素
# button_element = driver.find_element(By.XPATH, '/html/body/div[1]/main/div/div/section[1]/div[1]/div[3]/div[1]/div[1]/div[11]/div/div/div/div[2]')
# # 點擊按鈕
# button_element.click()
# # 定位到按鈕元素
# button_element = driver.find_element(By.XPATH, '/html/body/div[1]/main/div/div/section[1]/div[1]/div[3]/div[1]/div[1]/div[6]/div/div[1]/div/button')
# # 點擊按鈕
# button_element.click()

# # 定位到按鈕元素
# button_XPATH = "/html/body/div[1]/main/div/div/section[2]/div/div/section/div[5]/div[1]/div[2]/div"
# button_element = driver.find_element(By.XPATH, button_XPATH)  # 使用變量而不是字符串

# # 點擊按鈕
# button_element.click()

# button2_XPATH = '/html/body/div[1]/main/div/div/section[2]/div/div/section/div[5]/div[1]/div[1]/label/div/div[1]'
# button2_element = driver.find_element(By.XPATH, button2_XPATH)  # 使用變量而不是字符串

# # 點擊按鈕
# for i in range(0, 100):
#     print(f"進行中：{i+1}/100")  # 顯示進度
#     button2_element.click()
#     try:
#         button_element = WebDriverWait(driver, 5).until(
#             EC.presence_of_element_located((By.XPATH, button_XPATH))
#         )
#     except:
#         print("找不到指定的元素。")
#     button_element.click()
#     time.sleep(1)

# for i in range(0, 100):
#     print(f"進行中：{i+1}/100")  # 顯示進度
#     button2_element.click()
#     time.sleep(5)
#     button_element.click()
#     time.sleep(1)

# # 關閉 WebDriver
# driver.quit()


from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import Select
import time

# 設定下載檔案的路徑
folder = "GaoXiong"
download_path = "D:/code_sets/ds_bigproject/data_set/raw_data/" + folder

# 設定 Chrome WebDriver 選項
options = Options()
options.add_experimental_option("prefs", {
  "download.default_directory": download_path,
  "download.prompt_for_download": True,
  "download.directory_upgrade": True,
  "safebrowsing.enabled": True
})

# # 初始化 WebDriver by changed
# service = Service("D:/user/chromedriver_win32/chromedriver.exe")
# driver = webdriver.Chrome(service=service, options=options)
# driver.get("https://codis.cwa.gov.tw/StationData")
# time.sleep(1)
# element = driver.find_element(By.XPATH, '//*[@id="auto_C0"]')
# element.click()

service = Service("D:/user/chromedriver_win32/chromedriver.exe")
driver = webdriver.Chrome(service=service)
driver.get("https://codis.cwa.gov.tw/StationData")
time.sleep(1)
element = driver.find_element(By.XPATH, '//*[@id="auto_C0"]')
element.click()

# 定位下拉選單元素
select_element = driver.find_element(By.XPATH, '//*[@id="station_area"]')
# 創建Select對象
select = Select(select_element)
# 選擇下拉選單的一個選項，這裡以選擇第一個選項為例
select.select_by_index(15)  # 通過索引選擇

# 定位到 <input> 元素
input_element = driver.find_element(By.XPATH, '/html/body/div/main/div/div/div/div/aside/div/div[1]/div/div/section/ul/li[5]/div/div[2]/div/input')
# 向 <input> 元素輸入文本
input_element.send_keys("高雄 (467441)")


# 定位到按鈕元素
button_element = driver.find_element(By.XPATH, '/html/body/div[1]/main/div/div/section[1]/div[1]/div[3]/div[1]/div[1]/div[11]/div/div/div/div[2]')
# 點擊按鈕
button_element.click()
# 定位到按鈕元素
button_element = driver.find_element(By.XPATH, '/html/body/div[1]/main/div/div/section[1]/div[1]/div[3]/div[1]/div[1]/div[6]/div/div[1]/div/button')
# 點擊按鈕
button_element.click()

# 定位到按鈕元素
button_XPATH = "/html/body/div[1]/main/div/div/section[2]/div/div/section/div[5]/div[1]/div[2]/div"
button_element = driver.find_element(By.XPATH, button_XPATH)  # 使用變量而不是字符串

# 點擊按鈕
button_element.click()

button2_XPATH = '/html/body/div[1]/main/div/div/section[2]/div/div/section/div[5]/div[1]/div[1]/label/div/div[1]'
button2_element = driver.find_element(By.XPATH, button2_XPATH)  # 使用變量而不是字符串



import time

# 點擊按鈕
# for i in range(0, 100):
#     print(f"進行中：{i+1}/100")  # 顯示進度
#     button2_element.click()
#     try:
#         button_element = WebDriverWait(driver, 5).until(
#             EC.presence_of_element_located((By.XPATH, button_XPATH))
#         )

#     except:
#         print("找不到指定的元素。")
#     button_element.click()
#     time.sleep(1)

year = 2;
for i in range(0, 365*year):
    print(f"進行中：{i+1}/"+str(365*year))  # 顯示進度
    button2_element.click()
    time.sleep(0.01)
    button_element.click()
    time.sleep(0.01)
