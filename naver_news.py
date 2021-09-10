import re
import random
import requests
from bs4 import BeautifulSoup
from datetime import datetime, timedelta

def get_soup_obj(url):
    headers = {'User-Agent' : 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/88.0.4324.150 Safari/537.36'}
    res = requests.get(url, headers = headers)
    soup = BeautifulSoup(res.text,'html.parser')
    
    return soup

def get_naver_news():
    startdate=datetime.today()

    while True:
        date=startdate-timedelta(random.randint(1,14))
        date = date.strftime('%Y%m%d')

        pages = [num for num in range(1,5)]
        page = random.choice(pages)

        sec_url = "https://news.naver.com/main/list.naver?mode=LPOD&mid=sec&oid=023&date={}&page={}".format(date,page)
        soup = get_soup_obj(sec_url)
        
        
        lis = soup.find('ul', class_='type06').find_all("li", limit=9)
        li = random.choice(lis)
        news_url = li.a.attrs.get('href')
        
        soup = get_soup_obj(news_url)
        try:
            text = soup.find('div', id='articleBodyContents').get_text()
            text = text.replace('// flash 오류를 우회하기 위한 함수 추가','')
            text = text.replace('function _flash_removeCallback() {}','')
            text = text.strip()
            """
            try:
                for match in re.finditer(r'\(\S*=연합뉴스\)',text):
                    pass
                start = match.start()
            except:
                start = 0
            """
            #if len(text[start::]) > 500 and len(text[start::]) < 5000:
            if len(text) > 500 and len(text) < 2000:
                break
        except:
            pass
    
    return text

if __name__ == "__main__":
	text=get_naver_news()
	print(text)