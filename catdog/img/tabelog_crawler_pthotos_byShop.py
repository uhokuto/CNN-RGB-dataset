# -*- coding:shift-jis -*-
# coding: cp932

from urllib.request import urlopen
from urllib.error import HTTPError
from urllib.error import URLError
import urllib.request
import urllib.parse
import urllib
import lxml.html
import pandas as pd
from pykakasi import kakasi

try:
	#取得した画像を保存するフォルダー名（教師ラベル番号とすると画像判別プログラムと整合性がとれる）
	folder_name='1'

	kakasi = kakasi()
	kakasi.setMode('H', 'a')
	kakasi.setMode('K', 'a')
	kakasi.setMode('J', 'a')

	conv = kakasi.getConverter()
	
	
	
	shop_review_url = 'https://tabelog.com/akita/A0501/A050101/5000203/' + 'dtlphotolst/1/smp2/D-normal/'
	max_photoList_pages=5
	
	for pageNo in range(max_photoList_pages):
		
		print(pageNo+1)
		bodyRes = urllib.request.urlopen('{0}{1}/'.format(shop_review_url,pageNo+2))		
		bodyHtml = bodyRes.read()						
		htmlTreeBody = lxml.html.fromstring(bodyHtml.decode('utf-8'))
		# <div class="thum-photobox__img">
        # <a href="https://tblg.k-img.com/restaurant/images/Rvw/73869/640x640_rect_73869352.jpg" 
		food_image_paths = htmlTreeBody.xpath("//div[@class='thum-photobox__img']/a")
		# <div class="rdheader-rstname">
		# <h2 class="display-name">
		# <a href="https://tabelog.com/tokyo/A1302/A130202/13193074/">稲庭うどんとめし 金子半之助 コレド室町店</a>
		shop_name_path = htmlTreeBody.xpath("//div[@class='rdheader-rstname']/h2/a")
		shop_name=shop_name_path[0].text_content()
		shop_name_kana=conv.do(shop_name)
		for i,food_image_path in enumerate(food_image_paths):
			
			print('counter:',pageNo*40+i)					
			food_photo = urllib.request.urlopen(food_image_path.get('href'))
			food_photo_binary = food_photo.read()
			picfile = open('./{0}/{1}{2}.jpg'.format(folder_name,shop_name_kana,str(pageNo*40+i)), 'wb')
			picfile.write(food_photo_binary)
			food_photo.close()
			picfile.close()
			
			
	

except HTTPError as e:
	print(e)
except URLError as e:
	print("The server could not be found.")

finally:
	
	print("It Worked")
