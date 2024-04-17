from PIL import Image, ImageDraw
import numpy as np
import random
from shapely.geometry import Polygon, Point

# 지도 이미지 경로 설정
map_image_path = "CNNData/Map.jpg"  # 지도 이미지 파일 경로 설정
pin_image_path = "CNNData/Pin.jpg"  # 핀 이미지 파일 경로 설정

# 지도 이미지와 핀 이미지 불러오기
map_image = Image.open(map_image_path)
pin_image = Image.open(pin_image_path)

# 핀 이미지 크기 설정 (원하는 크기로 조정 가능)
pin_width = 15
pin_height = 15
pin_image_resized = pin_image.resize((pin_width, pin_height))

polygon_points = [(100, 100), (300, 200), (400, 400), (200, 300)]  # 임의의 폴리곤 좌표 설정
polygon = Polygon(polygon_points)

min_x, min_y, max_x, max_y = polygon.bounds

draw = ImageDraw.Draw(map_image)

while True:
    # 경계 상자 내에 랜덤한 점 생성
    x = random.uniform(min_x, max_x)
    y = random.uniform(min_y, max_y)
    point = Point(x, y)
    
    # 생성된 점이 폴리곤 내부에 있는지 확인
    if polygon.contains(point):
        # 핀 이미지를 해당 위치에 그림
        map_with_pin = map_image.copy()
        map_with_pin.paste(pin_image_resized, (int(x - pin_width / 2), int(y - pin_height / 2)), pin_image_resized)
        
        # 결과 이미지 저장
        map_with_pin.save("map_with_pin_in_polygon.jpg")
        break
# # 핀을 올릴 랜덤한 위치 설정 (지도 이미지 내)
# map_width, map_height = map_image.size
# x = random.randint(0, map_width - pin_width)
# y = random.randint(0, map_height - pin_height)

# # 핀을 지도 이미지에 합성
# map_with_pin = map_image.copy()
# map_with_pin.paste(pin_image_resized, (x, y), pin_image_resized)

# # 결과 이미지 저장
# map_with_pin.save("map_with_pin.jpg")

# 결과 이미지 표시 (옵션)
map_with_pin.show()