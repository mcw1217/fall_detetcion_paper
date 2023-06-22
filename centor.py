import cv2

def crop_center(image_path, output_path, crop_width, crop_height):
    # 이미지 로드
    image = cv2.imread(image_path)
    
    # 이미지 크기 얻기
    height, width = image.shape[:2]
    
    # 중앙 좌표 계산
    center_x = width // 2
    center_y = height // 2
    
    # 중앙을 기준으로 잘라내기
    left = center_x - crop_width // 2
    top = center_y - crop_height // 2
    right = left + crop_width
    bottom = top + crop_height
    
    cropped_image = image[top:bottom, left:right]
    height, width = cropped_image.shape[:2]
    image = cv2.resize(cropped_image,(4*width, 4*height), interpolation=cv2.INTER_LINEAR)
    
    
    # 잘라낸 이미지 저장
    cv2.imshow("test",image)
    cv2.imwrite(output_path, image)
    cv2.waitKey(0)

# 이미지 경로와 잘라낼 크기 설정
image_path = "testtest.jpg"
output_path = "cropped_image.jpg"
crop_width = 50
crop_height = 50

# 이미지 중앙을 잘라내기
crop_center(image_path, output_path, crop_width, crop_height)