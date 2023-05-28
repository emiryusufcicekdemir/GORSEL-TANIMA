import cv2
import numpy as np

# Görsel linklerini bir liste olarak tanımla
image_links = ["https://drive.google.com/drive/folders/1WXM3kbKfKt-k8L87to47WQamYn6oeIym?usp=share_link"]

# OpenCV ve NumPy kütüphanelerini içe aktar
import cv2
import numpy as np

# Bir döngü ile listenizdeki her görseli oku ve gri tonlamaya dönüştür
images = [cv2.cvtColor(cv2.imread(link), cv2.COLOR_BGR2GRAY) for link in image_links]

# Bir döngü ile listenizdeki her görseli 20x20 piksel parçalara böl ve bu parçaları bir sözlükte tut
patches = {}
patch_size = (20, 20)
for link, image in zip(image_links, images):
    h, w = image.shape[:2]
    patches[link] = []
    for y in range(0, h, patch_size[1]):
        for x in range(0, w, patch_size[0]):
            patch = image[y:y+patch_size[1], x:x+patch_size[0]]
            patches[link].append(patch)

# Bir döngü ile her görsel için diğer görsellerle olan benzerliklerini hesapla
# Benzerlik ölçütü olarak cv2.TM_CCOEFF_NORMED yöntemini kullan
# Benzerlik skorlarını ve eşleşen parçaların koordinatlarını bir liste içinde tut
similarities = []
for i in range(len(image_links)):
    for j in range(i+1, len(image_links)):
        link1 = image_links[i]
        link2 = image_links[j]
        patches1 = patches[link1]
        patches2 = patches[link2]
        for k in range(len(patches1)):
            for l in range(len(patches2)):
                patch1 = patches1[k]
                patch2 = patches2[l]
                result = cv2.matchTemplate(patch1, patch2, cv2.TM_CCOEFF_NORMED)
                score = result[0][0]
                similarities.append((score, link1, link2, k, l))

# Benzerlik skorlarına göre listeni büyükten küçüğe sırala ve en yüksek skora sahip olan eşleşmeyi yazdır
similarities.sort(reverse=True)
best_match = similarities[0]
print(f"En çok benzeyen parçalar {best_match[1]} görselinin {best_match[3]}. parçası ile {best_match[2]} görselinin {best_match[4]}. parçasıdır.")
print(f"Benzerlik skoru: {best_match[0]}")
