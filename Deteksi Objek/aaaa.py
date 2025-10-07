import cv2

# Inisialisasi objek SIFT
sift = cv2.SIFT_create()

def sift_feature_matching(img1, img2):
    # Temukan titik kunci dan deskriptor untuk kedua gambar
    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)

    # Gambar titik kunci pada kedua gambar
    img1_with_keypoints = cv2.drawKeypoints(img1, kp1, None)
    img2_with_keypoints = cv2.drawKeypoints(img2, kp2, None)

    scale_percent = 75  
    width = int(img1_with_keypoints.shape[1] * scale_percent / 100)
    height = int(img2_with_keypoints.shape[0] * scale_percent / 100)
    dim = (width, height)
    img1_result_resized = cv2.resize(img1_with_keypoints, dim, interpolation=cv2.INTER_AREA)
    img2_result_resized = cv2.resize(img2_with_keypoints, dim, interpolation=cv2.INTER_AREA)

    # Tampilkan gambar dengan titik kunci
    cv2.imshow('Keypoints 1', img1_result_resized)
    cv2.imshow('Keypoints 2', img2_result_resized)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # Buat objek matcher BFMatcher
    bf = cv2.BFMatcher()

    # Lakukan pencocokan fitur
    matches = bf.knnMatch(des1, des2, k=2)

    # Simpan semua pencocokan yang baik (persentase jarak terbaik / terdekat < 0,75)
    good_matches = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good_matches.append(m)

    return good_matches

def image_classifier(train_img, query_img):
    # Lakukan deteksi fitur dan pencocokan SIFT
    matches = sift_feature_matching(train_img, query_img)

    # Hitung jumlah pencocokan yang baik
    num_good_matches = len(matches)

    # Tampilkan hasil pencocokan di gambar query
    img_result = cv2.drawMatches(train_img, sift.detect(train_img), query_img, sift.detect(query_img), matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    
    # Resize gambar hasil
    scale_percent = 75  
    width = int(img_result.shape[1] * scale_percent / 100)
    height = int(img_result.shape[0] * scale_percent / 100)
    dim = (width, height)
    img_result_resized = cv2.resize(img_result, dim, interpolation=cv2.INTER_AREA)

    # Tambahkan teks klasifikasi
    if num_good_matches > 10:  # Ubah ambang batas sesuai kebutuhan Anda
        result_text = "Match"
    else:
        result_text = "No match"
    cv2.putText(img_result_resized, result_text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Tampilkan hasil gambar
    cv2.imshow("Result", img_result_resized)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Load gambar train dan query
query_img = cv2.imread('image/buku.jpg', cv2.IMREAD_GRAYSCALE)
train_img = cv2.imread('train/semua.jpg', cv2.IMREAD_GRAYSCALE)

# Panggil fungsi klasifikasi gambar
image_classifier(train_img, query_img)
