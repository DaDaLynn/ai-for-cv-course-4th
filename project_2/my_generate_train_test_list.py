import cv2

def str2int(val):
    return int(float(val))

def expand_roi(face_loc, expand_ratio, w, h):
    [x1, y1, x2, y2] = face_loc
    w_expand = int(expand_ratio * 0.5 * (x2 - x1))
    h_expand = int(expand_ratio * 0.5 * (y2 - y1))
    x1 = max(0, x1 - w_expand)
    y1 = max(0, y1 - h_expand)
    x2 = min(w - 1, x2 + w_expand)
    y2 = min(h - 1, y2 + h_expand)
    return [x1, y1, x2, y2]

def show_sample(root_path, file_txt):
    file_txt_path = root_path + '\\' + file_txt    
    with open(file_txt_path) as fp:
        lines = fp.readlines()
        cv2.namedWindow("Face")
        for line in lines:
            info_array = line.strip().split()
            img_name = info_array[0]            
            face_loc = info_array[1:5]
            face_loc = [str2int(i) for i in face_loc]
            key_pts = info_array[5:]
            key_pts = [str2int(i) for i in key_pts]
            img_path = root_path + "\\" + img_name
            img = cv2.imread(img_path)
            cv2.rectangle(img, 
                          (face_loc[0], face_loc[1]), 
                          (face_loc[2], face_loc[3]),
                          (0, 255, 0))
            key_pt_nums = len(key_pts) // 2
            for i in range(key_pt_nums):
                cv2.circle(img, 
                           (key_pts[i * 2] + face_loc[0], key_pts[i * 2 + 1] + face_loc[1]),
                            1,
                            (255, 0, 0))
            #cv2.imwrite(r"D:\Lynn\AI-for-CV\Class material\project\projectII_face_keypoints_detection\code\python\my_result.jpg", img)
            cv2.imshow("Face", img)
            cv2.waitKey(1000)
            #break
        cv2.destroyAllWindows()

def expand_data(root_path, in_txt, out_txt):
    file_txt_path = root_path + '\\' + in_txt
    file_out_txt  = root_path + '\\' + out_txt
    with open(file_txt_path, 'r') as fp:
        with open(file_out_txt, 'w') as fp1:
            lines = fp.readlines()
            for line in lines:
                info_array = line.strip().split()
                img_name = info_array[0]
                face_loc = info_array[1:5]
                face_loc = [str2int(i) for i in face_loc]
                key_pts = info_array[5:]
                key_pts = [str2int(i) for i in key_pts]
                img_path = root_path + "\\" + img_name
                img = cv2.imread(img_path)
                size = img.shape
                face_loc = expand_roi(face_loc, 0.15, size[1], size[0])
                fp1.write(img_name)
                for i in face_loc:
                    fp1.write(" " + str(i))
                key_pt_nums = len(key_pts) // 2
                for i in range(key_pt_nums):
                    key_pts[i * 2    ] -= face_loc[0]
                    key_pts[i * 2 + 1] -= face_loc[1]
                    fp1.write(" " + str(key_pts[i * 2]) + " " + str(key_pts[i * 2 + 1]))
                fp1.write("\n")

if __name__ == "__main__":
    root_path = "D:\\Lynn\\AI-for-CV\\Class material\\project\\projectII_face_keypoints_detection\\data\\I\\I"
    in_file_name = "label.txt"
    out_file_name = "expand_label.txt"
    expand_data(root_path, in_file_name, out_file_name)
    #show_sample(root_path, out_file_name)