def medianBlur(img, kernel, padding_way):
#        img & kernel is List of List; padding_way a string
#        Please finish your code under this blank
    H = len(img)
    W = len(img[0])
    n = len(kernel)
    m = len(kernel[0])
    if (H and W) == 0:
        print("img is null")
        return img
    if (n % 2) * (m % 2) == 0:
        print("kernel size should be odd number")
        return img
    pad_W = W + m - 1
    pad_H = H + n - 1
    pad_top = n // 2
    pad_left = m // 2
    img_pad = [[0 for i in range(pad_W)] for j in range(pad_H)]
    for i in range(H):
        for j in range(W):
            img_pad[pad_top + i][pad_left + j] = img[i][j]
    if padding_way == "REPLICA":
        # top left area
        for i in range(pad_top):
            for j in range(pad_left):
                img_pad[i][j] = img[0][0]
            for j in range(pad_left, pad_left + W):
                img_pad[i][j] = img[0][j - pad_left]
            for j in range(pad_left + W, pad_W):
                img_pad[i][j] = img[0][-1]
        for i in range(pad_top, pad_top + H):
            for j in range(pad_left):
                img_pad[i][j] = img[i - pad_top][0]
            for j in range(pad_left + W, pad_W):
                img_pad[i][j] = img[i - pad_top][-1]
        for i in range(pad_top + H, pad_H):
            for j in range(pad_left):
                img_pad[i][j] = img[-1][0]
            for j in range(pad_left, pad_left + W):
                img_pad[i][j] = img[-1][j - pad_left]
            for j in range(pad_left + W, pad_W):
                img_pad[i][j] = img[-1][-1]
    elif padding_way == "ZERO":
        pass
    else:
        print("padding method is not supported")
        return img

    ret = []
    mid_index = m * n // 2
    for i in range(H):
        row = []
        for j in range(W):
            window = []
            for ii in range(n):
                for jj in range(m):
                    window.append(img_pad[i + ii][j + jj])
            window.sort()
            row.append(window[mid_index])
        ret.append(row)
    return ret

import random
if __name__ == "__main__":
    img = [[random.randint(5, 15) for i in range(5)] for j in range(5)]
    kernel = [[1 for i in range(3)] for j in range(3)]
    padding_way = "REPLICA"
    ret = medianBlur(img, kernel, padding_way)
    print(img)
    print(ret)
