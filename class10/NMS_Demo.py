def iou(rect0, rect1):
    [Ax1,Ay1,Ax2,Ay2] = rect0
    [Bx1,By1,Bx2,By2] = rect1
    area_A = (Ax2 - Ax1) * (Ay2 - Ay1)
    area_B = (Bx2 - Bx1) * (By2 - By1)
    inter_area = (min(Bx2, Ax2) - max(Bx1, Ax1)) * (min(By2, Ay2) - max(By1, Ay1))
    return inter_area / (area_A + area_B - inter_area)
    

def NMS(reg_pro_lists, thre):
    num_reg = len(reg_pro_lists)
    if num_reg == 0:
        return None
    
    D = []
    while len(reg_pro_lists):
        score = [item[4] for item in reg_pro_lists]
        m = score.index(max(score))
        M = reg_pro_lists[m]
        D.extend(M)
        reg_pro_lists = list(set(reg_pro_lists) - set(M))
        for item in reg_pro_lists:
            if iou(M, item) >= thre:
                reg_pro_lists = list(set(reg_pro_lists) - set(item))
    return D
