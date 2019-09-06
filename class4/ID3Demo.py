import pandas as pd
import numpy as np

def calc_entropy(yes_prob):
    if yes_prob == 0 or yes_prob == 1:
        return 0
    else:
        no_prob = 1 - yes_prob
        return - (yes_prob * np.log2(yes_prob) + no_prob * np.log2(no_prob))

def calc_feat_entropy(Data, feature = "", attr = ""):
    if feature == "":
        yes_count = len(Data[Data["Welcome"] == "Y"])
        nSum = len(Data)
        yes_prob  = yes_count / nSum
        return calc_entropy(yes_prob),1
    else:
        App_Great = Data[Data[feature] == attr]
        yes_count = len(App_Great[App_Great["Welcome"] == "Y"])
        nSum = len(App_Great)
        if nSum == 0:
            return 0, 0
        else:
             yes_prob  = yes_count / nSum
             return calc_entropy(yes_prob),nSum / len(Data)
'''
def myID3_demo(Data, level = 0, num = 0):
    node_entropy,1 = calc_feat_entropy(Data)
    if node_entropy == 1:
        print("node_{}_{}".format(level, num))
    else:
'''
        


if  __name__ == "__main__":
    Data_raw = [['Good', 'Low', 'Older', 'Steady', 'N'],
            ['Good', 'Low', 'Older', 'Unstable', 'N'],
            ['Great', 'Low', 'Older', 'Steady', 'Y'],
            ['Ah', 'Good', 'Older', 'Steady', 'Y'],
            ['Ah', 'Great', 'Younger', 'Steady', 'Y'],
            ['Ah', 'Great', 'Younger', 'Unstable', 'N'],
            ['Great', 'Great', 'Younger', 'Unstable', 'Y'],
            ['Good', 'Good', 'Older', 'Steady', 'N'],
            ['Good', 'Great', 'Younger', 'Steady', 'Y'],
            ['Ah', 'Good', 'Younger', 'Steady', 'Y'],
            ['Good', 'Good', 'Younger', 'Unstable', 'Y'],
            ['Great', 'Good', 'Older', 'Steady', 'Y'],
            ['Great', 'Low', 'Younger', 'Steady', 'Y'],
            ['Ah', 'Good', 'Older', 'Unstable', 'N']]
    columns = ['Appearance', 'Income', 'Age', 'Profession', 'Welcome']
    Data = pd.DataFrame(Data_raw, columns = columns)
    
    # a. Calculate the entropy of the system
    entropy,p  = calc_feat_entropy(Data)
    print("entropy = {}".format(entropy))

    # b. Calculate entropies for each feature
    # Appearance.Great
    entropy_app_great,p_great   = calc_feat_entropy(Data, 'Appearance', 'Great')
    print("entropy_app_great = {} {}".format(entropy_app_great, p_great))
    # Appearance.Good
    entropy_app_good,p_good   = calc_feat_entropy(Data, 'Appearance', 'Good')
    print("entropy_app_good = {} {}".format(entropy_app_good, p_good))
    # Appearance.Ah
    entropy_app_ah,p_ah   = calc_feat_entropy(Data, 'Appearance', 'Ah')
    print("entropy_app_ah = {} {}".format(entropy_app_ah, p_ah))
    H_Fapp = p_great * entropy_app_great + p_good * entropy_app_good + p_ah * entropy_app_ah
    print("H_Fapp = {}".format(H_Fapp))

    # Income.Great
    entropy_Inc_great,p_great   = calc_feat_entropy(Data, 'Income', 'Great')
    #print("entropy_Inc_great = {} {}".format(entropy_Inc_great, p_great))
    # Income.Good
    entropy_Inc_good,p_good   = calc_feat_entropy(Data, 'Income', 'Good')
    #print("entropy_Inc_good = {} {}".format(entropy_Inc_good, p_good))
    # Income.Low
    entropy_Inc_low,p_low   = calc_feat_entropy(Data, 'Income', 'Low')
    #print("entropy_Inc_low = {} {}".format(entropy_Inc_low, p_low))
    H_Finc = p_great * entropy_Inc_great + p_good * entropy_Inc_good + p_low * entropy_Inc_low
    print("H_Finc = {}".format(H_Finc))

    # Age.Older
    entropy_age_older,p_older   = calc_feat_entropy(Data, 'Age', 'Older')
    #print("entropy_age_older = {} {}".format(entropy_age_older, p_older))
    # Age.younger
    entropy_age_younger,p_younger   = calc_feat_entropy(Data, 'Age', 'Younger')
    #print("entropy_age_younger = {} {}".format(entropy_age_younger, p_younger))
    H_Fage = p_older * entropy_age_older + p_younger * entropy_age_younger
    print("H_Fage = {}".format(H_Fage))

    # Profession.Steady
    entropy_pro_steady,p_steady   = calc_feat_entropy(Data, 'Profession', 'Steady')
    #print("entropy_pro_steady = {} {}".format(entropy_pro_steady, p_steady))
    # Profession.Unstable
    entropy_pro_unstable,p_unstable   = calc_feat_entropy(Data, 'Profession', 'Unstable')
    #print("entropy_pro_unstable = {} {}".format(entropy_pro_unstable, p_unstable))
    H_Fpro = p_steady * entropy_pro_steady + p_unstable * entropy_pro_unstable
    print("H_Fpro = {}".format(H_Fpro))

    H_feat = [H_Fapp, H_Finc, H_Fage, H_Fpro]
    node0 = columns[H_feat.index(min(H_feat))]
    print("node0 = {}".format(node0))
    
    # 由于appearance.great的熵已为零，所以此分支已结束
    # 对于appearance.good和appearance.ah分支，按照上述步骤继续split
    # appearance.good
    app_good_part  = Data[Data["Appearance"] == "Good"]

    # Income.Great
    entropy_Inc_great,p_great   = calc_feat_entropy(app_good_part, 'Income', 'Great')
    # Income.Good
    entropy_Inc_good,p_good   = calc_feat_entropy(app_good_part, 'Income', 'Good')
    # Income.Low
    entropy_Inc_low,p_low   = calc_feat_entropy(app_good_part, 'Income', 'Low')
    H_Finc = p_great * entropy_Inc_great + p_good * entropy_Inc_good + p_low * entropy_Inc_low
    print("H_Finc = {}".format(H_Finc))

    # Age.Older
    entropy_age_older,p_older   = calc_feat_entropy(app_good_part, 'Age', 'Older')
    # Age.younger
    entropy_age_younger,p_younger   = calc_feat_entropy(app_good_part, 'Age', 'Younger')
    H_Fage = p_older * entropy_age_older + p_younger * entropy_age_younger
    print("H_Fage = {}".format(H_Fage))

    # Profession.Steady
    entropy_pro_steady,p_steady   = calc_feat_entropy(app_good_part, 'Profession', 'Steady')
    # Profession.Unstable
    entropy_pro_unstable,p_unstable   = calc_feat_entropy(app_good_part, 'Profession', 'Unstable')
    H_Fpro = p_steady * entropy_pro_steady + p_unstable * entropy_pro_unstable
    print("H_Fpro = {}".format(H_Fpro))

    H_feat = [H_Fapp, H_Finc, H_Fage, H_Fpro]
    node1_0 = columns[H_feat.index(min(H_feat))]
    print("node1_0 = {}".format(node1_0))

    # 此时Age作为节点，信息熵已为零，不需要再分

    # appearance.ah
    app_ah_part = Data[Data["Appearance"] == "Ah"]
    # Income.Great
    entropy_Inc_great,p_great   = calc_feat_entropy(app_ah_part, 'Income', 'Great')
    # Income.Good
    entropy_Inc_good,p_good   = calc_feat_entropy(app_ah_part, 'Income', 'Good')
    # Income.Low
    entropy_Inc_low,p_low   = calc_feat_entropy(app_ah_part, 'Income', 'Low')
    H_Finc = p_great * entropy_Inc_great + p_good * entropy_Inc_good + p_low * entropy_Inc_low
    print("H_Finc = {}".format(H_Finc))

    # Age.Older
    entropy_age_older,p_older   = calc_feat_entropy(app_ah_part, 'Age', 'Older')
    # Age.younger
    entropy_age_younger,p_younger   = calc_feat_entropy(app_ah_part, 'Age', 'Younger')
    H_Fage = p_older * entropy_age_older + p_younger * entropy_age_younger
    print("H_Fage = {}".format(H_Fage))

    # Profession.Steady
    entropy_pro_steady,p_steady   = calc_feat_entropy(app_ah_part, 'Profession', 'Steady')
    # Profession.Unstable
    entropy_pro_unstable,p_unstable   = calc_feat_entropy(app_ah_part, 'Profession', 'Unstable')
    H_Fpro = p_steady * entropy_pro_steady + p_unstable * entropy_pro_unstable
    print("H_Fpro = {}".format(H_Fpro))

    H_feat = [H_Fapp, H_Finc, H_Fage, H_Fpro]
    node1_1 = columns[H_feat.index(min(H_feat))]
    print("node1_1 = {}".format(node1_1))
    
    # 此时Profession作为节点，信息熵已为零，不需要再分

    ## 分支节点的划分都停止，则决策树划分结束

