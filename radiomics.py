import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import RFE #RFE库
from sklearn.feature_selection import VarianceThreshold
from sklearn.linear_model import LogisticRegression #逻辑回归库
from sklearn.svm import SVC #支持向量机
from sklearn.neighbors import KNeighborsClassifier #K近邻
from sklearn.tree import DecisionTreeClassifier #决策树
from sklearn.ensemble import RandomForestClassifier #随机森林
from sklearn.naive_bayes import GaussianNB #高斯朴素贝叶斯
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, roc_auc_score, confusion_matrix, accuracy_score
import  matplotlib.pyplot as plt
import seaborn as sns
from sklearn.inspection import permutation_importance

filename = 'data_US2.csv'

data = pd.read_csv(filename, encoding='ANSI')
X = data[data.columns[1:]]  # x 是数据矩阵（第 sign 列是分组）
y = data['label']   # y 是分组
X = X.apply(pd.to_numeric, errors='ignore') # 将数据类型转化为数值型
colNames = X.columns #读取特征的名字
X = X.astype(np.float64) #转换 float64 类型，防止报 warning
scaler = MinMaxScaler()
X = scaler.fit_transform(X)
X = pd.DataFrame(X)
X.columns = colNames

vt = VarianceThreshold(threshold=.02) #方差选择法
_ = vt.fit(X)
mask_0 = vt.get_support()
X_0 = X.loc[:, mask_0]

model = LogisticRegression(max_iter=8000)
rfe = RFE(estimator=model, n_features_to_select=30, step=1) #选择30个最佳特征变量，并进行RFE
fit = rfe.fit(X_0, y) #进行RFE递归
mask = fit.get_support()
X_select = X_0.loc[:, mask]

def Confidence_Interval(y_test, y_pred_proba, y_pred):
    n_bootstraps = 1000
    auc_bootstrap = []
    accuracy_bootstrap = []
    specificity_bootstrap = []
    sensitivity_bootstrap = []

    for _ in range(n_bootstraps):
        indices = np.random.choice(len(y_test), len(y_test), replace=True)
        y_test_bootstrap = y_test[indices]
        y_pred_proba_bootstrap = y_pred_proba[indices]
        y_pred_bootstrap = y_pred[indices]

        auc_bootstrap.append(roc_auc_score(y_test_bootstrap, y_pred_proba_bootstrap))
        accuracy_bootstrap.append(accuracy_score(y_test_bootstrap, y_pred_bootstrap))
        tn, fp, fn, tp = confusion_matrix(y_test_bootstrap, y_pred_bootstrap).ravel()
        specificity_bootstrap.append(tn / (tn + fp))
        sensitivity_bootstrap.append(tp / (tp + fn))

    # 计算95%置信区间
    confidence_level = 0.95
    alpha = 1 - confidence_level
    lower_percentile = alpha / 2 * 100
    upper_percentile = (1 - alpha / 2) * 100

    auc_ci = np.percentile(auc_bootstrap, [lower_percentile, upper_percentile])
    accuracy_ci = np.percentile(accuracy_bootstrap, [lower_percentile, upper_percentile])
    specificity_ci = np.percentile(specificity_bootstrap, [lower_percentile, upper_percentile])
    sensitivity_ci = np.percentile(sensitivity_bootstrap, [lower_percentile, upper_percentile])
    
    print(f"        AUC 95% Confidence Interval:{auc_ci}")
    print(f"Sensitivity 95% Confidence Interval:{sensitivity_ci}")
    print(f"Specificity 95% Confidence Interval:{specificity_ci}")
    print(f"   Accuracy 95% Confidence Interval:{accuracy_ci}")

def classification(X_select, algorithm):
    X_train, X_test, y_train, y_test = train_test_split(X_select.values, y.values, test_size=.2, random_state=666) # MR:666   US：666
    if algorithm == 'LR':
        model = LogisticRegression(max_iter=8000, class_weight='balanced') # 创建逻辑回归训练模型
        optimal_threshold = 0.48
    elif algorithm == 'SVM':
        model = SVC(kernel='linear', C=1, probability=True, class_weight='balanced') #创建SVM训练模型
        optimal_threshold = 0.5
    elif algorithm == 'KNN':
        model = KNeighborsClassifier(weights='distance') #创建K近邻训练模型
        optimal_threshold = 0.18
    elif algorithm == 'DT':
        model = DecisionTreeClassifier(splitter='random', max_depth=8, random_state=10, class_weight='balanced') #创建决策树训练模型
        optimal_threshold = 0.5
    elif algorithm == 'RF':
        model = RandomForestClassifier(n_estimators=40, max_depth=6, random_state=6, class_weight='balanced') #创建随机森林训练模型
        optimal_threshold = 0.38
    elif algorithm == 'GNB':
        model = GaussianNB() #创建高斯朴素贝叶斯训练模型
        optimal_threshold = 0.12
    elif algorithm == 'GBDT':
        params = {'n_estimators': 100, 'max_depth': 7, 'min_samples_split': 100,
                   'learning_rate': 0.1, 'random_state': 666}  #MR加临床
        model = GradientBoostingClassifier(**params) #创建梯度提升决策树训练模型
        optimal_threshold = 0.19
    model.fit(X_train, y_train)
    imps = permutation_importance(model, X_test, y_test, random_state=66, n_repeats=30)
    coef = np.array(imps.importances_mean)
    
    y_score_1 = model.predict_proba(X_train)[:,1]  #train score
    fpr_1,tpr_1,thresholds = roc_curve(y_train, y_score_1, pos_label=1, drop_intermediate=False)
    y_score_2 = model.predict_proba(X_test)[:,1]  #test score
    fpr_2,tpr_2,_ = roc_curve(y_test, y_score_2, pos_label=1, drop_intermediate=False) 
    rate = [fpr_1,tpr_1,fpr_2,tpr_2]
    y_pre_1 = np.where(y_score_1 > optimal_threshold, 1, 0)
    auc_1 = roc_auc_score(y_train, y_score_1) #train auc
    y_pre_2 = np.where(y_score_2 > optimal_threshold, 1, 0)
    auc_2 = roc_auc_score(y_test, y_score_2) #test auc
    M_1 = confusion_matrix(y_train, y_pre_1)
    sen_1 = M_1[1,1]/(M_1[1,0]+M_1[1,1])  #train sensitivity
    spe_1 = M_1[0,0]/(M_1[0,0]+M_1[0,1])  #train specificity
    M_2 = confusion_matrix(y_test, y_pre_2)
    sen_2 = M_2[1,1]/(M_2[1,0]+M_2[1,1])  #test sensitivity
    spe_2 = M_2[0,0]/(M_2[0,0]+M_2[0,1])  #test specificity
    acc_1 = model.score(X_train, y_train)  #train acc
    acc_2 = model.score(X_test, y_test)  #test acc
    indicator = np.array([auc_1,sen_1,spe_1,acc_1,auc_2,sen_2,spe_2,acc_2])
    print('\n', algorithm, ':')
    print('        train auc:', indicator[0])
    print('train sensitivity:', indicator[1])
    print('train specificity:', indicator[2])
    print('        train acc:', indicator[3])
    print('         test auc:', indicator[4])
    print(' test sensitivity:', indicator[5])
    print(' test specificity:', indicator[6])
    print('         test acc:', indicator[7])
    coef = coef / abs(coef).max()
    
    Confidence_Interval(y_train, y_score_1, y_pre_1)
    Confidence_Interval(y_test, y_score_2, y_pre_2)
    
    return indicator, rate, coef

def picture(dt, L, dt_auc):
    Color = ['b','g','r','c','m','y','k']
    plt.figure()
    s = ['Train', 'Test']
    for k in [0,1]:
        plt.subplot(1,2,k+1)
        plt.title(s[k])
        for i,name in enumerate(L):
            lab = name +' '*(8-len(name))+'auc='+str('{0:.4f}'.format(dt_auc[i,k]))
            plt.plot(dt[i][2*k],dt[i][2*k+1],color=Color[i],label=lab)
        plt.plot([0, 1], [0, 1], linewidth=2, linestyle='--')
        plt.xlabel('False Positive Rate', fontsize=20)
        plt.ylabel('True Positive Rate', fontsize=20)
        plt.xlim(0.0, 1.0)
        plt.ylim(0.0, 1.05)
        plt.rcParams.update({'font.size': 14})
        plt.legend(loc="lower right")
        plt.grid(True)
    plt.show()

def heat_map(df): # 热力图
    plt.figure()
    sns.set(font="simhei")
    plt.rcParams['axes.unicode_minus']=False
    sns.heatmap(df.corr(), annot=False, vmax=1, square=True, xticklabels=df.columns, yticklabels=df.columns)#绘制new_df的矩阵热力图
    plt.show()#显示图片

def coef_map(coef_list): # 相关性图
    fig, ax = plt.subplots()
    # vmax, vmin = coef_list.max(), coef_list.min()
    sns.heatmap(pd.DataFrame(np.round(coef_list,2)), annot=False, vmax=1, vmin=-1, xticklabels=list(X_select.columns), yticklabels=L, square=True, cmap="YlGnBu")
    plt.show()#显示图片

print('选择的特征：', X_select.columns)
L = ['LR', 'SVM', 'KNN', 'DT', 'RF', 'GNB', 'GBDT']
# LR:逻辑回归；SVM：支持向量机；KNN：K近邻；DT：决策树；
# RF:随机森林；GNB：高斯朴素贝叶斯；GBDT：梯度提升决策树

indicator_dict = []
rate_list = []
coef_list = []
for algorithm in L:
    indicator, rate, coef = classification(X_select, algorithm)
    indicator_dict.append(indicator)
    rate_list.append(rate)
    coef_list.append(coef)
df_1 = pd.DataFrame(np.array(indicator_dict), index=L)
picture(rate_list, L, np.array(df_1.iloc[:,[0,4]]))
coef_list = np.array(coef_list)

heat_map(X_select)

coef_map(coef_list)