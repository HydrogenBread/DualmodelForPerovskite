import math
from sklearn.metrics import mean_squared_error
import numpy as np
import pandas as pd
import matplotlib
# matplotlib.use('Agg') # 使用 Agg 渲染器
from matplotlib import pyplot as plt
from sklearn import metrics
from sklearn.metrics import r2_score
import seaborn as sns
import re

#相关系数计算函数
def calc_corr(a,b):
    a_avg = sum(a)/len(a)
    b_avg = sum(b)/len(b)
    cov_ab = sum([(x - a_avg)*(y - b_avg) for x,y in zip(a, b)])
    sq = math.sqrt(sum([(x - a_avg)**2 for x in a])*sum([(x - b_avg)**2 for x in b]))
    corr_factor = cov_ab/sq
    return corr_factor

#回归作图
def regplot(train,trainpre,test,testpre,modelname = 'ML',target = 'PCE'):
    fontsize = 12
    plt.figure(figsize=(3,3))
    plt.style.use('default')
    plt.rc('xtick', labelsize=fontsize)
    plt.rc('ytick', labelsize=fontsize)
    plt.rcParams['font.family']="Arial"

    a = plt.scatter(train, trainpre, s=25,c='#5861AC')
    plt.plot([train.min(), train.max()], [train.min(),train.max()], 'k:', lw=1.5)
    plt.xlabel('PCE Observation', fontsize=fontsize)
    plt.ylabel('PCE Prediction', fontsize=fontsize)
    plt.tick_params(direction='in')
    plt.title('{} model for {} prediction'.format(modelname,target),fontsize=fontsize)

    b = plt.scatter(test, testpre, s=25,c='#FF7F00')
    plt.legend((a,b),('Train','Test'),fontsize=fontsize,handletextpad=0.1,borderpad=0.1)
    plt.rcParams['font.family']="Arial"
    plt.tight_layout()
    plt.show()

    print ('Train r:',calc_corr(train, trainpre))
    print ('Train R2:',r2_score(train, trainpre))
    print ('Train RMSE:', np.sqrt(metrics.mean_squared_error(train, trainpre)))
    print('--------------------------------------')
    print ('Test r:', calc_corr(test, testpre))
    print ('Test R2:',r2_score(test, testpre))
    print ('Test RMSE:', np.sqrt(metrics.mean_squared_error(test, testpre)))

# 基于模型预测的函数来发现异常值
def find_outliers(model, X, y, sigma=2):
    # 使用模型预测 y 值
    try:
        y_pred = pd.Series(model.predict(X), index=y.index)
    # 如果无法预测，则先使模型拟合训练集
    except:
        model.fit(X, y)
        y_pred = pd.Series(model.predict(X), index=y.index)

    # 计算模型预测 y 值与真实 y 值之间的残差
    resid = y - y_pred
    mean_resid = resid.mean()
    std_resid = resid.std()

    # 计算异常值定义的参数 z 参数,数据的|z|大于σ将会被视为异常
    z = (resid - mean_resid) / std_resid
    outliers = z[abs(z) > sigma].index

    # 打印结果并绘制图像
    print('R2 = ', model.score(X, y))
    print('MSE = ', mean_squared_error(y, y_pred))
    print('------------------------------------------')
    print('mean of residuals', mean_resid)
    print('std of residuals', std_resid)
    print('------------------------------------------')
    print(f'find {len(outliers)}', 'outliers： ')
    print(outliers.tolist())

    plt.figure(figsize=(15, 5))

    ax_131 = plt.subplot(1, 3, 1)
    plt.plot(y, y_pred, '.')
    plt.plot(y.loc[outliers], y_pred.loc[outliers], 'ro')
    plt.legend(['Accepted', 'Outliers'])
    plt.xlabel('y')
    plt.ylabel('y_pred');

    ax_132 = plt.subplot(1, 3, 2)
    plt.plot(y, y - y_pred, '.')
    plt.plot(y.loc[outliers], y.loc[outliers] - y_pred.loc[outliers], 'ro')
    plt.legend(['Accepted', 'Outliers'])
    plt.xlabel('y')
    plt.ylabel('y - y_pred');

    ax_133 = plt.subplot(1, 3, 3)
    z.plot.hist(bins=50, ax=ax_133)
    z.loc[outliers].plot.hist(color='r', bins=50, ax=ax_133)
    plt.legend(['Accepted', 'Outliers'])
    plt.xlabel('z')

    return outliers

#将训练集和测试集的实际值与预测值合并成一个便于分析的 DataFrame
def getdata(y_train, y_train_hat,y_test, y_test_hat):
    data_train = np.column_stack((y_train, y_train_hat))
    data_test = np.column_stack((y_test, y_test_hat))
    data = pd.DataFrame(np.append(data_train, data_test, axis=0), columns=["Observation", "Prediction"])
    data['type'] = ['Train' if i < len(data_train) else 'Test' for i in range(len(data))] # 添加类型列
    return data

#使用默认字体绘回归点图
def myscatterplotnotime(y_train, y_train_hat,y_test, y_test_hat,modelname="ML", target="PCE",plot_height = 6,savepic = False,picname = 'picname'):

    data = getdata(y_train, y_train_hat,y_test, y_test_hat)
    # 定义默认参数
    # plot_height = 6
    dpi = 300
    plot_aspect = 1.2
    plot_palette = ["#5861AC", "#FF7F00"]
    plot_scatter_kw = {"edgecolor": "black"}
    face_color = "white"
    spine_color = "white"
    label_size = 15
    direction = 'in'
    grid_which = 'major'
    grid_ls = '--'
    grid_c = 'k'
    grid_alpha = .4
    xlim_left = -0.1
    xlim_right = 28
    ylim_bottom = -0.1
    ylim_top = 28

    x_value="Observation"
    y_value="Prediction"
    hue_value="type"
    hue_order_values=['Train', 'Test']

    title_fontdict = {"size": 23, "color": "k"}
    text_fontdict = {'size': '22', 'weight': 'bold', 'color': 'black'}

    # 创建一个正方形画布
    fig, ax = plt.subplots(figsize=(plot_height, plot_height),dpi=dpi)

    # 调用scatterplot函数在空坐标系上绘图
    sns.scatterplot(x=x_value, y=y_value, hue=hue_value, hue_order=hue_order_values, data=data, s=90, alpha=.7,
                    edgecolor=plot_scatter_kw.get('edgecolor', 'none'), palette=plot_palette, ax=ax)

    ax.set_facecolor(face_color)
    for spine in ['top', 'bottom', 'left', 'right']:
        ax.spines[spine].set_color(spine_color)
    ax.tick_params(labelsize=label_size, direction=direction)  # 修改坐标轴数字大小

    ax.grid(which=grid_which, ls=grid_ls, c=grid_c, alpha=grid_alpha)

    ax.set_xlim(left=xlim_left, right=xlim_right)
    ax.set_ylim(bottom=ylim_bottom, top=ylim_top)

    ax.set_title(f"{modelname} for {target} prediction", fontdict=title_fontdict)

    # 设置坐标轴标签字体大小
    ax.set_xlabel(x_value.capitalize(), fontdict={'fontsize': 25})
    ax.set_ylabel(y_value.capitalize(), fontdict={'fontsize': 25})
    ax.plot([-0.5, 25.5], [-0.5, 25.5], linestyle='--', color='gray', linewidth=2)

    # 调整图例放置位置并调整字体大小
    sns.set(font_scale=1.5)  # 设置图例字体大小
    # 调整图例位置和大小
    plt.legend(loc='upper left', fontsize=16)  # 将图例放到左上方并将其字体大小设置为15

    train_text1 = 'Train R: {:.4f}'.format(calc_corr(y_train, y_train_hat))
    test_text1 = 'Test R: {:.4f}'.format(calc_corr(y_test, y_test_hat))
    test_text2 = 'Test R2: {:.4f}'.format(r2_score(y_test, y_test_hat))

    ax.text(0.67, 0.25, train_text1, transform=ax.transAxes, fontsize=15, va='top', ha='left')
    ax.text(0.67, 0.19, test_text1, transform=ax.transAxes, fontsize=15, va='top', ha='left')
    ax.text(0.67, 0.13, test_text2, transform=ax.transAxes, fontsize=15, va='top', ha='left')

    # Test RMSE
    test_rmse = np.sqrt(mean_squared_error(y_test, y_test_hat))
    rmse_text = 'Test RMSE: {:.3f}'.format(test_rmse)
    ax.text(0.67, 0.07, rmse_text, transform=ax.transAxes, fontsize=15, va='top', ha='left')

    if savepic is True:
        plt.savefig('./img/{}.png'.format(picname),transparent = True)
    else:
        pass

    # 显示图像
    plt.show()

#使用默认字体绘回归点图
def myscatterplot(y_train, y_train_hat,y_test, y_test_hat,modelname="ML", target="PCE",plot_height = 6,savepic = False,picname = 'picname'):

    data = getdata(y_train, y_train_hat,y_test, y_test_hat)
    # 定义默认参数
    # plot_height = 6
    dpi = 300
    plot_aspect = 1.2
    plot_palette = ["#5861AC", "#FF7F00"]
    plot_scatter_kw = {"edgecolor": "black"}
    face_color = "white"
    spine_color = "white"
    label_size = 15
    direction = 'in'
    grid_which = 'major'
    grid_ls = '--'
    grid_c = 'k'
    grid_alpha = .4
    xlim_left = -0.1
    xlim_right = 28
    ylim_bottom = -0.1
    ylim_top = 28

    x_value="Observation"
    y_value="Prediction"
    hue_value="type"
    hue_order_values=['Train', 'Test']

    title_fontdict = {"size": 23, "color": "k", 'family': 'Times New Roman'}
    text_fontdict = {'family': 'Times New Roman', 'size': '22', 'weight': 'bold', 'color': 'black'}

    # 创建一个正方形画布
    fig, ax = plt.subplots(figsize=(plot_height, plot_height),dpi=dpi)

    # 调用scatterplot函数在空坐标系上绘图
    sns.scatterplot(x=x_value, y=y_value, hue=hue_value, hue_order=hue_order_values, data=data, s=90, alpha=.7,
                    edgecolor=plot_scatter_kw.get('edgecolor', 'none'), palette=plot_palette, ax=ax)

    ax.set_facecolor(face_color)
    for spine in ['top', 'bottom', 'left', 'right']:
        ax.spines[spine].set_color(spine_color)
    ax.tick_params(labelsize=label_size, direction=direction)  # 修改坐标轴数字大小

    ax.grid(which=grid_which, ls=grid_ls, c=grid_c, alpha=grid_alpha)

    ax.set_xlim(left=xlim_left, right=xlim_right)
    ax.set_ylim(bottom=ylim_bottom, top=ylim_top)

    ax.set_title(f"{modelname} for {target} prediction", fontdict=title_fontdict)

    # 设置坐标轴标签字体大小
    ax.set_xlabel(x_value.capitalize(), fontdict={'fontsize': 25, 'family': 'Times New Roman'})
    ax.set_ylabel(y_value.capitalize(), fontdict={'fontsize': 25, 'family': 'Times New Roman'})
    ax.plot([-0.5, 25.5], [-0.5, 25.5], linestyle='--', color='gray', linewidth=2)

    # 调整图例放置位置并调整字体大小
    sns.set(font_scale=1.5)  # 设置图例字体大小
    # 调整图例位置和大小
    plt.legend(loc='upper left', fontsize=16)  # 将图例放到左上方并将其字体大小设置为15

    train_text1 = 'Train R: {:.4f}'.format(calc_corr(y_train, y_train_hat))
    test_text1 = 'Test R: {:.4f}'.format(calc_corr(y_test, y_test_hat))
    test_text2 = 'Test R2: {:.4f}'.format(r2_score(y_test, y_test_hat))

    ax.text(0.67, 0.25, train_text1, transform=ax.transAxes, fontsize=15, va='top', ha='left')
    ax.text(0.67, 0.19, test_text1, transform=ax.transAxes, fontsize=15, va='top', ha='left')
    ax.text(0.67, 0.13, test_text2, transform=ax.transAxes, fontsize=15, va='top', ha='left')

    # Test RMSE
    test_rmse = np.sqrt(mean_squared_error(y_test, y_test_hat))
    rmse_text = 'Test RMSE: {:.3f}'.format(test_rmse)
    ax.text(0.67, 0.07, rmse_text, transform=ax.transAxes, fontsize=15, va='top', ha='left')

    if savepic is True:
        plt.savefig('./img/{}.png'.format(picname),transparent = True)
    else:
        pass

    # 显示图像
    plt.show()

from sklearn.model_selection import cross_val_score, KFold
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt

#分类交叉验证可视化
def classCVS(regressor,X ,Y,rs = 0):
    regressor = RandomForestClassifier()
    stable_clf = regressor

    kf = KFold(n_splits=10, shuffle=True, random_state=rs)
    scores_accuracy = cross_val_score(stable_clf, X, Y, cv=kf, scoring='accuracy')
    scores_f1 = cross_val_score(stable_clf, X, Y, cv=kf, scoring='f1')

    average_score_accuracy = np.mean(scores_accuracy)
    average_score_f1 = np.mean(scores_f1)

    print("Accuracy Scores:", scores_accuracy)
    print("*************************************")
    print("Average Accuracy Score:", average_score_accuracy)
    print("Average F1 Score:", average_score_f1)

    plt.plot(range(1, 11), scores_accuracy, marker='o', label='Accuracy Score')
    # plt.plot(range(1, 11), np.abs(scores_mae), marker='o', label='MAE')
    plt.xlabel('Fold')
    plt.ylabel('Score')
    plt.title('Cross Validation Scores')
    plt.legend()
    plt.show()

    return scores_accuracy,scores_f1

#回归交叉验证可视化
def regCVS(regressor,X ,Y,rs = 0):
    regressor = RandomForestRegressor()
    PCE_clf = regressor

    kf = KFold(n_splits=10, shuffle=True, random_state=rs)
    scores_r2 = cross_val_score(PCE_clf, X, Y, cv=kf, scoring='r2')
    scores_mae = -cross_val_score(PCE_clf, X, Y, cv=kf, scoring='neg_mean_absolute_error')
    scores_mse = -cross_val_score(PCE_clf, X, Y, cv=kf, scoring='neg_mean_squared_error')
    
    average_score_r2 = np.mean(scores_r2)
    average_score_mae = np.mean(scores_mae)
    average_scores_rmse = np.mean(scores_mse)

    print("r2 Scores:", scores_r2)
    print("*************************************")
    print("Average r2 Scores:", average_score_r2)
    print("Average MAE Scores:", average_score_mae)
    print("Average MSE Scores:", average_scores_rmse)

    plt.plot(range(1, 11), np.abs(scores_r2), marker='o', label='r2 Score')
    # plt.plot(range(1, 11), np.abs(scores_mae), marker='o', label='MAE')
    plt.xlabel('Fold')
    plt.ylabel('Score')
    plt.title('Cross Validation Scores')
    plt.legend()
    plt.show()

    return scores_r2,scores_mae,scores_mse

#保存
def save_plot_data(y_train, y_train_hat, y_test, y_test_hat, savename):
    # 创建一个字典，将数组转换为列
    data = {'y_train': y_train,
            'y_train_predict': y_train_hat,
            'y_test': y_test,
            'y_test_predict': y_test_hat}
    
    # 创建DataFrame
    df = pd.DataFrame(data)
    
    # 保存DataFrame为CSV文件
    df.to_csv('./img/{}.csv'.format(savename), index=False)

def save_arrays_with_nan(y_train, y_train_hat, y_test, y_test_hat, savename):
    # 找到最长的数组长度
    max_length = max(len(y_train), len(y_train_hat), len(y_test), len(y_test_hat))
    
    # 将数组填充到相同长度，并使用NaN填充缺失值
    filled_y_train = np.pad(y_train, (0, max_length - len(y_train)), mode='constant', constant_values=np.nan)
    filled_y_train_hat = np.pad(y_train_hat, (0, max_length - len(y_train_hat)), mode='constant', constant_values=np.nan)
    filled_y_test = np.pad(y_test, (0, max_length - len(y_test)), mode='constant', constant_values=np.nan)
    filled_y_test_hat = np.pad(y_test_hat, (0, max_length - len(y_test_hat)), mode='constant', constant_values=np.nan)
    
    # 创建DataFrame并保存为CSV文件
    df = pd.DataFrame({'y_train': filled_y_train,
                       'y_train_predict': filled_y_train_hat,
                       'y_test': filled_y_test,
                       'y_test_predict': filled_y_test_hat})
    df.to_csv('./img/{}.csv'.format(savename), index=False)

from sklearn.metrics import confusion_matrix

#混淆矩阵的绘制
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix


def draw_confusion_matrix(label_true, label_pred, label_name, title="Confusion Matrix", pdf_save_path=None, dpi=100):
    """

    @param label_true: 真实标签，比如[0,1,2,7,4,5,...]
    @param label_pred: 预测标签，比如[0,5,4,2,1,4,...]
    @param label_name: 标签名字，比如['cat','dog','flower',...]
    @param title: 图标题
    @param pdf_save_path: 是否保存，是则为保存路径pdf_save_path=xxx.png | xxx.pdf | ...等其他plt.savefig支持的保存格式
    @param dpi: 保存到文件的分辨率，论文一般要求至少300dpi
    @return:

    example：
            draw_confusion_matrix(label_true=y_gt,
                          label_pred=y_pred,
                          label_name=["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"],
                          title="Confusion Matrix on Fer2013",
                          pdf_save_path="Confusion_Matrix_on_Fer2013.png",
                          dpi=300)

    """
    x_value="Actual Label"
    y_value="Predicted Label"
    # 设置全局字体为 Arial
    plt.rcParams['font.family'] = 'Arial'
    plt.rcParams['font.serif'] = ['Arial']

    cm = confusion_matrix(y_true=label_true, y_pred=label_pred, normalize='true')

    # 创建图形和坐标轴
    fig, ax = plt.subplots(figsize=(8, 6))

    # 关闭网格线与刻度线
    ax.grid(False)
    ax.tick_params(top=False, bottom=False, left=False, right=False)

    # 绘制混淆矩阵
    cax = ax.imshow(cm, cmap='Blues',vmin=0,vmax=1)

    # 设置标题和标签
    # ax.set_title(title, fontdict={'fontsize': 20, 'family': 'Arial','weight': 'bold'},pad=20)
    ax.set_yticks(range(len(label_name)))
    ax.set_yticklabels(label_name, fontdict={'fontsize': 32, 'family': 'Arial'})
    ax.set_xticks(range(len(label_name)))
    ax.set_xticklabels(label_name, fontdict={'fontsize': 32, 'family': 'Arial'}, rotation=45)

    # 设置坐标轴标签字体大小
    ax.set_xlabel(x_value.capitalize(), fontdict={'fontsize': 36, 'family': 'Arial','weight': 'bold'})
    ax.set_ylabel(y_value.capitalize(), fontdict={'fontsize': 36, 'family': 'Arial','weight': 'bold'})

    plt.tight_layout()

    plt.colorbar(cax)

    for i in range(label_name.__len__()):
        for j in range(label_name.__len__()):
            color = (1, 1, 1) if i == j else (0, 0, 0)  # 对角线字体白色，其他黑色
            value = float(format('%.2f' % cm[j, i]))
            plt.text(i, j, value, verticalalignment='center', horizontalalignment='center', color=color, fontdict={'fontsize': 32, 'family': 'Arial','weight': 'bold'})

    # plt.show()
    if not pdf_save_path is None:
        plt.savefig(pdf_save_path, bbox_inches='tight', dpi=dpi)





