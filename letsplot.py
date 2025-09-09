import math

import numpy as np
import pandas as pd
import matplotlib
# matplotlib.use('Agg') # 使用 Agg 渲染器
from matplotlib import pyplot as plt
from sklearn import metrics
from sklearn.metrics import r2_score
import seaborn as sns

#相关系数计算函数
def calc_corr(a,b):
    a_avg = sum(a)/len(a)
    b_avg = sum(b)/len(b)
    cov_ab = sum([(x - a_avg)*(y - b_avg) for x,y in zip(a, b)])
    sq = math.sqrt(sum([(x - a_avg)**2 for x in a])*sum([(x - b_avg)**2 for x in b]))
    corr_factor = cov_ab/sq
    return corr_factor

#作图
def letsplot(train,trainpre,test,testpre,modelname = 'ML',target = 'PCE'):
    fontsize = 12
    plt.figure(figsize=(3,3))
    plt.style.use('default')
    plt.rc('xtick', labelsize=fontsize)
    plt.rc('ytick', labelsize=fontsize)
    plt.rcParams['font.family']="Arial"

    a = plt.scatter(train, trainpre, s=25,c='#b2df8a')
    plt.plot([train.min(), train.max()], [train.min(),train.max()], 'k:', lw=1.5)
    plt.xlabel('PCE Observation', fontsize=fontsize)
    plt.ylabel('PCE Prediction', fontsize=fontsize)
    plt.tick_params(direction='in')
    plt.title('{} model for {} prediction'.format(modelname,target),fontsize=fontsize)

    b = plt.scatter(test, testpre, s=25,c='#4682B4')
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


from sklearn.metrics import mean_squared_error

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

# 使用weplot函数进行画图
def weplot(y_train, y_pred_train, y_test, y_pred_test, modelname = 'ML',target = 'PCE'):

    # plt.rcParams['font.family'] = 'Times New Roman'
    fig, ax = plt.subplots(figsize=(5,5))

    ax.scatter(y_train, y_pred_train, color='c', alpha=0.5, label='Train', s=30)
    ax.scatter(y_test, y_pred_test, color='m', alpha=0.5, label='Test', s=30)

    ax.set_xlabel('Target Labels', fontsize=16)
    ax.set_ylabel('Predictions', fontsize=16)
    ax.set_title('{} for {} prediction'.format(modelname,target), fontsize=18)

    ax.set_xlim(0, 26)
    ax.set_ylim(0, 26)
    maxlength = max(ax.get_xlim()[1] - ax.get_xlim()[0], ax.get_ylim()[1] - ax.get_ylim()[0])
    ax.plot([-0.1 * maxlength, 1.1 * maxlength], [-0.1 * maxlength, 1.1 * maxlength], ls="--", c=".3")
    ax.legend(loc='upper left',  fontsize='small', prop={'size': 15})

    # Train R-square and Test R-square
    test_r2 = r2_score(y_test, y_pred_test)
    train_text1 = 'Train R: {:.4f}'.format(calc_corr(y_train, y_pred_train))
    test_text1 = 'Test R: {:.4f}'.format(calc_corr(y_test, y_pred_test))
    test_text2 = 'Test R2: {:.4f}'.format(test_r2)

    ax.text(0.6, 0.25, train_text1, transform=ax.transAxes, fontsize=14, va='top', ha='left')
    ax.text(0.6, 0.19, test_text1, transform=ax.transAxes, fontsize=14, va='top', ha='left')
    ax.text(0.6, 0.13, test_text2, transform=ax.transAxes, fontsize=14, va='top', ha='left')

    # Test RMSE
    test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
    rmse_text = 'Test RMSE: {:.3f}'.format(test_rmse)
    ax.text(0.6, 0.07, rmse_text, transform=ax.transAxes, fontsize=14, va='top', ha='left')

    fig.tight_layout()
    plt.show()



def getdata(y_train, y_train_hat,y_test, y_test_hat):
    data_train = np.column_stack((y_train, y_train_hat))
    data_test = np.column_stack((y_test, y_test_hat))
    data = pd.DataFrame(np.append(data_train, data_test, axis=0), columns=["Observation", "Prediction"])
    data['type'] = ['Train' if i < len(data_train) else 'Test' for i in range(len(data))] # 添加类型列
    return data


def myscatterplotnotime(y_train, y_train_hat,y_test, y_test_hat,modelname="ML", target="PCE",plot_height = 6,savepic = False,picname = 'picname'):

    data = getdata(y_train, y_train_hat,y_test, y_test_hat)
    # 定义默认参数
    # plot_height = 6
    dpi = 300
    plot_aspect = 1.2
    plot_palette = ["#BC3C28", "#0072B5"]
    plot_scatter_kw = {"edgecolor": "black"}
    face_color = "white"
    spine_color = "white"
    label_size = 15
    direction = 'in'
    grid_which = 'major'
    grid_ls = '--'
    grid_c = 'k'
    grid_alpha = .6
    xlim_left = -0.1
    xlim_right = 26
    ylim_bottom = -0.1
    ylim_top = 26

    x_value="Observation"
    y_value="Prediction"
    hue_value="type"
    hue_order_values=['Train', 'Test']

    title_fontdict = {"size": 23, "color": "k"}
    text_fontdict = {'size': '22', 'weight': 'bold', 'color': 'black'}

    # 创建一个正方形画布
    fig, ax = plt.subplots(figsize=(plot_height, plot_height),dpi=dpi)

    # 调用scatterplot函数在空坐标系上绘图
    sns.scatterplot(x=x_value, y=y_value, hue=hue_value, hue_order=hue_order_values, data=data, s=90, alpha=.65,
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
        plt.savefig('./img/{}.png'.format(picname))
    else:
        pass

    # 显示图像
    plt.show()


def myscatterplot(y_train, y_train_hat,y_test, y_test_hat,modelname="ML", target="PCE",plot_height = 6,savepic = False,picname = 'picname'):

    data = getdata(y_train, y_train_hat,y_test, y_test_hat)
    # 定义默认参数
    # plot_height = 6
    dpi = 300
    plot_aspect = 1.2
    plot_palette = ["#BC3C28", "#0072B5"]
    plot_scatter_kw = {"edgecolor": "black"}
    face_color = "white"
    spine_color = "white"
    label_size = 15
    direction = 'in'
    grid_which = 'major'
    grid_ls = '--'
    grid_c = 'k'
    grid_alpha = .6
    xlim_left = -0.1
    xlim_right = 26
    ylim_bottom = -0.1
    ylim_top = 26

    x_value="Observation"
    y_value="Prediction"
    hue_value="type"
    hue_order_values=['Train', 'Test']

    title_fontdict = {"size": 23, "color": "k", 'family': 'Times New Roman'}
    text_fontdict = {'family': 'Times New Roman', 'size': '22', 'weight': 'bold', 'color': 'black'}

    # 创建一个正方形画布
    fig, ax = plt.subplots(figsize=(plot_height, plot_height),dpi=dpi)

    # 调用scatterplot函数在空坐标系上绘图
    sns.scatterplot(x=x_value, y=y_value, hue=hue_value, hue_order=hue_order_values, data=data, s=90, alpha=.65,
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
        plt.savefig('./img/{}.png'.format(picname))
    else:
        pass

    # 显示图像
    plt.show()


def myscatterplot2(y_train, y_train_hat,y_test, y_test_hat,modelname="ML", target="PCE",plot_height = 6,savepic = False,picname = 'picname'):

    data = getdata(y_train, y_train_hat,y_test, y_test_hat)
    # 定义默认参数
    # plot_height = 6
    plot_aspect = 1.2
    plot_palette = ["#BC3C28", "#0072B5"]
    plot_scatter_kw = {"edgecolor": "black"}
    face_color = "white"
    spine_color = "white"
    label_size = 15
    direction = 'in'
    grid_which = 'major'
    grid_ls = '--'
    grid_c = 'k'
    grid_alpha = .6

    xlim_left = y_train.min()
    xlim_right = y_train.max()
    ylim_bottom = y_train.min()
    ylim_top = y_train.max()

    x_value="Observation"
    y_value="Prediction"
    hue_value="type"
    hue_order_values=['Train', 'Test']

    title_fontdict = {"size": 23, "color": "k", 'family': 'Arial'}
    text_fontdict = {'family': 'Arial', 'size': '22', 'weight': 'bold', 'color': 'black'}

    # 创建一个正方形画布
    fig, ax = plt.subplots(figsize=(plot_height, plot_height))

    # 调用scatterplot函数在空坐标系上绘图
    sns.scatterplot(x=x_value, y=y_value, hue=hue_value, hue_order=hue_order_values, data=data, s=90, alpha=.65,
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
    ax.set_xlabel(x_value.capitalize(), fontdict={'fontsize': 25, 'family': 'Arial'})
    ax.set_ylabel(y_value.capitalize(), fontdict={'fontsize': 25, 'family': 'Arial'})
    ax.plot([-25, 50], [-25, 50], linestyle='--', color='gray', linewidth=2)

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
        plt.savefig('{}.png'.format(picname),dpi=300,bbox_inches='tight')
    else:
        pass

    # 显示图像
    plt.show()




    
def myscatterplot3(y_train, y_train_hat,y_test, y_test_hat,modelname="ML", target="PCE",plot_height = 6,savepic = False,picname = 'picname'):

    data = getdata(y_train, y_train_hat,y_test, y_test_hat)
    # 定义默认参数
    # plot_height = 6
    plot_aspect = 1.2
    plot_palette = ["#BC3C28", "#0072B5"]
    plot_scatter_kw = {"edgecolor": "black"}
    face_color = "white"
    spine_color = "white"
    label_size = 15
    direction = 'in'
    grid_which = 'major'
    grid_ls = '--'
    grid_c = 'k'
    grid_alpha = .6

    xlim_left = 8
    xlim_right = 27
    ylim_bottom = 8
    ylim_top = 27

    x_value="Observation"
    y_value="Prediction"
    hue_value="type"
    hue_order_values=['Train', 'Test']

    title_fontdict = {"size": 23, "color": "k", 'family': 'Times New Roman'}
    text_fontdict = {'family': 'Times New Roman', 'size': '22', 'weight': 'bold', 'color': 'black'}

    # 创建一个正方形画布
    fig, ax = plt.subplots(figsize=(plot_height, plot_height))

    # 调用scatterplot函数在空坐标系上绘图
    sns.scatterplot(x=x_value, y=y_value, hue=hue_value, hue_order=hue_order_values, data=data, s=90, alpha=.65,
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
    ax.plot([-5, 40], [-5, 40], linestyle='--', color='gray', linewidth=2)

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
        plt.savefig('./img/{}.png'.format(picname))
    else:
        pass

    # 显示图像
    plt.show()





#     # 试验一下画图
# from rdkit.Chem import Draw
# from rdkit import Chem
# from rdkit import Chem
# from rdkit.Chem import AllChem
# import numpy as np
# from rdkit.Chem import DataStructs

# # 蓝色: 蓝色标注说明该原子是中心原子
# # 黄色：说明该原子是芳香原子
# # 灰色： 说明该原子是脂肪烃原子

# def drawit(onbits,n_bits = 1024,df111 = df111):
#     # 按列名提取指定的 one-hot 编码列
#     fp_col_name = "fp_{}".format(onbits)  # 提取的one-hot 编码列名
#     idx= df111[df111[fp_col_name] == 1].index.tolist()
#     print('数据集中含有该分子指纹的条数为',len(idx))
#     print(df111["Smiles"][idx])
#     # 计算 Morgan 指纹，返回包含图像数据的列表
#     imgs=[]
#     for i in idx:
#         bi = {}
#         mol = Chem.MolFromSmiles(df111["Smiles"][i])
#         fp = AllChem.GetMorganFingerprintAsBitVect(mol,nBits=n_bits, radius=2, bitInfo=bi)
#         mfp2_svg = Draw.DrawMorganBit(mol, onbits, bi)
#         imgs.append(mfp2_svg)
#     return imgs

# # 绘制图像
# def displayimgsinrow(onbits, col=4):
#     imgs = drawit(onbits)
#     num_imgs = len(imgs)
#     rows = (num_imgs - 1) // col + 1
#     cols = min(num_imgs, col)
#     fig, axes = plt.subplots(rows, cols, figsize=(cols*5, rows*5))
#     for i, ax in enumerate(axes.ravel()):
#         if i < num_imgs:
#             ax.imshow(imgs[i])
#             ax.set_axis_off()
#         else:
#             fig.delaxes(ax)
#     plt.show()


def myscatterplot111(y_train, y_train_hat, y_test, y_test_hat, modelname="ML", target="PCE", plot_height=8, savepic=False, picname='picname',label1 = 'Dataset',label2 = 'Experiment'):
    fig, ax = plt.subplots(figsize=(8, plot_height))
    
    # 画训练集的散点图，设置点的大小为10，形状为圆形，颜色为蓝色
    ax.scatter(y_train, y_train_hat, s=10, marker='o', color="#0072B5", label=label1)
    
    # 画测试集的散点图，设置点的大小为20，形状为三角形，颜色为红色
    ax.scatter(y_test, y_test_hat, s=100, marker='^', color="#BC3C28", label=label2)
    
    ax.set_xlabel('Observation')
    ax.set_ylabel('Prediction')
    ax.set_title('{}'.format(modelname))
    ax.legend()

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
        plt.savefig('./img/{}.png'.format(picname),bbox_inches = 'tight') #,transparent = True
    else:
        pass
    plt.show()


from sklearn.metrics import mean_absolute_error
import numpy as np
from sklearn.model_selection import cross_val_score, KFold
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt

def CVS(regressor,X ,Y,rs = 0):
    regressor = RandomForestRegressor()
    PCE_clf = regressor

    kf = KFold(n_splits=10, shuffle=True, random_state=rs)
    scores_r2 = cross_val_score(PCE_clf, X, Y, cv=kf, scoring='r2')
    scores_mae = -cross_val_score(PCE_clf, X, Y, cv=kf, scoring='neg_mean_absolute_error')
    scores_mse = -cross_val_score(PCE_clf, X, Y, cv=kf, scoring='neg_mean_squared_error')
    
    average_score_r2 = np.mean(scores_r2)
    average_score_mae = np.mean(scores_mae)
    average_scores_mse = np.mean(scores_mse)

    print("r2 Scores:", scores_r2)
    print("*************************************")
    print("Average r2 Scores:", average_score_r2)
    print("Average MAE Scores:", average_score_mae)
    print("Average MSE Scores:", average_scores_mse)

    plt.plot(range(1, 11), scores_r2, marker='o', label='r2 Score')
    # plt.plot(range(1, 11), np.abs(scores_mae), marker='o', label='MAE')
    plt.xlabel('Fold')
    plt.ylabel('Score')
    plt.title('Cross Validation Scores')
    plt.legend()
    plt.show()

    return scores_r2,scores_mae,scores_mse

def CVSMAE(regressor, X, Y, rs=0):
    regressor = RandomForestRegressor()
    PCE_clf = regressor

    kf = KFold(n_splits=10, shuffle=True, random_state=rs)
    scores_r2 = cross_val_score(PCE_clf, X, Y, cv=kf, scoring='r2')
    scores_mae = -cross_val_score(PCE_clf, X, Y, cv=kf, scoring='neg_mean_absolute_error')
    scores_mse = -cross_val_score(PCE_clf, X, Y, cv=kf, scoring='neg_mean_squared_error')
    scores_rmse = np.sqrt(-cross_val_score(PCE_clf, X, Y, cv=kf, scoring='neg_mean_squared_error'))

    average_score_r2 = np.mean(scores_r2)
    average_score_mae = np.mean(scores_mae)
    average_scores_rmse = np.mean(scores_rmse)
    print("r2 Scores:", scores_r2)
    print("*************************************")
    print("Average r2 Scores:", average_score_r2)
    print("Average MAE Scores:", average_score_mae)
    print("Average RMSE Scores:", average_scores_rmse)

    plt.figure(figsize=(10, 5))
    plt.bar(range(1, 11), scores_mae, align='center', alpha=0.5)
    plt.xlabel('Fold')
    plt.ylabel('MAE Score')
    plt.title('Cross Validation Scores - MAE')
    plt.show()

    return scores_r2, scores_mae, scores_mse

def CVSRMSE(regressor, X, Y, rs=0):
    regressor = RandomForestRegressor()
    PCE_clf = regressor

    kf = KFold(n_splits=10, shuffle=True, random_state=rs)
    scores_r2 = cross_val_score(PCE_clf, X, Y, cv=kf, scoring='r2')
    scores_mae = -cross_val_score(PCE_clf, X, Y, cv=kf, scoring='neg_mean_absolute_error')
    scores_rmse = np.sqrt(-cross_val_score(PCE_clf, X, Y, cv=kf, scoring='neg_mean_squared_error'))

    average_score_r2 = np.mean(scores_r2)
    average_score_mae = np.mean(scores_mae)
    average_scores_rmse = np.mean(scores_rmse)

    print("r2 Scores:", scores_r2)
    print("*************************************")
    print("Average r2 Scores:", average_score_r2)
    print("Average MAE Scores:", average_score_mae)
    print("Average RMSE Scores:", average_scores_rmse)

    plt.figure(figsize=(10, 5))
    plt.bar(range(1, 11), scores_rmse, align='center', alpha=0.5)
    plt.xlabel('Fold')
    plt.ylabel('RMSE Score')
    plt.title('Cross Validation Scores - RMSE')
    plt.show()

    return scores_r2, scores_mae, scores_rmse


import pandas as pd
import numpy as np

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