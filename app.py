import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import seaborn as sns

from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

from io import BytesIO
import pickle

import i18n
from languages import language_init, translate_list


@st.cache  # in order to prevent from run each operation whenever the page loads
def load_data():
    df = pd.read_csv('./S&P500_Stock_Data.csv')

    return df


def load_model(model):
    with open(model, 'rb') as file:
        data = pickle.load(file)

    return data


def show_metrics(y_test, y_pred, n, k):
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    mape = np.mean(np.abs((y_test-y_pred) / y_test)) * 100
    r2 = r2_score(y_test, y_pred)
    adj_r2 = (1-(1-r2) * (n-1)/(n-k-1))

    dict_metrics = {
        'RMSE': rmse,
        'MSE': mse,
        'MAE': mae,
        'MAPE': mape,
        'R2': r2,
        'Adjusted R2': adj_r2
    }

    df_metrics = pd.DataFrame(dict_metrics, index=[""])
    st.table(df_metrics)


def show_predictions_comparison(actual, pred):
    st.metric(i18n.t('Actual value'), round(actual, 2))
    st.metric(i18n.t('Prediction value'), round(pred, 2))
    st.metric("Error", round(np.abs(actual - pred), 2))


def show_3d_graph(X_test, y_test, x_surf, y_surf, z_pred_surf, y_pred, dot_idx, vi_elev=-140, vi_azim=60):
    fig = plt.figure(figsize=(6, 6))

    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(X_test.iloc[dot_idx, 0], X_test.iloc[dot_idx, 1], y_test.iloc[dot_idx], 
               color='b', marker ='.')
    ax.plot_surface(x_surf, y_surf, z_pred_surf, color='r', alpha=0.2)
    ax.scatter(X_test.iloc[dot_idx, 0], X_test.iloc[dot_idx, 1], y_pred[dot_idx], 
               color='w', marker ='.', alpha=0.4)

    ax.set_xlabel(i18n.t('Interest Rates'))
    ax.set_ylabel(i18n.t('Employment'))
    ax.set_zlabel(i18n.t('S&P 500 Price'))
    # Set the camera
    ax.view_init(vi_elev, vi_azim)
    plt.title(i18n.t('title_final_graph'))
    
    redlight_patch = mpatches.Patch(color='red', alpha=0.2, label=i18n.t('legend_1_final_graph'))
    red_patch = mpatches.Patch(color='red', label=i18n.t('legend_2_final_graph'))
    blue_patch = mpatches.Patch(color='blue', label=i18n.t('legend_3_final_graph'))

    plt.legend(handles=[redlight_patch, red_patch, blue_patch])

    return fig
    


language_init()

st.markdown(i18n.t('title'), unsafe_allow_html=True)
st.markdown('\n')
st.markdown(i18n.t('intro_1'))
st.markdown(i18n.t('intro_2'))
df = load_data()
st.dataframe(pd.DataFrame(df.values, index=df.index, columns=translate_list(df.columns)))
st.markdown("\n")

df_describe = df.describe()
st.table(pd.DataFrame(df_describe.values, index=translate_list(df_describe.index), columns=translate_list(df_describe.columns)))
st.markdown(i18n.t('comment_describe'))
st.markdown("\n")

st.markdown(i18n.t('comment_graph_1'))
st.markdown('\n')

buf = BytesIO()
col1, col2 = st.columns(2)

fig1 = sns.jointplot(x='Employment', y='S&P 500 Price', data=df)
fig1.ax_joint.set_xlabel(i18n.t('Employment'))
fig1.ax_joint.set_ylabel(i18n.t('S&P 500 Price'))
fig1.savefig(buf, format="png")
col1.markdown(i18n.t('title_graph_1a'))
col1.image(buf)

fig2 = sns.jointplot(x='Interest Rates', y='S&P 500 Price', data=df)
fig2.ax_joint.set_xlabel(i18n.t('Interest Rates'))
fig2.ax_joint.set_ylabel(i18n.t('S&P 500 Price'))
fig2.savefig(buf, format="png")
col2.markdown(i18n.t('title_graph_1b'))
col2.image(buf)
st.markdown('\n')

st.markdown(i18n.t('comment_graph_2'))
st.markdown('\n')
fig3 = sns.pairplot(df)
st.markdown(i18n.t('title_graph_2'))
x_labels = ['S&P 500 Price', 'Employment', 'Interest Rates']
axes = fig3.axes
axes[2, 0].set_xlabel(i18n.t(f'{x_labels[0]}'))
axes[2, 1].set_xlabel(i18n.t(f'{x_labels[1]}'))
axes[2, 2].set_xlabel(i18n.t(f'{x_labels[2]}'))
axes[0, 0].set_ylabel(i18n.t(f'{x_labels[2]}'))
axes[1, 0].set_ylabel(i18n.t(f'{x_labels[1]}'))
axes[2, 0].set_ylabel(i18n.t(f'{x_labels[0]}'))
st.pyplot(fig3)
st.markdown('\n')

data = load_model('./saved_model_p2.pkl')
model = data['model']
X_train = data['X_train']
X_test = data['X_test']
y_train = data['y_train']
y_test = data['y_test']

st.markdown(i18n.t('comment_plane'))
st.markdown(i18n.t('S&P 500 Price') + " = " + str(round(model.coef_[0], 2)) + " • " + i18n.t('Interest Rates') + " \+ " + str(round(model.coef_[1], 2)) + " • " + i18n.t('Employment') + "\+ " + str(round(model.intercept_, 2)))
st.markdown('\n')
st.markdown(i18n.t('comment_parameters'))


y_pred = model.predict(X_test)
k = X_test.shape[1]
n = len(X_test)
show_metrics(y_test, y_pred, n, k)

st.markdown(i18n.t('comment_params_result'))
st.markdown('\n')

st.markdown(i18n.t('comment_final_1'))
st.markdown(i18n.t('comment_final_2'))

x_surf, y_surf = np.meshgrid(np.linspace(df['Interest Rates'].min(), df['Interest Rates'].max(), 100), np.linspace(df['Employment'].min(), df['Employment'].max() , 100))
df_xy_ravelled = pd.DataFrame({
    'Interest Rates': x_surf.ravel(), 
    'Employment': y_surf.ravel()
})
z_pred_ravelled = model.predict(df_xy_ravelled)
z_pred_surf = z_pred_ravelled.reshape(x_surf.shape)

col1, col2 = st.columns((4, 1))

with col2:
    dot_idx = st.select_slider(i18n.t('test_data_point'), np.arange(0, len(X_test - 1)))
    show_predictions_comparison(y_test.iloc[dot_idx], y_pred[dot_idx])
    vi_elev = st.select_slider(i18n.t('cam_elevation'), np.arange(-180, 181), -140)
    vi_azim = st.select_slider(i18n.t('cam_rotation'), np.arange(-180, 181), 60)

fig_3d = show_3d_graph(X_test, y_test, x_surf, y_surf, z_pred_surf, y_pred, dot_idx, vi_elev, vi_azim)
fig_3d.savefig(buf, format="png")
col1.image(buf)
