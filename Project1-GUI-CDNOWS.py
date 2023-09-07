import streamlit as st
import pandas as pd
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import squarify
from datetime import datetime
import scipy
import pickle
from sklearn.preprocessing import RobustScaler
from feature_engine.wrappers import SklearnTransformerWrapper
import base64
import warnings
warnings.filterwarnings("ignore")


df = pd.read_csv('CDNOW_master.txt', names=['customer_id', 'date', 'quantity', 'price'], sep = "\s+", index_col=None)

# chuyển cột date về kiểu datetime
df.date = df.date.astype(str)
df.date= pd.to_datetime(df.date, infer_datetime_format=True)

# ### markdown: right
# from pathlib import Path
# def img_to_bytes(img_path):
#     img_bytes = Path(img_path).read_bytes()
#     encoded = base64.b64encode(img_bytes).decode()
#     return encoded
# def img_to_html(img_path, width, height):
#     img_html = "<img src='data:image/png;base64, {}' class='img-fluid' width='{}' height='{}'>".format(
#         img_to_bytes(img_path), width, height
#     )
#     return img_html

# st.markdown("<p style='text-align: center; color: grey;'>"+img_to_html('images/CapstoreProject.png', 700, 350)+"</p>", unsafe_allow_html=True)
#----------------------------------------------------------------------------------------------------------------------------------#

# Tạo cột bên trái cho menu
left_column = st.sidebar
# Chèn đoạn mã HTML để tùy chỉnh giao diện của chữ "Chọn dự án" : left_column.markdown('<span style="font-weight: bold; color: blue;">Chọn dự án</span>', unsafe_allow_html=True)
# Tạo danh sách các dự án
projects =  ['Project 1: Customer Segmentation','Project 2: Recommendation System', 'Project 3: Sentiment Analysis']

# Tạo menu dropdown list cho người dùng lựa chọn dự án
project = left_column.selectbox(":blue[**Select project:**]", projects)

# Lưu trữ chỉ số index của dự án được chọn
project_num = projects.index(project) + 1

if project_num == 1:
    # Hiển thị tên của dự án 
    st.subheader("Project 1: Customer Segmentation")
    # Hiển thị danh sách các tùy chọn cho người dùng lựa chọn từng bước trong dự án
    step = left_column.radio('Chọn bước', ['Business Understanding', 'Preprocessing + EDA', 'Applicable models', 'Prediction'])
    
    # Xử lý sự kiện khi người dùng lựa chọn từng mục trong danh sách và hiển thị hình ảnh tương ứng
    if step == 'Business Understanding':

        image2 = st.image('a1.jpg')

        st.markdown('### Từ mục tiêu/ vấn đề đã xác định: Xem xét các dữ liệu cần thiết')

        st.markdown('''
        The file CDNOW_master.txt contains the entire purchase history up to the end of June 1998 of the cohort of 23,570 individuals who made their first-ever purchase at CDNOW in the first quarter of 1997. This CDNOW dataset was first used by Fader and Hardie (2001).

        Each record in this file, 69,659 in total, comprises four fields: the customer's ID, the date of the transaction, the number of CDs purchased, and the dollar value of the transaction.

        See [Notes on the CDNOW Master Data Set](http://brucehardie.com/notes/026/) for details of how [the 1/10th systematic sample](http://brucehardie.com/datasets/CDNOW_sample.zip) used in many papers was created. 

        Reference:

        Fader, Peter S. and Bruce G.,S. Hardie, (2001), "Forecasting Repeat Sales at CDNOW: A Case Study," Interfaces, 31 (May-June), Part 2 of 2, S94-S107.''')

        st.markdown('### Data preparation/ Prepare')

        image = Image.open('processing.png')

        st.image(image, caption='Data Science Processing')

        st.markdown('### Modeling & Evaluation/ Analyze & Report')

        st.markdown('#### Xây dựng giải pháp phân cụm khách hàng theo RFM.')

        st.markdown('#### Xây dựng model phân cụm khách hàng theo RFM phối hợp với thuật toán phân cụm:')

        st.markdown('''
        * RFM + Kmeans (LDS6)
        * RFM + Hierarchical Clustering (LDS6)
        * RFM + Kmeans (LDS9)
        * Các đề xuất khác (+++)''')

        st.markdown('#### Thực hiện so sánh/đánh giá các kết quả')

        st.markdown('### Deployment & Feedback/ Act')

        st.markdown('#### Đưa ra chiến dịch quảng cáo, bán hàng, chăm sóc khách hàng phù hợp cho mỗi nhóm.')

    elif step == 'Preprocessing + EDA':

        st.markdown('### Dataframe:')
        st.dataframe(df.head())
        
        st.write('**Kiểm tra dữ liệu trùng và thiếu:**')
        st.code('Dữ liệu trùng '+ str(df.isnull().any().sum()))
        st.write('Không có dữ liệu thiếu')
        st.code('Dữ liệu thiếu ' + str(df.duplicated().sum()))
        st.write('Loại bỏ dữ liệu trùng')
        df =df.drop_duplicates()

        df['Year']= pd.DatetimeIndex(df['date']).year
        df['Year'] = df['Year'].astype(str)

        # Doanh thu
        st.write('**Doanh thu qua các năm**')
        price_year = df.groupby('Year').agg(revenue=("price", 'sum'))
        price_year.reset_index(inplace=True)
        st.dataframe(price_year)
        
        
        price_year_plot = px.bar(price_year,x='Year', y='revenue', text_auto = True, color_discrete_sequence =['#903749'])
        st.plotly_chart(price_year_plot)
        

        # lượt mua
        st.write('**Lượt mua qua các năm**')
        customer_year = df.groupby('Year')['customer_id'].count()
        customer_year = customer_year.reset_index()
        st.dataframe(customer_year)
        
        customer_year_plot = px.bar(customer_year,x='Year', y='customer_id', text_auto = True, color_discrete_sequence =['#11999E'])
        st.plotly_chart(customer_year_plot)
        

        # customer_year_plot = sns.barplot(x='Year', y='customer_id', data=customer_year)
        # customer_year_plot.bar_label(customer_year_plot.containers[0], fmt='%.1f')
        # customer_year_plot.set_title('Number of Customer in Each Year')
        # st.pyplot(customer_year_plot.get_figure())

        st.write('**Price and Time**')
        st.line_chart(x='date', y='price', data=df)
        
        df1 = df.reset_index().set_index('date')[['price']].resample(rule="MS").sum()
        st.line_chart(df1)

        st.markdown('''
        **Nhận xét:**
        * Có thể thấy năm 1997 cao hơn năm 1998 cả về số lượng đĩa CD bán ra và doanh thu
        * Doanh thu từ việc bán giảm mạnh trong khoảng từ tháng 01/1997 đến tháng 04/1997, sau đó không có sự thay đổi quá lớn ''')
        
        st.write('**Tương quan giữa các biến**')
        st.code(df.corr())
        
        st.write('Ngoài số lượng bán và doanh thu có tương quan thuận mạnh với nhau, các biến còn lại không tương quan')
        
        st.write('**Outliers**')
        # Number of upper, lower outliers
        def check_outlier(df, feature):
            fig, ax = plt.subplots()
            ax.boxplot(feature)
            st.pyplot(fig)
            Q1 = np.percentile(feature, 25)
            Q3 = np.percentile(feature, 75)
            n_O_upper = df[feature > (Q3 + 1.5*scipy.stats.iqr(feature))].shape[0]
            print("Number of upper outliers:", n_O_upper)
            n_O_lower = df[feature < (Q1 - 1.5*scipy.stats.iqr(feature))].shape[0]
            print("Number of lower outliers:", n_O_lower)
            # Percentage of outliers
            outliers_per = (n_O_upper+n_O_lower)/df.shape[0]
            print("Percentage of outliers:", outliers_per)
            return Q1, Q3, n_O_upper, n_O_lower, outliers_per
        st.write('Quantity')
        st.code(check_outlier(df, df.quantity))
        
        st.write('Price')
        st.code(check_outlier(df, df.price))
        
        st.write('Biến quantity và price đều có outliers, số lượng không nhiều vì vậy có thể loại bỏ các outliers này mà không ảnh hưởng lớn đến dữ liệu, tuy nhiên, do đây là dữ liệu về lượt mua đĩa CD vì vậy các giá trị ngoại lai này cũng có thể hiểu là một hành vi bất thường không hiếm gặp của khách hàng mua sắm. => đối với data này xem xét giữ lại outliers để tính toán')


    elif step == 'Applicable models':
        models = ['RFM Analysis', 'RFM + K-means', 'RFM + Hierarchical Clustering', 'RFM + K-means (Big Data)', 'Compare models']
        model = st.sidebar.selectbox('Chọn mô hình', models)

        if model == 'RFM Analysis':
            df =df.drop_duplicates()
            # RFM
            # Convert string to date, get max date of dataframe
            max_date = df['date'].max().date()

            Recency = lambda x : (max_date - x.max().date()).days
            Frequency  = lambda x: x.count()
            Monetary = lambda x : round(sum(x), 2)

            st.markdown('### Dataframe')

            df_RFM = df.groupby('customer_id').agg({'date': Recency,
                                                    'customer_id': Frequency,
                                                    'price': Monetary })
            st.dataframe(df_RFM.head())

            # reanme the columns of df
            df_RFM.columns = ['Recency', 'Frequency', 'Monetary']

            # descending sorting
            df_RFM = df_RFM.sort_values('Monetary', ascending=False)
            
            st.markdown('### RFM Dataframe')
            
            st.dataframe(df_RFM.head())

            st.write('Shape of the dataframe:',df_RFM.shape)

            ### Virsulization
            # fig, axs = plt.subplots(3, figsize=(8,10))

            # plot1 = sns.distplot(df_RFM['Recency'])# Plot distribution of R
            # plot2 = sns.distplot(df_RFM['Frequency'])# Plot distribution of F
            # plot3 = sns.distplot(df_RFM['Monetary']) # Plot distribution of M

            dist =  Image.open('displot.png')
            st.image(dist)

            
            st.write('**Nhận xét:** Frequency, Monetary: lệch phải')
            
            st.markdown('#### Calculate RFM')
            t0 = datetime.now()

            # assign these labels to 4 equal percentile groups
            df_RFM['R'] = pd.qcut(df_RFM['Recency'].rank(method='first'), q=5, labels=[4, 3, 2, 1, 0])
            df_RFM['F'] = pd.qcut(df_RFM['Frequency'].rank(method='first'), q=5, labels=[0, 1, 2, 3, 4])
            df_RFM['M'] = pd.qcut(df_RFM['Monetary'].rank(method='first'), q=5, labels=[0, 1, 2, 3, 4])
            
            # code = '''    
            # df_RFM['R'] = pd.qcut(df_RFM['Recency'].rank(method='first'), q=5, labels=[4, 3, 2, 1, 0])
            # df_RFM['F'] = pd.qcut(df_RFM['Frequency'].rank(method='first'), q=5, labels=[0, 1, 2, 3, 4])
            # df_RFM['M'] = pd.qcut(df_RFM['Monetary'].rank(method='first'), q=5, labels=[0, 1, 2, 3, 4]) '''
            
            # st.code(code, language='python')

            
            def join_rfm(x): return str(int(x['R'])) + str(int(x['F'])) + str(int(x['M']))
            df_RFM['RFM_Segment'] = df_RFM.apply(join_rfm, axis=1)
            
            df_RFM['RFM_Score'] = df_RFM[['R','F','M']].sum(axis=1)

            segment_dict = {12:'Champion', 11:'Loyal Customers', 10:'Promising', 9:'New Customers', 8:'Abandoned Checkouts', 7:'Callback Requests', 6:'Warm_Leads', 5:'Cold Leads', 4:'Need Attention', 3:'Should not Lose', 2:'Sleepers', 1:'Lost', 0:'Lost'}

            df_RFM['RFM_Level'] = df_RFM['RFM_Score'].map(segment_dict)
            
            st.markdown('#### RFM Dataframe')
            
            st.dataframe(df_RFM.head())
            
            st.markdown('**RFM Groups**')

            level = df_RFM.RFM_Level.value_counts().reset_index()
            st.dataframe(level.style.background_gradient(cmap='Blues'))
            
            df_report = pd.concat([df, df_RFM], axis=1)
            df_report['Year'] = df_report['date'].dt.year
            grouped_df_report_year= df_report.groupby(['Year', 'RFM_Level'])['quantity'].sum().reset_index().sort_values(by=["RFM_Level","Year"],ascending=[True,True])
            grouped_df_report_year['Year'] = grouped_df_report_year['Year'].astype(str)  
            
            grouped_df_report_year_plot = px.line(grouped_df_report_year, x='Year', y='quantity', color='RFM_Level', title='Order Amount by Customer Group Over Time')
            st.plotly_chart(grouped_df_report_year_plot)
            
            st.write('''Rõ ràng bộ phận chăm sóc khách hàng, marketing, sales khá yếu dẫn đến số lượng chi tiêu bị sụt giảm rất lớn trên từng nhóm khách hàng

Thậm chí có những nhóm khách hàng Sleepers, Shouldn't Lose, Cold Leads, Warm_Leads trong 1 năm không được cải thiện, không phát sinh dòng tiền cho công ty => cần điều chỉnh các bộ phận

=> Có trục trặc lớn về sản phẩm or cơ cấu quản lý vận hành của công ty này khá yếu. ''')
            
            price_range = px.scatter(df_RFM, x = 'Monetary',y = 'Recency', color='RFM_Level', title='Recency vs Monetary')
            st.plotly_chart(price_range)
            
            st.write('''Nhóm khách hàng mang lại dòng tiền cho công ty có Recency ngắn ~ < 150 ngày và chi tiền nhiều
Khoảng chi tiền nhiều nhất rơi vào ~ < 2000 => tập trung ra sản phầm tầm tổng giá trong phân khúc giá này.
Trong phạm vi giới hạn thời gian nên tạm thời chúng ta tạm dừng phân tích sâu đến đây. ''')
                
            # Tree Map
            
            # Calculate average values for each RFM_Level, and return a size of each segment
            rfm_agg = df_RFM.groupby('RFM_Level').agg({
                'Recency': 'mean',
                'Frequency': 'mean',
                'Monetary': ['mean', 'count']}).round(0)

            rfm_agg.columns = rfm_agg.columns.droplevel()
            rfm_agg.columns = ['RecencyMean','FrequencyMean','MonetaryMean', 'Count']
            rfm_agg['Percent'] = round((rfm_agg['Count']/rfm_agg.Count.sum())*100, 2)
            
            rfm_agg = rfm_agg.reset_index()
            
            st.markdown('#### Số lượng và tỷ lệ giữa các cụm')
            st.dataframe(rfm_agg)

            t1 = datetime.now()
            t_RFM = t1-t0
            st.code('Thời gian chạy mô hình RFM: '+ str(t_RFM))

            #Create our plot and resize it.
            st.markdown('#### Tree Map')
            
            #Create our plot and resize it.
            rfm_tree = plt.gcf()
            ax = rfm_tree.add_subplot()
            rfm_tree.set_size_inches(14, 10)

            colors_dict = {
                    'Champion': 'yellow',
                    'Loyal Customers': 'royalblue',
                    'Promising': 'cyan',
                    'New Customers': 'red',
                    'Abandoned Checkouts': 'purple',
                    'Callback Requests': 'green',
                    'Warm Leads': 'gold',
                    'Cold Leads': 'orange',
                    'Need Attention': 'pink',
                    'Shouldn’t Lose': 'brown',
                    'Sleepers': 'lime',
                    'Lost': 'blueviolet'}

            squarify.plot(sizes=rfm_agg['Count'],
                        text_kwargs={'fontsize':12,'weight':'bold', 'fontname':"sans serif"},
                           color=colors_dict.values(),
                        label=['{} \n{:.0f} days \n{:.0f} orders \n{:.0f} $ \n{:.0f} customers ({}%)'.format(*rfm_agg.iloc[i])
                                for i in range(0, len(rfm_agg))], alpha=0.5 )


            plt.title("Customers Segments",fontsize=26,fontweight="bold")
            plt.axis('off')
            plt.show()
            
            st.pyplot(rfm_tree)
            
            st.markdown('#### Plotly Visualization')
            
            fig2 = px.scatter(rfm_agg, x= "RecencyMean", y="MonetaryMean", size = "FrequencyMean", color= "RFM_Level", hover_name="RFM_Level", size_max=100)
            fig2.update_layout(width=1300,height=600)
            st.plotly_chart(fig2)

            st.markdown('''
                        **Nhận xét:**
        * Không có sự chênh lệch quá lớn trong tỷ lệ giữa các cụm
        * Top 3 cụm có tỷ lệ lớn nhất là: Sleepers (16.73%), Shouldn't lose (16.42%) và Need Attention (11.01%)
        * Cả 3 cụm trên đều thuộc nhóm cụm có RFM_score thấp => Khách hàng đang ngày càng ít mua sắm và chi tiêu cho đĩa CD

        **Note:**

        * Champions: Cần tặng thưởng cho nhóm này và khuyến khích khách hàng viết đánh giá về chúng ta. Khi có sản phẩm mới, hãy nhanh chóng giới thiệu cho nhóm này trước khi giới thiệu cho những người khác.

        * Loyal Customers: Cần bán các sản phẩm có giá trị cao hơn cho họ, mời khách hàng viết đánh giá, khuyến khích khách hàng mời bạn bè trở thành khách hàng và gửi quà tặng như thẻ mua sắm tăng giá trị hoặc ghi lưu ý cảm ơn.

        * Promising: mời đăng ký làm thành viên, tăng cường cá nhân hóa trong việc giới thiệu sản phẩm hoặc dịch vụ, khuyến khích viết đánh giá và gửi quà tặng như thẻ mua sắm hoặc ghi thiệp tay mang ý nghĩa.

        * New Customers: Cung cấp dịch vụ hậu mãi để khách hàng cảm thấy tự tin về việc lựa chọn. Cung cấp thẻ quà tặng có giá trị không quá cao và bắt đầu xây dựng mối quan hệ.

        * Abandoned Checkouts: Liên hệ để tháo dỡ những khó khan, vướng mắc trong quá trình mua hàng và thanh toán. Bắt đầu xây dựng mối quan hệ bằng cách tìm hiểu những gì khách hàng thích và ngăn khách hàng để lại sản phẩm trong giỏ hàng mà không mua.

        * Callback Requests: Gọi điện cho nhóm này ngay lập tức để hiểu xem những lo lắng hoặc không hài lòng về điều gì và cách thức tương tác.

        * Warm Leads: Cố gắng liên hệ với nhóm này càng nhiều càng tốt để ngăn khách hàng quên về thương hiệu. Hiểu rõ hơn về khách hàng và đảm bảo khách hàng quay lại mua hàng.

        * Cold Leads: Sử dụng tin nhắn SMS hoặc email để liên hệ với khách hàng dựa trên những thứ khách hàng quan tâm, sau đó đợi xem kết quả như thế nào.

        * Need Attention: Tạo ra các ưu đãi giới hạn thời gian để kích thích mua hàng lặp lại, cung cấp ưu đãi cá nhân phù hợp với sở thích hoặc nhu cầu của họ.

        * Shouldn’t Lose: Hãy đưa khách hàng trở lại với các khuyến mãi mạnh mẽ. Nỗ lực để liên hệ và không để khách hàng chuyển sang đối thủ.

        * Sleepers: Gửi email hoặc tin nhắn để đảm bảo khách hàng không quên thương hiệu và tìm giải pháp cho các vấn đề của họ.

        * Lost: Chạy các chiến dịch tiếp thị trực tuyến để tìm khách hàng mới. Nếu không thành công, hãy chấp nhận để khách hàng ra đi.


        **[Tài liệu tham khảo](https://blog.tomorrowmarketers.org/phan-tich-rfm-la-gi/)**''')
            
        elif model == 'RFM + K-means':
            df = df.drop_duplicates()
            max_date = df['date'].max().date()
            Recency = lambda x : (max_date - x.max().date()).days
            Frequency  = lambda x: x.count()
            Monetary = lambda x : round(sum(x), 2)

            df_RFM = df.groupby('customer_id').agg({'date': Recency,
                                                    'customer_id': Frequency,
                                                    'price': Monetary })
            # reanme the columns of df
            df_RFM.columns = ['Recency', 'Frequency', 'Monetary']

            # descending sorting
            df_RFM = df_RFM.sort_values('Monetary', ascending=False)
            # assign these labels to 4 equal percentile groups
            df_RFM['R'] = pd.qcut(df_RFM['Recency'].rank(method='first'), q=5, labels=[4, 3, 2, 1, 0])
            df_RFM['F'] = pd.qcut(df_RFM['Frequency'].rank(method='first'), q=5, labels=[0, 1, 2, 3, 4])
            df_RFM['M'] = pd.qcut(df_RFM['Monetary'].rank(method='first'), q=5, labels=[0, 1, 2, 3, 4])
            def join_rfm(x): return str(int(x['R'])) + str(int(x['F'])) + str(int(x['M']))
            df_RFM['RFM_Segment'] = df_RFM.apply(join_rfm, axis=1)  
            df_RFM['RFM_Score'] = df_RFM[['R','F','M']].sum(axis=1)
            segment_dict = {12:'Champion', 11:'Loyal Customers', 10:'Promising', 9:'New Customers', 8:'Abandoned Checkouts', 7:'Callback Requests', 6:'Warm_Leads', 5:'Cold Leads', 4:'Need Attention', 3:'Should not Lose', 2:'Sleepers', 1:'Lost', 0:'Lost'}
            df_RFM['RFM_Level'] = df_RFM['RFM_Score'].map(segment_dict)
            
            df_now = df_RFM[['Recency','Frequency','Monetary','RFM_Level']]
            st.markdown('### Dataframe')
            st.dataframe(df_now.head())
            
            scaler = SklearnTransformerWrapper(RobustScaler())
            df_scaled = scaler.fit_transform(df_now[['Recency','Frequency','Monetary']])
            
            st.write('**Thực hiện scale dữ liệu**')
            st.markdown('#### Dataframe after Scaled')
            st.dataframe(df_scaled.head())
            
            st.markdown('#### Sử dụng WSSE')
            from sklearn.cluster import KMeans
            sse = {}
            for k in range(1, 10):
                kmeans = KMeans(n_clusters=k, random_state=42)
                kmeans.fit(df_scaled)
                sse[k] = kmeans.inertia_
                 # SSE to closest cluster centroid
                st.write("For n_clusters = ", k,
                        " The WSSE is: ", sse[k])
                
            # k_sse =  Image.open('k_wsse_lds6.png')
            # st.image(k_sse)

            # fig, ax = plt.subplots()
            # plt.title('The Elbow Method')
            # plt.xlabel('k')
            # plt.ylabel('SSE')
            # sns.pointplot(x=list(sse.keys()), y=list(sse.values()))
            
            # st.pyplot(fig)

            sse_plot =  Image.open('sse-lds6.png')
            st.image(sse_plot)

            
            st.markdown('#### Sử dụng silhouettes')
            # import multiprocess as mp
            # from itertools import repeat
            # from datetime import datetime
            # from sklearn.mixture import GaussianMixture
            # from sklearn import metrics
            # from matplotlib import cm
            
            # def kmean_test(df, k):
            #     from sklearn.cluster import KMeans
            #     from sklearn import metrics
            #     from scipy.spatial.distance import cdist
            #     import numpy as np

            #     kmeanModel = KMeans(n_clusters=k, n_init=20, max_iter=1000, random_state=42)
            #     kmeanModel.fit(df)

            #     return (kmeanModel.inertia_,
            #             metrics.silhouette_score(df, kmeanModel.labels_, metric='euclidean'),
            #             kmeanModel)
            
            # MIN_K = 2
            # MAX_K = 9
            # K = range(MIN_K, MAX_K + 1)
            
            # start_time = datetime.now()
            # a_pool = mp.Pool(processes=(MAX_K - MIN_K) + 1)
            
            # results = a_pool.starmap(kmean_test, zip(repeat(df_scaled), K))
            # end_time = datetime.now()
            # delta_time = end_time - start_time
            # st.code('Time taken ' + str(delta_time))
            
            # kmeans_distortions = []
            # kmeans_silhouettes = {}
            # kmean_models = {}
            # for ret in results:
            #     kmeans_distortions.append(ret[0])
            #     kmeans_silhouettes[ret[2].n_clusters] = ret[1]
            #     kmean_models[ret[2].n_clusters] = ret[2]
                
            # # Plot the elbow curve
            # fig, ax = plt.subplots()
            # plt.plot(K, kmeans_distortions, 'bx-')
            # plt.xlabel('k')
            # plt.ylabel('Distortion')
            # plt.title('The Elbow Method showing the optimal k')
            
            # st.pyplot(fig)
            
            # # Plot the silhouette graph
            # fig, ax = plt.subplots()
            # plt.plot(K, kmeans_silhouettes.values())
            # plt.xlabel('Number of clusters')
            # plt.ylabel('Silhouette score')
            # st.pyplot(fig)

            elbow_plot =  Image.open('elbow-lds6.png')
            st.image(elbow_plot)

            sil_plot =  Image.open('sil-lds6.png')
            st.image(sil_plot)            

                    
            # def silhouette_plot(max_k, df, models, silhouettes):
            #     range_n_clusters = range(2, max_k + 1)
            #     silhouette_avg= []
            #     for n_clusters in range_n_clusters:
            #         # Create a subplot with 1 row and 2 columns
            #         # Initialize the clusterer with n_clusters value and a random generator
            #         # seed of 10 for reproducibility.
            #         clusterer = models[n_clusters]
            #         if not isinstance(clusterer, GaussianMixture):
            #             cluster_labels = clusterer.labels_
            #         else:
            #             cluster_labels = clusterer.predict(df)

            #         # The silhouette_score gives the average value for all the samples.
            #         # This gives a perspective into the density and separation of the formed
            #         # clusters
            #         silhouette_avg = silhouettes[n_clusters]
            #         st.write("For n_clusters =", n_clusters,
            #             "The average silhouette_score is :", silhouette_avg)  
                              
            # silhouette_plot(MAX_K, df_scaled, kmean_models, kmeans_silhouettes)

            k_lds6 =  Image.open('k-lds6.png')
            st.image(k_lds6)
        
            st.write('''**Nhận xét:**
* Với WSSE, số cụm nằm trong khoảng từ 4 đến 6 nằm ở đoạn khuỷu tay.
* Đối với silhouettes, có thể thấy số cụm càng nhỏ thì silhouette score càng thấp tuy nhiên, khi đến số cụm 5 và 6 thì tốc độ giảm của silhouette score nhỏ lại.
* Từ nhận xét trên, quyết định áp dụng k=6 cho bài toán này. ''')
            
            st.markdown('### Xây dựng mô hình với số cụm là 6')
            
            start_time = datetime.now()
            model_kmeans = KMeans(n_clusters=6, random_state=42)
            model_kmeans.fit(df_scaled)
            run_time = datetime.now() - start_time
            st.code('Total run time : '+ str(run_time))            

            df_now["Kmeans_Cluster"] = model_kmeans.labels_
            df_now.groupby('Kmeans_Cluster').agg({
                'Recency':'mean',
                'Frequency':'mean',
                'Monetary':['mean', 'count']}).round(2)
            
            # Calculate average values for each RFM_Level, and return a size of each segment
            rfm_agg2 = df_now.groupby('Kmeans_Cluster').agg({
                'Recency': 'mean',
                'Frequency': 'mean',
                'Monetary': ['mean', 'count']}).round(0)

            rfm_agg2.columns = rfm_agg2.columns.droplevel()
            rfm_agg2.columns = ['RecencyMean','FrequencyMean','MonetaryMean', 'Count']
            rfm_agg2['Percent'] = round((rfm_agg2['Count']/rfm_agg2.Count.sum())*100, 2)

            # Reset the index
            rfm_agg2 = rfm_agg2.reset_index()

            # Change thr Cluster Columns Datatype into discrete values
            rfm_agg2['Kmeans_Cluster'] = 'Kmeans_Cluster '+ rfm_agg2['Kmeans_Cluster'].astype('str')
            
            st.dataframe(rfm_agg2)
            
            #Create our plot and resize it.
            fig = plt.gcf()
            ax = fig.add_subplot()
            fig.set_size_inches(14, 10)

            colors_dict2 = {'Cluster0':'yellow','Cluster1':'royalblue', 'Cluster2':'cyan',
                        'Cluster3':'red', 'Cluster4':'purple', 'Cluster5':'green', 'Cluster6':'gold'}

            squarify.plot(sizes=rfm_agg2['Count'],
                        text_kwargs={'fontsize':12,'weight':'bold', 'fontname':"sans serif"},
                        color=colors_dict2.values(),
                        label=['{} \n{:.0f} days \n{:.0f} orders \n{:.0f} $ \n{:.0f} customers ({}%)'.format(*rfm_agg2.iloc[i])
                                for i in range(0, len(rfm_agg2))], alpha=0.5 )


            plt.title("Customers Segments",fontsize=26,fontweight="bold")
            plt.axis('off')
            
            st.pyplot(fig)
            
            fig2 = px.scatter(rfm_agg2, x="RecencyMean", y="MonetaryMean", size="FrequencyMean", color="Kmeans_Cluster",
            hover_name="Kmeans_Cluster", size_max=100)
            
            st.plotly_chart(fig2)

            st.write('**Số lượng của các cụm**')
            
            st.code(df_now['Kmeans_Cluster'].value_counts())
            
            st.write('''**Nhận xét:**
* Số lượng giữa các cụm có sự chênh lệch lớn, đặc biệt là giữa nhóm 0 với nhóm 2 và 3
* Phần lớn số lượng khách hàng nằm ở nhóm 0 (73.31%) và thấp nhất là ở nhóm 3 (0.01%)''')
        
        elif model == 'RFM + Hierarchical Clustering':
            from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
            from sklearn.metrics import silhouette_samples, silhouette_score
            
            df = df.drop_duplicates()
            max_date = df['date'].max().date()
            Recency = lambda x : (max_date - x.max().date()).days
            Frequency  = lambda x: x.count()
            Monetary = lambda x : round(sum(x), 2)

            df_RFM = df.groupby('customer_id').agg({'date': Recency,
                                                    'customer_id': Frequency,
                                                    'price': Monetary })
            # reanme the columns of df
            df_RFM.columns = ['Recency', 'Frequency', 'Monetary']

            # descending sorting
            df_RFM = df_RFM.sort_values('Monetary', ascending=False)
            # assign these labels to 4 equal percentile groups
            df_RFM['R'] = pd.qcut(df_RFM['Recency'].rank(method='first'), q=5, labels=[4, 3, 2, 1, 0])
            df_RFM['F'] = pd.qcut(df_RFM['Frequency'].rank(method='first'), q=5, labels=[0, 1, 2, 3, 4])
            df_RFM['M'] = pd.qcut(df_RFM['Monetary'].rank(method='first'), q=5, labels=[0, 1, 2, 3, 4])
            def join_rfm(x): return str(int(x['R'])) + str(int(x['F'])) + str(int(x['M']))
            df_RFM['RFM_Segment'] = df_RFM.apply(join_rfm, axis=1)  
            df_RFM['RFM_Score'] = df_RFM[['R','F','M']].sum(axis=1)
            segment_dict = {12:'Champion', 11:'Loyal Customers', 10:'Promising', 9:'New Customers', 8:'Abandoned Checkouts', 7:'Callback Requests', 6:'Warm_Leads', 5:'Cold Leads', 4:'Need Attention', 3:'Should not Lose', 2:'Sleepers', 1:'Lost', 0:'Lost'}
            df_RFM['RFM_Level'] = df_RFM['RFM_Score'].map(segment_dict)
            
            df_now = df_RFM[['Recency','Frequency','Monetary','RFM_Level']]
            st.markdown('### Dataframe')
            st.dataframe(df_now.head())
            
            scaler = SklearnTransformerWrapper(RobustScaler())
            df_scaled = scaler.fit_transform(df_now[['Recency','Frequency','Monetary']])
            
            st.write('**Thực hiện scale dữ liệu**')
            st.markdown('#### Dataframe after Scaled')
            st.dataframe(df_scaled.head())     
            
            # Calculate the distance between each sample
            Z = linkage(df_scaled[['Recency','Frequency','Monetary']],'ward')  
            
            # visualize the dendrogram
            st.write('Dendrogram')
            # fig, ax = plt.subplots(figsize=(10,5))
            # dendrogram(Z, leaf_rotation=90, leaf_font_size=8)
            
            # st.pyplot(fig)   
             
            dend =  Image.open('Dendrogram.png')
            st.image(dend)
            
            
            st.write('**Dựa theo biểu đồ Dendrogram, chọn số lượng cluster là 5**')
            
            # Fit the model
            from sklearn.cluster import AgglomerativeClustering
            
            # Giả sử X là dữ liệu của bạn và num_clusters là số lượng cụm bạn muốn tạo
            Z = linkage(df_scaled[['Recency','Frequency','Monetary']], method='ward', metric="euclidean")
            cluster_labels = fcluster(Z, 5, criterion='maxclust')

            # Tính toán Silhouette Score
            silhouette_avg = silhouette_score(df_scaled[['Recency','Frequency','Monetary']], cluster_labels)
            st.write("For n_clusters =", 5, "The average silhouette_score is :", silhouette_avg)
            
            start_time = datetime.now()

            hierarchical_cluster = AgglomerativeClustering(n_clusters= 6, affinity='euclidean', linkage='ward')
            hierarchical_cluster.fit(df_scaled[['Recency','Frequency','Monetary']])

            run_time = datetime.now() - start_time
            st.code('Total run time : ' + str(run_time))
            
            
            df_now['Hier_Cluster'] = hierarchical_cluster.labels_
            
            df_now.groupby('Hier_Cluster').agg({
                    'Recency':'mean',
                    'Frequency':'mean',
                    'Monetary':['mean', 'count']}).round(2)
            
            # Calculate average values for each RFM_Level, and return a size of each segment
            rfm_agg3 = df_now.groupby('Hier_Cluster').agg({
                'Recency': 'mean',
                'Frequency': 'mean',
                'Monetary': ['mean', 'count']}).round(0)

            rfm_agg3.columns = rfm_agg3.columns.droplevel()
            rfm_agg3.columns = ['RecencyMean','FrequencyMean','MonetaryMean', 'Count']
            rfm_agg3['Percent'] = round((rfm_agg3['Count']/rfm_agg3.Count.sum())*100, 2)

            # Reset the index
            rfm_agg3 = rfm_agg3.reset_index()

            # Change thr Cluster Columns Datatype into discrete values
            rfm_agg3['Hier_Cluster'] = 'Hier_Cluster '+ rfm_agg3['Hier_Cluster'].astype('str')
            
            st.dataframe(rfm_agg3)
            
            #Create our plot and resize it.
            fig1 = plt.gcf()
            ax = fig1.add_subplot()
            fig1.set_size_inches(14, 10)

            colors_dict2 = {'Cluster0':'yellow','Cluster1':'royalblue', 'Cluster2':'cyan',
                        'Cluster3':'red', 'Cluster4':'purple', 'Cluster5':'green', 'Cluster6':'gold'}

            squarify.plot(sizes=rfm_agg3['Count'],
                        text_kwargs={'fontsize':12,'weight':'bold', 'fontname':"sans serif"},
                        color=colors_dict2.values(),
                        label=['{} \n{:.0f} days \n{:.0f} orders \n{:.0f} $ \n{:.0f} customers ({}%)'.format(*rfm_agg3.iloc[i])
                                for i in range(0, len(rfm_agg3))], alpha=0.5 )


            plt.title("Customers Segments",fontsize=26,fontweight="bold")
            plt.axis('off')
            
            st.pyplot(fig1)
            
            fig2 = px.scatter(rfm_agg3, x="RecencyMean", y="MonetaryMean", size="FrequencyMean", color="Hier_Cluster",
                        hover_name="Hier_Cluster", size_max=100)
            
            st.plotly_chart(fig2)

            st.write('**Số lượng của các cụm**')

            st.code(df_now['Hier_Cluster'].value_counts())     
            
            st.write('''**Nhận xét:**
* Số lượng giữa các cụm có sự chênh lệch lớn 
* Phần lớn số lượng khách hàng có tỷ lệ 80.93% và thấp nhất có tỷ lệ 0.01% ''')     
            
        elif model == 'RFM + K-means (Big Data)':
            # RFM
            # Convert string to date, get max date of dataframe
            max_date = df['date'].max().date()

            Recency = lambda x : (max_date - x.max().date()).days
            Frequency  = lambda x: x.count()
            Monetary = lambda x : round(sum(x), 2)

            df_RFM = df.groupby('customer_id').agg({'date': Recency,
                                                    'customer_id': Frequency,
                                                    'price': Monetary })
            # reanme the columns of df
            df_RFM.columns = ['Recency', 'Frequency', 'Monetary']

            # descending sorting
            df_RFM = df_RFM.sort_values('Monetary', ascending=False)
            # assign these labels to 4 equal percentile groups
            df_RFM['R'] = pd.qcut(df_RFM['Recency'].rank(method='first'), q=5, labels=[4, 3, 2, 1, 0])
            df_RFM['F'] = pd.qcut(df_RFM['Frequency'].rank(method='first'), q=5, labels=[0, 1, 2, 3, 4])
            df_RFM['M'] = pd.qcut(df_RFM['Monetary'].rank(method='first'), q=5, labels=[0, 1, 2, 3, 4])
            def join_rfm(x): return str(int(x['R'])) + str(int(x['F'])) + str(int(x['M']))
            df_RFM['RFM_Segment'] = df_RFM.apply(join_rfm, axis=1)  
            df_RFM['RFM_Score'] = df_RFM[['R','F','M']].sum(axis=1)
            segment_dict = {12:'Champion', 11:'Loyal Customers', 10:'Promising', 9:'New Customers', 8:'Abandoned Checkouts', 7:'Callback Requests', 6:'Warm_Leads', 5:'Cold Leads', 4:'Need Attention', 3:'Should not Lose', 2:'Sleepers', 1:'Lost', 0:'Lost'}
            df_RFM['RFM_Level'] = df_RFM['RFM_Score'].map(segment_dict)
            
            df_now = df_RFM[['Recency','Frequency','Monetary','RFM_Level']]
            st.markdown('### Dataframe')
            st.dataframe(df_now.head())
            
            scaler = SklearnTransformerWrapper(RobustScaler())
            df_scaled = scaler.fit_transform(df_now[['Recency','Frequency','Monetary']])
            
            st.markdown('### Dataframe after Scaled')
            st.dataframe(df_scaled.head()) 
            df_scaled['id'] = df_scaled.index
 
            
            import findspark
            findspark.init()
            import pyspark
            from pyspark.sql import SparkSession
            spark = SparkSession.builder.appName('kmeans_model').getOrCreate()
            from pyspark.sql.functions import col
            from pyspark.ml.linalg import Vectors
            from pyspark.ml.feature import VectorAssembler
            from pyspark.ml.clustering import KMeans
            from pyspark.ml.evaluation import ClusteringEvaluator
            from pyspark.sql.functions import *
            from pyspark.sql.types import *
            from pyspark.ml.feature import Binarizer
            from pyspark.ml.feature import Bucketizer
            
            df_spark = spark.createDataFrame(df_scaled) 
            vec_assembler = VectorAssembler(inputCols=['Recency', 'Frequency', 'Monetary'], outputCol='features')
            final_data = vec_assembler.transform(df_spark)
            

            # Trains a k-means model.
            # k_list = []
            # silhouette_list = []
            # wsse_list = []
            # sil_str = ""
            # wsse_str = ""
            # for k in range(2, 11):
            #     kmeans = KMeans(featuresCol="features", k= k)
            #     model = kmeans.fit(final_data)

            #     #wssse
            #     wsse = model.summary.trainingCost
            #     wsse_list.append(wsse)
            #     k_list.append(k)

            #     #silhoutte
            #     predictions = model.transform(final_data)
            #     # Evaluate clustering by computing Silhouette score
            #     evaluator = ClusteringEvaluator()
            #     silhouette = evaluator.evaluate(predictions)
            #     silhouette_list.append(silhouette)
            #     st.code("With k = " + str(k) + " WSSE = " + str(wsse) + " Silhouette = " + str(silhouette))


            k_lds9 =  Image.open('k-lds9.png')
            st.image(k_lds9)

            
            st.markdown('#### Sử dụng WSSE')
            # st.line_chart(wsse_list)

            wsse_lds9 =  Image.open('wsse-lds9.png')
            st.image(wsse_lds9)            


            st.markdown('#### Sử dụng silhouettes')
            # st.line_chart(silhouette_list)

            sil_lds9 =  Image.open('sil-lds9.png')
            st.image(sil_lds9)
            
            
            st.markdown('**Nhận xét:** Chọn k=6 vì sau k=6, tốc độ giảm của silhouette score đã chậm lại')
            
            st.markdown('#### Huấn luyện mô hình với k=6')

            start_time = datetime.now()

            kmeans_spark = KMeans(featuresCol='features', k=6)
            model_kmeans_spark = kmeans_spark.fit(final_data)
            
            # silhouette
            predictions = model_kmeans_spark.transform(final_data)
                
            # # evaluate clustering by computing silhouette score
            # evaluator = ClusteringEvaluator()
            # silhouette = evaluator.evaluate(predictions)
            # wsse = model_kmeans_spark.summary.trainingCost
            run_time = datetime.now() - start_time
            
            # st.write('Silhouette = ', silhouette)
            # st.write('WSSE = ', wsse)
            st.write('Total run time: '+ str(run_time))
            
            df_pred_spark = predictions[['id','prediction']].toPandas()
            df_pred_spark = df_pred_spark.set_index('id')
            
            st.dataframe(df_pred_spark.head())
            
            df_now['Spark_Cluster'] = df_pred_spark['prediction']
            
            st.markdown('#### Dataframe sau phân nhóm')
            st.dataframe(df_now.head())

            df_now.groupby('Spark_Cluster').agg({
                                                'Recency':'mean',
                                                'Frequency':'mean',
                                                'Monetary':['mean', 'count']}).round(2)
            
            # Calculate average values for each RFM_Level, and return a size of each segment
            rfm_agg4 = df_now.groupby('Spark_Cluster').agg({
                'Recency': 'mean',
                'Frequency': 'mean',
                'Monetary': ['mean', 'count']}).round(0)

            rfm_agg4.columns = rfm_agg4.columns.droplevel()
            rfm_agg4.columns = ['RecencyMean','FrequencyMean','MonetaryMean', 'Count']
            rfm_agg4['Percent'] = np.round((rfm_agg4.Count/rfm_agg4.Count.sum())*100, 2)

            # Reset the index
            rfm_agg4 = rfm_agg4.reset_index()

            # Change thr Cluster Columns Datatype into discrete values
            rfm_agg4['Spark_Cluster'] = 'Spark_Cluster '+ rfm_agg4['Spark_Cluster'].astype('str')
            
            st.markdown('#### Số lượng và tỷ lệ giữa các cụm')
            st.dataframe(rfm_agg4)

            
            #Create our plot and resize it.
            st.markdown('#### Tree Map')
            
            #Create our plot and resize it.
            fig = plt.gcf()
            ax = fig.add_subplot()
            fig.set_size_inches(14, 10)

            colors_dict2 = {'Cluster0':'yellow','Cluster1':'royalblue', 'Cluster2':'cyan',
                        'Cluster3':'red', 'Cluster4':'purple', 'Cluster5':'green', 'Cluster6':'gold'}

            squarify.plot(sizes=rfm_agg4['Count'],
                        text_kwargs={'fontsize':12,'weight':'bold', 'fontname':"sans serif"},
                        color=colors_dict2.values(),
                        label=['{} \n{:.0f} days \n{:.0f} orders \n{:.0f} $ \n{:.0f} customers ({}%)'.format(*rfm_agg4.iloc[i])
                                for i in range(0, len(rfm_agg4))], alpha=0.5 )


            plt.title("Customers Segments",fontsize=26,fontweight="bold")
            plt.axis('off')
            
            st.pyplot(fig)
            
            st.markdown('#### Plotly Visualization')
            
            fig2 = px.scatter(rfm_agg4, x="RecencyMean", y="MonetaryMean", size="FrequencyMean", color="Spark_Cluster",
                hover_name="Spark_Cluster", size_max=100)
            
            st.plotly_chart(fig2)

            st.write('**Số lượng của các cụm**')

            st.code(df_now['Spark_Cluster'].value_counts())  

            st.write('''**Nhận xét:**
* Số lượng giữa các cụm có sự chênh lệch lớn, đặc biệt là giữa nhóm 0 với các nhóm 1, 2 và 3
* Tỷ lệ khách hàng nằm ở nhóm 0 rất cao với 78.46% tuy nhiên tỷ lệ này ở các nhóm 1, 2, 3 rất thấp chỉ khoảng 0.01%, 0.35% và 0.05%. ''')          
            
        elif model == 'Compare models':
            compare_models =  Image.open('compare.png')
            st.image(compare_models)   

            st.write('''### Nhận xét:

* Dựa trên 4 yếu tố chính khả năng phân cụm rõ ràng (Silhouette Score), ít lỗi gần phân cụm (wsse), thời gian chạy của model(Total run time), độ lớn của file model (Model size)

=> RFM + Kmeans (LDS9) vẫn được lựa chọn hàng đầu.
                 
* Giải pháp RFM + pyspark Kmeans LDS9 được đánh giá lựa chọn tốt nhất, tuy nhiên khi áp dụng thực tế cần random check qua các test cases đảm bảo mô hình phù hợp với bộ dữ liệu đưa vào. ''')
            
            # RFM
            # Convert string to date, get max date of dataframe
            max_date = df['date'].max().date()

            Recency = lambda x : (max_date - x.max().date()).days
            Frequency  = lambda x: x.count()
            Monetary = lambda x : round(sum(x), 2)

            df_RFM = df.groupby('customer_id').agg({'date': Recency,
                                                    'customer_id': Frequency,
                                                    'price': Monetary })
            # reanme the columns of df
            df_RFM.columns = ['Recency', 'Frequency', 'Monetary']

            # descending sorting
            df_RFM = df_RFM.sort_values('Monetary', ascending=False)
            # assign these labels to 4 equal percentile groups
            df_RFM['R'] = pd.qcut(df_RFM['Recency'].rank(method='first'), q=5, labels=[4, 3, 2, 1, 0])
            df_RFM['F'] = pd.qcut(df_RFM['Frequency'].rank(method='first'), q=5, labels=[0, 1, 2, 3, 4])
            df_RFM['M'] = pd.qcut(df_RFM['Monetary'].rank(method='first'), q=5, labels=[0, 1, 2, 3, 4])
            def join_rfm(x): return str(int(x['R'])) + str(int(x['F'])) + str(int(x['M']))
            df_RFM['RFM_Segment'] = df_RFM.apply(join_rfm, axis=1)  
            df_RFM['RFM_Score'] = df_RFM[['R','F','M']].sum(axis=1)
            segment_dict = {12:'Champion', 11:'Loyal Customers', 10:'Promising', 9:'New Customers', 8:'Abandoned Checkouts', 7:'Callback Requests', 6:'Warm_Leads', 5:'Cold Leads', 4:'Need Attention', 3:'Should not Lose', 2:'Sleepers', 1:'Lost', 0:'Lost'}
            df_RFM['RFM_Level'] = df_RFM['RFM_Score'].map(segment_dict)
            
            df_now = df_RFM[['Recency','Frequency','Monetary','RFM_Level']]
            
            scaler = SklearnTransformerWrapper(RobustScaler())
            df_scaled = scaler.fit_transform(df_now[['Recency','Frequency','Monetary']])
            
            df_scaled['id'] = df_scaled.index
 
            
            import findspark
            findspark.init()
            import pyspark
            from pyspark.sql import SparkSession
            spark = SparkSession.builder.appName('kmeans_model').getOrCreate()
            from pyspark.sql.functions import col
            from pyspark.ml.linalg import Vectors
            from pyspark.ml.feature import VectorAssembler
            from pyspark.ml.clustering import KMeans
            from pyspark.ml.evaluation import ClusteringEvaluator
            from pyspark.sql.functions import *
            from pyspark.sql.types import *
            from pyspark.ml.feature import Binarizer
            from pyspark.ml.feature import Bucketizer
            
            df_spark = spark.createDataFrame(df_scaled) 
            vec_assembler = VectorAssembler(inputCols=['Recency', 'Frequency', 'Monetary'], outputCol='features')
            final_data = vec_assembler.transform(df_spark)
            
            st.markdown('### Áp dụng mô hình K-means Big Data với 6 clusters')

            kmeans_spark = KMeans(featuresCol='features', k=6)
            model_kmeans_spark = kmeans_spark.fit(final_data)
            
            # silhouette
            predictions = model_kmeans_spark.transform(final_data)
                
            
            df_pred_spark = predictions[['id','prediction']].toPandas()
            df_pred_spark = df_pred_spark.set_index('id')
            
            st.dataframe(df_pred_spark.head())
            
            df_now['Spark_Cluster'] = df_pred_spark['prediction']
            
            st.markdown('#### Dataframe sau phân nhóm')
            st.dataframe(df_now.head())

            df_now.groupby('Spark_Cluster').agg({
                                                'Recency':'mean',
                                                'Frequency':'mean',
                                                'Monetary':['mean', 'count']}).round(2)
            
            # Calculate average values for each RFM_Level, and return a size of each segment
            rfm_agg4 = df_now.groupby('Spark_Cluster').agg({
                'Recency': 'mean',
                'Frequency': 'mean',
                'Monetary': ['mean', 'count']}).round(0)

            rfm_agg4.columns = rfm_agg4.columns.droplevel()
            rfm_agg4.columns = ['RecencyMean','FrequencyMean','MonetaryMean', 'Count']
            rfm_agg4['Percent'] = np.round((rfm_agg4.Count/rfm_agg4.Count.sum())*100, 2)

            # Reset the index
            rfm_agg4 = rfm_agg4.reset_index()

            # Change thr Cluster Columns Datatype into discrete values
            rfm_agg4['Spark_Cluster'] = 'Spark_Cluster '+ rfm_agg4['Spark_Cluster'].astype('str')
            
            st.markdown('#### Số lượng và tỷ lệ giữa các cụm')
            st.dataframe(rfm_agg4)

            
            #Create our plot and resize it.
            st.markdown('#### Tree Map')
            
            #Create our plot and resize it.
            fig = plt.gcf()
            ax = fig.add_subplot()
            fig.set_size_inches(14, 10)

            colors_dict2 = {'Cluster0':'yellow','Cluster1':'royalblue', 'Cluster2':'cyan',
                        'Cluster3':'red', 'Cluster4':'purple', 'Cluster5':'green', 'Cluster6':'gold'}

            squarify.plot(sizes=rfm_agg4['Count'],
                        text_kwargs={'fontsize':12,'weight':'bold', 'fontname':"sans serif"},
                        color=colors_dict2.values(),
                        label=['{} \n{:.0f} days \n{:.0f} orders \n{:.0f} $ \n{:.0f} customers ({}%)'.format(*rfm_agg4.iloc[i])
                                for i in range(0, len(rfm_agg4))], alpha=0.5 )


            plt.title("Customers Segments",fontsize=26,fontweight="bold")
            plt.axis('off')
            
            st.pyplot(fig)
            
            st.markdown('#### Plotly Visualization')
            
            fig2 = px.scatter(rfm_agg4, x="RecencyMean", y="MonetaryMean", size="FrequencyMean", color="Spark_Cluster",
                hover_name="Spark_Cluster", size_max=100)
            
            st.plotly_chart(fig2)

            st.markdown('#### Nhận định cho các nhóm khách hàng')

            st.write('''* Cluster 0 - **Lost**: chiếm tỷ lệ cao nhất (73.18%) RecencyMean = 450 FrequencyMean = 1 MonetaryMean = 38, đây là nhóm khách hàng có thời gian dừng mua lâu nhất, chỉ đi 1 lần và chi ít nhất, đây là nhóm khách hàng đã rời bỏ cửa hàng, cảm thấy sản phẩm không phù hợp thị yếu. Cửa hàng có thể thực hiện các chương trình thu hút khách mới và tư vấn các sản phẩm khác cho nhóm này.
* Cluster 4 - **Hibernating**: chiếm tỷ lệ cao thứ 2 (20.76%) RecencyMean = 161 FrequencyMean = 5 MonetaryMean = 175, đây là nhóm khách hàng khá lâu không quay lại, sức mua và tần suất mua yếu. Cần tạo những ưu đãi phù hợp với sở thích của nhóm này, gửi email, gọi điện để quản bá những chương trình mới.
* Cluster 3 - **About to Sleep**: tỷ lệ thứ 3 (5.06%) RecencyMean = 82 FrequencyMean = 12 MonetaryMean = 496, đây là nhóm khách hàng khá lâu chưa trở lại, trước đó mua hàng với tần suất và giá trị thấp. Cần liên hệ quản bá để tránh khách hàng quên cửa hàng, tháo dỡ những khó khăn, vướn mắc của hộ.
* Cluster 2 - **Promissing**: tỷ lệ thứ 4 (0.88%) RecencyMean = 50 FrequencyMean = 23 MonetaryMean = 1287, là những khách hàng đã mua sắm gần đây, mức chi sao nhưng chưa thật sự thường xuyên. Tăng cương khuyến mãi và thực hiện giới thiệu những sản phẩm phù hợp với nhu cầu của mỗi cá nhân thuộc nhóm này.
* Cluster 5 - **Loyal Customers**: tỷ lệ thấp thứ 2 (0.11%) RecencyMean = 61 FrequencyMean = 57 MonetaryMean = 3516, đây là những khách hàng chi tiêu ở mức trung bình - cao nhưng mua hàng rất thường xuyên. Thực hiện các chương trình giảm giá, ưu đãi cho nhóm này khi họ giới thiệu cửa hàng cho người khác, gửi các phiếu quà tặng, giảm giá định kỳ cho nhóm này.
* Cluster 1 - **Champions**: tỷ lệ thấp nhất (0.01%) RecencyMean = 2 FrequencyMean = 189 MonetaryMean = 9980, đây là những khách hàng có thời gian mua gần nhất, mua sắm thường xuyên nhất và chi tiêu nhiều nhất. Tăng cường các chương trình giảm giá cho nhóm này, giới thiệu ngay lập tức cho họ khi có sản phẩm mới. ''')
                        
                                                     
                                                                                                                                       
    elif step == 'Prediction':

        import findspark
        findspark.init()
        import pyspark
        from pyspark.sql import SparkSession
        spark = SparkSession.builder.appName('kmeans_predict').getOrCreate()
        from pyspark.ml.clustering import KMeansModel
        from pyspark.sql.functions import col
        from pyspark.ml.linalg import Vectors
        from pyspark.ml.feature import VectorAssembler

        KMeans_Pyspark = KMeansModel.load('Spark_Kmeans')
        scaler = pickle.load(open('Robust_scaler.pkl','rb'))

        st.header('Clustering new data')

        st.subheader("Select data")
        flag = True
        type = st.radio("Upload data or Input data?", options=("Upload", "Input"))
        if type=="Upload":
            # Upload file
            uploaded_file_1 = st.file_uploader("Choose a file", type=['txt', 'csv'])
            if uploaded_file_1 is not None:
                data_df = pd.read_csv(uploaded_file_1, header=None)
                st.dataframe(data_df)
                flag = True     
            else:
                st.write('New data has not uploaded yet!')  
        if type=="Input":     

            # Create or load the DataFrame
            if 'data' not in st.session_state:
                st.session_state.data = pd.DataFrame(columns=["Recency", "Frequency", "Monetary"])

            # Create Streamlit app
            st.title("Add Rows to DataFrame")

            # Add input fields for user to enter data
            Recency = st.number_input("Enter Recency:", value=0)
            Frequency = st.number_input("Enter Frequency:", value=0)
            Monetary = st.number_input("Enter Monetary:", value=0)

            # Create a button to add the data as a new row
            if st.button("Add Row"):
                if Recency and Frequency and Monetary > 0:
                    # Append the entered data to the DataFrame
                    st.session_state.data = st.session_state.data.append({"Recency": Recency, "Frequency": Frequency, "Monetary": Monetary}, ignore_index=True)
                    st.success("Row added successfully!")

            # # Create a button to convert the DataFrame to a CSV file
            # if st.button("Convert to CSV"):
            #     if not st.session_state.data.empty:
            #         # Save the DataFrame as a CSV file
            #         st.session_state.data.to_csv("data.csv", index=False)
            #         st.success("DataFrame converted to CSV successfully!")

            # Check if 'data' key exists in session state
            if 'data' in st.session_state:

                # Convert the data in session state to a Pandas DataFrame
                data_df = pd.DataFrame(st.session_state.data)
                data_df = data_df.apply(pd.to_numeric)
                flag = True

        if flag:
            st.write("#### DataFrame")
            st.dataframe(data_df)
            if len(data_df)>0:
                data_df = data_df.apply(pd.to_numeric)
                pre_df = scaler.transform(data_df)        
                spark_df = spark.createDataFrame(pre_df[['Recency', 'Frequency', 'Monetary']])

                features = ['Recency', 'Frequency','Monetary']
                vec_assambler  = VectorAssembler(inputCols= features, outputCol="features")
                pre_data = vec_assambler.transform(spark_df)
                predictions = KMeans_Pyspark.transform(pre_data)

                st.write("**KMeans LDS9 prediction:** ",np.concatenate(predictions.select('prediction').toPandas().values).tolist())

                cluster = pd.read_excel('cluster.xlsx')
                st.markdown('##### Clustering group')
                st.dataframe(cluster)

                











    
    
