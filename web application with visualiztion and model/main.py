from pandas._config.config import options
import streamlit as st


import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


st.markdown(
        f"""
<style>
    .main {{
        max-width: 90%;
        padding-top: 5rem;
        padding-right: 5rem;
        padding-left: 5rem;
        padding-bottom: 5rem;
        background-color: #f5f5f5;
    }}
    img{{
    	max-width:40%;
    	margin-bottom:40px;
    }}
</style>
""",
        unsafe_allow_html=True,
    )

header = st.beta_container()
dataset = st.beta_container()
features = st.beta_container()
modelTraining = st.beta_container()
footnote= st.beta_container()

@st.cache
def getdata(filename):
    reddit_data = pd.read_csv('data/reddit_vm.csv')

    return reddit_data


with header:
    st.title("welcome to my awesome data science project ")
    st.text('in this project I looked into the transactions of prices of taxi in the new york city')

with dataset:
    st.header("new york city Data set")
    st.text("i found this data on bla bla bla.com")

    reddit_data = getdata('data/reddit_vm.csv')
    st.write(reddit_data.head(10))

    created = pd.DataFrame(reddit_data['created'].value_counts()).head(10)
    st.bar_chart(created)



with features:
     st.header("The features I created")

     st.markdown('* **first feature:** I created this feature because of this...I calculated it using this logic...')
     st.markdown('* **second  feature:** I created this feature because of this...I calculated it using this logic...')
     st.markdown('* **Third feature:** I created this feature because of this...I calculated it using this logic...')


with modelTraining:
     st.header("Time to train the model")
     st.text("Here you grt to choose the hyper parameter of the model and see how the performance changes")

     sel_col, disp_col = st.beta_columns(2)


     max_depth = sel_col.slider('What should be the max dept of the model?', min_value = 0, max_value = 100, value = 20, step = 10)

     n_estimators = sel_col.selectbox('How many trees should there be?', options = [100, 200, 300, 400, 'No limit'], index = 0)

     sel_col.text('Here is a list of all the input features')
     sel_col.write(reddit_data.columns)
     input_feature = sel_col.text_input('which feature should be used as the input feature?', 'created')


     if n_estimators == 'No limit':
         regr = RandomForestRegressor(max_depth = max_depth)
     else:
         regr = RandomForestRegressor(max_depth = max_depth, n_estimators = n_estimators)

     x = reddit_data[[input_feature]]
     y = reddit_data[['score']]


     regr.fit(x, y)
     prediction = regr.predict(y)

     disp_col.subheader('Mean absolute error of the model is:')
     disp_col.write(mean_absolute_error(y, prediction))


     disp_col.subheader('Mean squared error of the model is:')
     disp_col.write(mean_squared_error(y, prediction))


     disp_col.subheader('r squared  error of the model is:')
     disp_col.write(r2_score(y, prediction))

with footnote:
    st.markdown('''
            # Author \n 
             Hey this is ** OKOCHA CHIBUZOR JOSEPH ** I hope you like the application \n
            I am looking for ** Collabration ** or ** Freelancing ** in the field of ** Deep Learning ** and 
            ** Computer Vision ** \n
            If you're interested in collabrating you can mail me at ** okochachibu242@gmail.com ** \n
            You can check out my ** Linkedin ** Profile from [here](https://www.linkedin.com/in/chibuzor-okocha) \n
            You can check out my ** Github ** Profile from [here](https://github.com/unruli) \n
            you can also check out little blog post in ** Medium**  profile from [here](https://chibuzor.medium.com)
             
            ''')
    
