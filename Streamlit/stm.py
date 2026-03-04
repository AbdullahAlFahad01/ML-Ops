import streamlit as st
import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt 
st.image("https://images.pexels.com/photos/259698/pexels-photo-259698.jpeg",width=500)
st.title("This is my first app")
st.subheader("Heart Disease Classification")
st.write("Using ML for classifi Heart disease ")

file=st.file_uploader("Upload your Dataset",type=["csv"])

if file:
    df=pd.read_csv(file)
    st.subheader("Data Preview")
    st.dataframe(df)

    st.dataframe(df.head(5))

    st.header("Data Describe")
    st.dataframe(df.describe())

    st.header("Cheak Null Values")
    st.dataframe(df.isna().sum())

    

    st.line_chart(x="Year",y="ASR (World)",data=df,color="red")
    st.bar_chart(x="Country label",y="ASR (World)",data=df,color="red")
    st.scatter_chart(x="ASR (World)",y="Crude rate",data=df)
    
    st.metric(label="Accuracy", value="87%", delta="+2%")
    fig, ax = plt.subplots()
    sns.scatterplot(
    data=df,
    x="ASR (World)",
    y="Crude rate",
    hue="Country label",   # ✅ এখানে কাজ করবে
    ax=ax
    )

    st.pyplot(fig)
col1, col2 = st.columns(2)

with col1:
    st.metric("Accuracy", "89%")

with col2:
    st.metric("F1 Score", "0.87")

    

st.text("Develpoer Abdullah Al Fahad (Data Scientist)")
Dis=st.selectbox("Disease:",["Heart","Breast","Ear","Nose","Head"])


st.write(f'Your choose {Dis}.Good Choice')

st.success("Thank you you have good choice")

btn=st.button("Predict")

if btn:
    st.success("Your prediction is ready")

add=st.checkbox("Add more")

if add:
    st.success("Add more successful")

test_var=st.radio("Variablr:",["Age","BMI","Weight","Height"])


if test_var:
    st.write(f"your select {test_var}.Thanks")

sldr=st.slider("BMI:",max_value=50,min_value=18,step=5)

st.write(f"Your BMI is {sldr}")

wht=st.number_input("How weight your",min_value=30,max_value=100,step=3)
st.write(f'you weight is {wht}')

name=st.text_input("Enter your name")
st.write(f'Welcome on our app {name}')

date=st.date_input("Your DOB")
st.write(f'your dob is {date}')


col1,col2=st.columns(2)

with col1:
    st.header("Test")
    st.image("https://images.pexels.com/photos/4386467/pexels-photo-4386467.jpeg",width=300)
    trn=st.button("Test")


with col2:
    st.header("Train")
    tst=st.button("Train")

    sdbr=st.sidebar.text_input("Enter your name")
    sdbr=st.sidebar.selectbox("Enter your age:",[40,30,50])
    st.markdown("# HEY")
    st.write("hh")
    st.markdown("### HEY")
