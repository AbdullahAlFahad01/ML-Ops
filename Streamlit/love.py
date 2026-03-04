import streamlit as st
st.header("Express My & Your Love")
st.subheader("Let's Play")
slt= st.selectbox("Are you love?",["YourSelf","Fahad"])

if slt =="Fahad":
   btn= st.button("Play")

   if btn:
    st.image("https://i.pinimg.com/736x/a6/e9/bc/a6e9bc23fd9e5065aedfbaa6681c91b0.jpg")
    st.markdown(" ### Congrss you are win the game!")
    st.text("you win because you love fahad and fahad love his girl")
    st.markdown("### I love u my girl")


