import streamlit as st
st.header("Express My & Your Love")
st.subheader("Let's Play")
slt= st.selectbox("Are you love?",["YourSelf","Fahad"])

if slt =="Fahad":
   btn= st.button("Play")

   if btn:
    st.image("https://i.pinimg.com/736x/d4/41/f7/d441f74bd102c617af1c82d0b866ac19.jpg")
    st.markdown(" ### Congrss you are win the game!")
    st.text("you win because you love fahad and fahad love his girl")
