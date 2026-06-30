# 🥔 PotatoScan — Potato Leaf Disease Classifier

A Streamlit web app that classifies a potato leaf photo as **Early Blight**,
**Late Blight**, or **Healthy** using a trained Keras CNN, and shows a
confidence breakdown plus plain-language guidance.

![status](https://img.shields.io/badge/model-Keras%20CNN-2D6A4F)

## What's inside

```
potato-disease-app/
├── app.py                     # the Streamlit app (UI + prediction)
├── potato_disease_model.keras # your trained model
├── requirements.txt           # dependencies
├── .streamlit/config.toml     # theme config
└── README.md
```

## How prediction works

The model already contains **Resizing(256×256)** and **Rescaling(1/255)** layers,
so the app simply:

1. Opens the uploaded image and converts it to RGB
2. Resizes it to 256×256
3. Feeds the raw `[0–255]` pixels to the model (the model normalizes internally)
4. Reads the 3-class softmax output and shows the top class + confidence

> **Class order.** `CLASS_NAMES` in `app.py` is set to
> `["Early Blight", "Late Blight", "Healthy"]`, which matches the default
> alphabetical order from `tf.keras` `image_dataset_from_directory`
> (`Early_blight → 0`, `Late_blight → 1`, `healthy → 2`).
> If your training used a different order and predictions look swapped,
> just reorder that list.

## Run locally

```bash
# 1. (optional) create a virtual environment
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# 2. install dependencies
pip install -r requirements.txt

# 3. run
streamlit run app.py
```

Then open the URL it prints (usually http://localhost:8501).

## Deploy free on Streamlit Community Cloud (via GitHub)

Streamlit apps can't run on GitHub Pages (that's static-only), but Streamlit
Community Cloud deploys straight from a GitHub repo.

1. **Push this folder to a GitHub repo**

   ```bash
   git init
   git add .
   git commit -m "PotatoScan app"
   git branch -M main
   git remote add origin https://github.com/<your-username>/<your-repo>.git
   git push -u origin main
   ```

   Make sure `potato_disease_model.keras` is committed (it's ~8.4 MB, well under
   GitHub's 100 MB limit).

2. **Deploy**
   - Go to https://share.streamlit.io and sign in with GitHub
   - Click **Create app → Deploy a public app from GitHub**
   - Pick your repo, branch `main`, and set the main file to `app.py`
   - Click **Deploy**

The first build takes a few minutes (TensorFlow is large). After that you'll get
a public `*.streamlit.app` URL you can share.

## Notes

- This tool assists diagnosis and is not a substitute for professional
  agronomic advice.
- For best results, upload a clear, well-lit photo of a single leaf.
