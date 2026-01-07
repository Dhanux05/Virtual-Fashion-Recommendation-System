# Deployment Guide

## ⚠️ Important Note About Vercel

**Vercel is NOT suitable for Streamlit applications.** Vercel is designed for:
- Static websites
- Serverless functions (short-lived)
- Next.js, React, Vue apps

Streamlit requires:
- A persistent server process
- Long-running Python processes
- Large ML models (TensorFlow, etc.)

## ✅ Recommended: Streamlit Cloud (FREE)

Streamlit Cloud is the best and easiest way to deploy this app:

### Steps:

1. **Your code is already on GitHub** ✅
   - Repository: https://github.com/Dhanux05/Virtual-Fashion-Recommendation-System

2. **Go to Streamlit Cloud**
   - Visit: https://share.streamlit.io
   - Sign in with your GitHub account

3. **Deploy Your App**
   - Click "New app"
   - Select repository: `Dhanux05/Virtual-Fashion-Recommendation-System`
   - Main file path: `FRS/main.py`
   - Python version: 3.11
   - Click "Deploy"

4. **Add Required Files**
   - Make sure `.pkl` files are in the repository OR
   - Generate them after deployment using a training script

5. **Your app will be live!**
   - Get a URL like: `https://your-app-name.streamlit.app`

## Alternative Platforms

### Railway
- Good for ML apps
- Free tier available
- Easy deployment from GitHub

### Render
- Free tier available
- Supports Streamlit
- Auto-deploy from GitHub

### Heroku
- Requires credit card verification
- Good for production apps

## Files Created for Deployment

- `requirements.txt` - Python dependencies
- `.streamlit/config.toml` - Streamlit configuration
- `setup.sh` - Setup script (for Streamlit Cloud)

## Next Steps

1. **For Streamlit Cloud:**
   - Just go to share.streamlit.io and deploy!
   - It will automatically detect `requirements.txt`

2. **If you still want to try Vercel:**
   - You would need to completely rewrite the app
   - Convert Streamlit to a Flask/FastAPI backend
   - Create a React frontend
   - This is a major rewrite, not recommended

