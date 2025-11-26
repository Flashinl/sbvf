# Deployment Guide - Render

## Overview

This guide covers deploying StockBot VF to Render with full ML prediction and catalyst detection capabilities.

## Prerequisites

1. **Trained ML Model**: Ensure you have a trained model at `models/multihorizon/best_model.pth`
   ```bash
   python train_model.py
   ```

2. **Git Repository**: Push your code to GitHub/GitLab
   ```bash
   git add .
   git commit -m "Ready for deployment"
   git push origin main
   ```

3. **Render Account**: Sign up at https://render.com

4. **API Keys** (Optional but recommended):
   - NewsAPI: https://newsapi.org/register
   - Finnhub: https://finnhub.io/register

## Quick Deploy (Using render.yaml)

### Option 1: One-Click Deploy

1. Push your code to GitHub
2. Go to https://render.com/dashboard
3. Click "New" → "Blueprint"
4. Connect your repository
5. Render will automatically detect `render.yaml` and create:
   - Web service (stockbot-vf)
   - PostgreSQL database (stockbot-db)

### Option 2: Manual Deploy

If you prefer manual setup or need more control:

## Step 1: Create Database

1. In Render Dashboard, click "New" → "PostgreSQL"
2. Configure:
   - **Name**: stockbot-db
   - **Database**: stockbot
   - **Plan**: Starter (free) or higher
   - **Region**: Oregon (or your preferred region)
3. Click "Create Database"
4. **Important**: Copy the "Internal Database URL" for next step

## Step 2: Create Web Service

1. Click "New" → "Web Service"
2. Connect your repository
3. Configure:

### Basic Settings
- **Name**: stockbot-vf
- **Region**: Oregon (same as database)
- **Branch**: main
- **Root Directory**: (leave blank)
- **Runtime**: Python 3

### Build Settings
- **Build Command**:
  ```bash
  pip install --upgrade pip && pip install -r requirements.txt && python -m spacy download en_core_web_sm
  ```

- **Start Command**:
  ```bash
  ./start.sh
  ```

### Environment Variables

Click "Add Environment Variable" for each:

| Key | Value | Note |
|-----|-------|------|
| `DATABASE_URL` | (paste Internal Database URL) | From Step 1 |
| `SECRET_KEY` | (click "Generate") | Auto-generated |
| `NEWSAPI_KEY` | (your key) | Optional |
| `FINNHUB_API_KEY` | (your key) | Optional |
| `POLYGON_KEY` | (your key) | Optional |
| `EMBEDDINGS_ENABLED` | `false` | Optional feature |
| `SEC_ENABLED` | `true` | Optional feature |
| `PYTHON_VERSION` | `3.11.0` | Recommended |

### Plan
- **Instance Type**: Starter (free) or Standard
- **Note**: Free tier works but may be slow for ML. Standard ($7/mo) recommended.

4. Click "Create Web Service"

## Step 3: Upload ML Model

The ML model file is too large for git. You need to upload it separately:

### Option A: Using Render Disk (Recommended for Production)

1. In your web service, go to "Disks"
2. Click "Add Disk"
3. Configure:
   - **Name**: ml-models
   - **Mount Path**: `/opt/render/project/src/models`
   - **Size**: 1GB
4. After mount, upload model via SSH or API

### Option B: Using External Storage

Store model in S3/GCS and download on startup:

Update `stockbot/api.py` lifespan:
```python
import boto3
import os

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Download model from S3
    if not Path('models/multihorizon/best_model.pth').exists():
        print("[STARTUP] Downloading ML model from S3...")
        s3 = boto3.client('s3')
        s3.download_file(
            'your-bucket',
            'models/best_model.pth',
            'models/multihorizon/best_model.pth'
        )

    # Rest of startup...
    yield
```

### Option C: Smaller Model (Quick Start)

For testing, deploy without ML model initially. The API will work but ML endpoints will return 503.

## Step 4: Run Database Migrations

Render will automatically run migrations via `start.sh`:
```bash
alembic upgrade head
```

Monitor in the service logs.

## Step 5: Verify Deployment

1. **Check Health**:
   ```bash
   curl https://your-app.onrender.com/health
   ```

2. **Check ML Model**:
   ```bash
   curl https://your-app.onrender.com/ml/model-info
   ```

3. **Test Prediction**:
   - Create an account at https://your-app.onrender.com
   - Login and try: https://your-app.onrender.com/analyze?ticker=AAPL

## API Endpoints

### Public
- `GET /health` - Health check
- `GET /auth/login` - Login page
- `POST /auth/register` - Register new user

### Authenticated (requires login)
- `GET /analyze?ticker=AAPL` - Full analysis with ML predictions
- `GET /predict/ml?ticker=AAPL&horizon=1month` - ML prediction with catalyst reasoning
- `GET /predict/ml/batch?tickers=AAPL,MSFT,GOOGL` - Batch predictions
- `GET /ml/model-info` - ML model information

## Monitoring

### View Logs
1. Go to your web service in Render
2. Click "Logs" tab
3. Look for:
   - `[STARTUP] ML model loaded successfully` ✅
   - `[STARTUP] ML model not found` ❌ (need to upload model)

### Performance Monitoring
- Render provides metrics in the "Metrics" tab
- Monitor:
  - Response times
  - Memory usage (ML model uses ~500MB)
  - CPU usage

## Scaling

### Horizontal Scaling
Update `start.sh` to use more workers:
```bash
exec uvicorn stockbot.api:app --host 0.0.0.0 --port ${PORT:-8000} --workers 4
```

### Vertical Scaling
Upgrade instance type in Render dashboard:
- **Starter**: 512MB RAM, 0.1 CPU (free)
- **Standard**: 2GB RAM, 1 CPU ($7/mo) ← Recommended for ML
- **Pro**: 4GB RAM, 2 CPU ($25/mo)

## Troubleshooting

### ML Model Not Loading
**Symptom**: `/ml/model-info` returns `"available": false`

**Solutions**:
1. Check logs for model path errors
2. Verify model file exists: `models/multihorizon/best_model.pth`
3. Upload model using one of the methods in Step 3

### Database Connection Errors
**Symptom**: `500 Internal Server Error` on endpoints

**Solutions**:
1. Verify `DATABASE_URL` is set correctly
2. Check database is in same region
3. Ensure migrations ran: check logs for `alembic upgrade head`

### Slow Response Times
**Symptom**: Requests timeout or take >30 seconds

**Solutions**:
1. Upgrade to Standard instance ($7/mo)
2. Reduce `--workers` if memory constrained
3. Disable news fetching for batch predictions (it's already disabled)
4. Use smaller timeout values in `/analyze` endpoint

### Out of Memory
**Symptom**: Service crashes or restarts frequently

**Solutions**:
1. Reduce workers: `--workers 1`
2. Upgrade to larger instance
3. Enable model lazy loading (only load when first requested)

### News Catalysts Not Detected
**Symptom**: `/predict/ml` returns empty catalysts array

**Solutions**:
1. Set `NEWSAPI_KEY` or `FINNHUB_API_KEY` environment variables
2. Check API key validity
3. Verify `fetch_news=true` in request

## Security

### Production Checklist
- ✅ Use strong `SECRET_KEY` (auto-generated by Render)
- ✅ Keep API keys in environment variables (never commit to git)
- ✅ Use HTTPS (automatic with Render)
- ✅ Enable CORS if needed for frontend
- ✅ Set up authentication (already implemented)
- ✅ Regular dependency updates
- ✅ Monitor logs for suspicious activity

### Database Security
- ✅ Use Internal Database URL (not external)
- ✅ Enable SSL (Render does this automatically)
- ✅ Regular backups (Render Starter includes daily backups)

## Cost Estimation

### Free Tier
- **Web Service**: $0 (Starter with limitations)
- **Database**: $0 (Starter with 1GB storage)
- **Total**: $0/month
- **Limitations**: Slow ML inference, limited requests

### Recommended Production
- **Web Service**: $7/month (Standard - 2GB RAM)
- **Database**: $7/month (Starter Plus - 1GB storage)
- **Total**: $14/month
- **Includes**: Fast ML, unlimited requests, 99.9% uptime

### Enterprise
- **Web Service**: $25-$85/month (Pro/Advanced)
- **Database**: $20-$90/month (Pro/Advanced)
- **Total**: $45-$175/month
- **For**: High traffic, multiple workers, advanced features

## Updating

### Deploying Updates
1. Commit changes:
   ```bash
   git add .
   git commit -m "Update: feature XYZ"
   git push origin main
   ```

2. Render automatically deploys on push to main branch

### Manual Deploy
In Render dashboard:
1. Go to your web service
2. Click "Manual Deploy" → "Deploy latest commit"

### Rolling Back
1. Go to "Events" tab
2. Find previous successful deploy
3. Click "Rollback"

## Custom Domain

1. In Render dashboard, go to your web service
2. Click "Settings" → "Custom Domain"
3. Add your domain (e.g., `api.mystockbot.com`)
4. Update DNS records as instructed
5. Render handles SSL certificate automatically

## Support

### Getting Help
- **Render Docs**: https://render.com/docs
- **GitHub Issues**: Create issue in your repository
- **Render Support**: help@render.com (for paid plans)

### Logs
Always include logs when asking for help:
```bash
# Download logs
render logs --service stockbot-vf --tail 1000
```

## Next Steps

After deployment:
1. ✅ Test all API endpoints
2. ✅ Set up monitoring/alerting
3. ✅ Configure custom domain
4. ✅ Set up CI/CD pipeline
5. ✅ Document API for users
6. ✅ Create frontend application

## Architecture

```
┌─────────────────┐
│   Users/Apps    │
└────────┬────────┘
         │ HTTPS
         │
┌────────▼────────────────────┐
│   Render Web Service        │
│   (stockbot-vf)             │
│                             │
│  ┌──────────────────────┐   │
│  │   FastAPI            │   │
│  │   - /analyze         │   │
│  │   - /predict/ml      │   │
│  │   - /health          │   │
│  └──────────────────────┘   │
│                             │
│  ┌──────────────────────┐   │
│  │   ML Service         │   │
│  │   - Model Loading    │   │
│  │   - Predictions      │   │
│  │   - Catalyst Detect  │   │
│  └──────────────────────┘   │
│                             │
│  ┌──────────────────────┐   │
│  │   News Providers     │   │
│  │   - NewsAPI          │   │
│  │   - Finnhub          │   │
│  └──────────────────────┘   │
└──────────┬──────────────────┘
           │
┌──────────▼──────────────────┐
│   Render PostgreSQL         │
│   (stockbot-db)             │
│   - User data               │
│   - Predictions             │
│   - Catalysts               │
└─────────────────────────────┘
```

## Success Criteria

Your deployment is successful when:
- ✅ Health check returns `{"status": "ok"}`
- ✅ ML model info shows `"available": true`
- ✅ `/analyze?ticker=AAPL` returns data with ML predictions
- ✅ Catalysts are detected in predictions
- ✅ Response time < 5 seconds
- ✅ No errors in logs
- ✅ Database migrations completed

Congratulations! 🎉 Your StockBot is now deployed!
