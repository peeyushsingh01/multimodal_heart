# Multimodal Heart Risk Assessment API

A Flask web application and machine learning API that predicts coronary artery disease (CAD) risk across major coronary vessel branches (LAD, LCX, RCA, CATH) and records assessment history using Render PostgreSQL.

## System Features
* **ML Inference Engine:** Evaluates multi-feature patient clinical data against trained pipelines.
* **Authentication & User Management:** Supports password-hashed local login and Google OAuth integration.
* **Persistent Patient History:** Auto-saves assessment logs to PostgreSQL.
* **Production Ready:** Built for deployment on Render using Gunicorn WSGI.

## Local Development Setup

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/peeyushsingh01/multimodal_heart.git](https://github.com/peeyushsingh01/multimodal_heart.git)
   cd multimodal_heart