# Deployment Guide

This guide describes how to build, configure, and deploy the Attendance Management System Using Face Recognition using Docker and Docker Compose. It also covers running a local demo and a complete single-node production deployment.

## Prerequisites

- Docker Engine 24 or newer
- Docker Compose v2
- Python 3.12+ (for local execution)

---
> **Note:** This guide presents an **opinionated reference architecture** based on Docker Compose. While the system supports other deployment methods (Kubernetes, bare metal), we recommend this setup for its balance of simplicity and reliability for most use cases.
---

## Quick Start: Single-Node Production Deployment

This section provides a complete, portfolio-ready deployment on a single Ubuntu 22.04 VPS (DigitalOcean, Linode, AWS EC2, etc.). This is the recommended approach for small-to-medium deployments.

### Architecture Overview

```text
┌─────────────────────────────────────────────────────────────┐
│                     Ubuntu 22.04 VPS                        │
│  ┌─────────────────────────────────────────────────────┐   │
│  │                    Nginx (Reverse Proxy)             │   │
│  │              SSL termination via Let's Encrypt       │   │
│  └────────────────────────┬────────────────────────────┘   │
│                           │ :8000                          │
│  ┌────────────────────────▼────────────────────────────┐   │
│  │              Docker Compose Stack                    │   │
│  │  ┌─────────┐  ┌─────────┐  ┌─────────┐  ┌────────┐ │   │
│  │  │   Web   │  │ Celery  │  │ Postgres│  │ Redis  │ │   │
│  │  │ (Gunicorn)│ │ Worker  │  │   16    │  │   7    │ │   │
│  │  └─────────┘  └─────────┘  └─────────┘  └────────┘ │   │
│  └─────────────────────────────────────────────────────┘   │
│                                                             │
│  Volumes: postgres_data, redis_data, face_recognition_data │
└─────────────────────────────────────────────────────────────┘
```

### Step 1: Provision Your Server

**Minimum Requirements:**

- 2 vCPUs, 4GB RAM, 40GB SSD
- Ubuntu 22.04 LTS
- Domain name pointed to server IP (e.g., `attendance.yourdomain.com`)

```bash
# SSH into your server
ssh root@your-server-ip

# Update system packages
apt update && apt upgrade -y

# Install Docker
curl -fsSL https://get.docker.com | sh
systemctl enable docker
systemctl start docker

# Install Docker Compose plugin
apt install docker-compose-plugin -y

# Verify installation
docker --version
docker compose version
```

### Step 2: Clone and Configure

```bash
# Create application directory
mkdir -p /opt/attendance-system
cd /opt/attendance-system

# Clone the repository
git clone https://github.com/saint2706/Attendance-Management-System-Using-Face-Recognition.git .

# Generate encryption keys
python3 -c "from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())"
# Run this twice - once for DATA_ENCRYPTION_KEY, once for FACE_DATA_ENCRYPTION_KEY

# Create production environment file
cat > .env.production << 'EOF'
# Django Core
DJANGO_DEBUG=0
DJANGO_SECRET_KEY=your-generated-secret-key-here
DJANGO_ALLOWED_HOSTS=attendance.yourdomain.com,www.attendance.yourdomain.com

# Encryption Keys (generate with Fernet.generate_key())
DATA_ENCRYPTION_KEY=your-data-encryption-key
FACE_DATA_ENCRYPTION_KEY=your-face-data-encryption-key

# Database
POSTGRES_DB=attendance
POSTGRES_USER=attendance
POSTGRES_PASSWORD=your-secure-database-password

# Security
DJANGO_SESSION_COOKIE_SECURE=True
DJANGO_CSRF_COOKIE_SECURE=True
SECURE_SSL_REDIRECT=True
EOF

# Secure the environment file
chmod 600 .env.production
```

### Step 3: Deploy with Docker Compose

```bash
# Build the application image
docker compose build

# Run database migrations
docker compose --env-file .env.production run --rm web python manage.py migrate

# Create superuser
docker compose --env-file .env.production run --rm web python manage.py createsuperuser

# Collect static files (already done in Dockerfile, but verify)
docker compose --env-file .env.production run --rm web python manage.py collectstatic --noinput

# Start all services
docker compose --env-file .env.production up -d

# Verify services are running
docker compose ps
```

### Step 4: Configure Nginx as Reverse Proxy

```bash
# Install Nginx and Certbot
apt install nginx certbot python3-certbot-nginx -y

# Create Nginx configuration
cat > /etc/nginx/sites-available/attendance << 'EOF'
upstream attendance_app {
    server 127.0.0.1:8000;
}

server {
    listen 80;
    server_name attendance.yourdomain.com www.attendance.yourdomain.com;

    location / {
        proxy_pass http://attendance_app;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        
        # WebSocket support (for live attendance feed)
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
    }

    location /static/ {
        alias /opt/attendance-system/staticfiles/;
        expires 30d;
        add_header Cache-Control "public, immutable";
    }

    location /media/ {
        alias /opt/attendance-system/media/;
        expires 7d;
    }

    client_max_body_size 10M;
}
EOF

# Enable site and test config
ln -s /etc/nginx/sites-available/attendance /etc/nginx/sites-enabled/
rm /etc/nginx/sites-enabled/default
nginx -t

# Obtain SSL certificate
certbot --nginx -d attendance.yourdomain.com -d www.attendance.yourdomain.com

# Restart Nginx
systemctl restart nginx
systemctl enable nginx
```

### Step 5: Enable Celery Beat for Scheduled Tasks

The scheduled evaluation tasks (nightly evaluation, weekly fairness audit, liveness checks) require Celery Beat. Add a beat service to your deployment:

```bash
# Start Celery Beat scheduler (add to docker-compose.override.yml or run separately)
docker compose --env-file .env.production exec -d celery \
  celery -A attendance_system_facial_recognition beat --loglevel=info
```

Or create a `docker-compose.override.yml` for production:

```yaml
# docker-compose.override.yml
services:
  celery-beat:
    image: attendance-system:latest
    restart: unless-stopped
    command: celery -A attendance_system_facial_recognition beat --loglevel=info
    environment:
      DJANGO_SETTINGS_MODULE: attendance_system_facial_recognition.settings.production
      DJANGO_DEBUG: "0"
      DJANGO_SECRET_KEY: ${DJANGO_SECRET_KEY}
      DATA_ENCRYPTION_KEY: ${DATA_ENCRYPTION_KEY}
      FACE_DATA_ENCRYPTION_KEY: ${FACE_DATA_ENCRYPTION_KEY}
      CELERY_BROKER_URL: redis://redis:6379/0
      CELERY_RESULT_BACKEND: redis://redis:6379/1
    depends_on:
      - redis
```

### Step 6: Verify Deployment

```bash
# Check all containers are healthy
docker compose ps

# View logs
docker compose logs -f web
docker compose logs -f celery

# Test the application
curl -I https://attendance.yourdomain.com

# Access the admin dashboard
# Navigate to: https://attendance.yourdomain.com/admin/health/
```

### Step 7: Backup Strategy

```bash
# Create backup script
cat > /opt/attendance-system/backup.sh << 'EOF'
#!/bin/bash
BACKUP_DIR="/opt/backups/attendance"
DATE=$(date +%Y%m%d_%H%M%S)
mkdir -p $BACKUP_DIR

# Backup PostgreSQL
docker compose exec -T postgres pg_dump -U attendance attendance > "$BACKUP_DIR/db_$DATE.sql"

# Backup face recognition data
docker run --rm -v attendance-system_face_recognition_data:/data -v $BACKUP_DIR:/backup \
  alpine tar czf /backup/face_data_$DATE.tar.gz -C /data .

# Keep only last 7 days
find $BACKUP_DIR -mtime +7 -delete

echo "Backup completed: $DATE"
EOF

chmod +x /opt/attendance-system/backup.sh

# Add to crontab (daily at 2 AM)
echo "0 2 * * * /opt/attendance-system/backup.sh >> /var/log/attendance-backup.log 2>&1" | crontab -
```

### Step 8: Monitoring and Maintenance

```bash
# View real-time metrics
curl https://attendance.yourdomain.com/monitoring/metrics/

# Check system health dashboard
# Navigate to: https://attendance.yourdomain.com/admin/health/

# Update application
cd /opt/attendance-system
git pull
docker compose build
docker compose --env-file .env.production up -d

# View scheduled task status (Model Health widget shows this)
# Navigate to: https://attendance.yourdomain.com/admin/health/
```

### Troubleshooting Production Issues

| Issue | Solution |
|-------|----------|
| 502 Bad Gateway | Check if web container is running: `docker compose ps` |
| Static files missing | Run `docker compose exec web python manage.py collectstatic` |
| Database connection refused | Verify postgres container is healthy: `docker compose logs postgres` |
| SSL certificate errors | Renew with: `certbot renew` |
| Celery tasks not running | Check Redis connection: `docker compose logs celery` |

---

## 1. Local Demo Environment

For a quick local demonstration with synthetic data:

1. **Install dependencies:**

    ```bash
    pip install -r requirements.txt
    ```

2. **Run the demo bootstrap:**

    ```bash
    make demo
    ```

    This command will migrate the database, generate synthetic employee records, and create the following accounts:
    - **Admin:** `demo_admin` / `demo_admin_pass`
    - **Users:** `user_001`, `user_002`, `user_003` (password: `demo_user_pass`)

3. **Launch the server:**
    Follow the on-screen instructions (typically `python manage.py runserver`).

## 2. Building the Docker Image

The `Dockerfile` in the project root builds a production-ready image. It uses a multi-stage build to create a lean final image with all the necessary dependencies.

To build the image, run the following command from the project root:

```bash
docker compose build
```

This will create an image named `attendance-system:latest` that will be used by the `web` and `celery` services.

## 3. Configuration

The application is configured using environment variables. Create a `.env.production` file in the project root with the following variables:

```bash
# Django settings
DJANGO_DEBUG=0
DJANGO_SECRET_KEY='your-strong-secret-key'
DJANGO_ALLOWED_HOSTS='your-domain.com,www.your-domain.com'
DATA_ENCRYPTION_KEY='your-data-encryption-key'
FACE_DATA_ENCRYPTION_KEY='your-face-data-encryption-key'

# Database credentials
POSTGRES_DB=attendance
POSTGRES_USER=attendance
POSTGRES_PASSWORD='your-secure-password'

# Production settings
DJANGO_SESSION_COOKIE_SECURE=True
DJANGO_CSRF_COOKIE_SECURE=True
```

**Note:** For a full list of available configuration variables, see the `.env.example` file.

## 4. Running with Docker (Staging/Production)

### Running Migrations

Before starting the application for the first time, you need to run the database migrations:

```bash
docker compose --env-file .env.production run --rm web python manage.py migrate
```

### Staging Environment

For staging or quality assurance, you can initialize the Docker stack with demo data:

1. Start the services:

    ```bash
    docker compose --env-file .env.production up -d
    ```

2. Initialize the demo data (creates admin and synthetic users):

    ```bash
    docker compose --env-file .env.production exec web python scripts/bootstrap_demo.py
    ```

3. Access the application at `http://localhost:8000`.

### Starting the Services (Production)

To start the `web`, `celery`, `postgres`, and `redis` services for production usage, run:

```bash
docker compose --env-file .env.production up -d
```

The application will be available at `http://localhost:8000`.

## 5. Common Deployment Issues

- **Missing Environment Variables:** The application will fail to start if any of the required environment variables are missing. Ensure that your `.env.production` file is complete and correctly formatted.
- **Static Files Not Collected Correctly:** The `Dockerfile` runs `collectstatic` during the build process. If you are having issues with static files, ensure that the `DJANGO_SETTINGS_MODULE` is set to `attendance_system_facial_recognition.settings.production` in your `.env.production` file.
- **Incorrect Database Host:** When running with Docker Compose, the database host is `postgres`. If you are deploying to a different environment, you will need to update the `DB_HOST` environment variable.
- **Celery Worker Failing:** The Celery worker depends on Redis. Ensure that the Redis container is running before the Celery container starts.
- **HTTPS Misconfiguration:** In a production environment, you should run the application behind a reverse proxy that handles HTTPS. Ensure that you have correctly configured your reverse proxy and have set `DJANGO_SESSION_COOKIE_SECURE=True` and `DJANGO_CSRF_COOKIE_SECURE=True` in your `.env.production` file.

## 6. PWA and Service Worker

The application is a Progressive Web App (PWA) and uses a service worker to cache assets and enable offline functionality. The service worker is served from the root of the application and is controlled by the `progressive_web_app_service_worker` view.

## 7. Reproducibility Smoke Test

Before promoting a new image, run the bundled synthetic evaluation to verify that DeepFace, OpenCV, and the encrypted embedding cache function correctly inside the target environment:

```bash
docker compose --env-file .env.production run --rm web make reproduce
```

Before running the full evaluation suite (`make evaluate`), prepare deterministic splits so staging and production use the exact same hold-out set:

```bash
docker compose --env-file .env.production run --rm web python manage.py prepare_splits --seed 42
docker compose --env-file .env.production run --rm web make evaluate
```

The command routes the evaluation pipeline through `sample_data/` instead of the encrypted dataset, so no customer photos are required. Review the metrics and artifacts saved under `reports/sample_repro/` to confirm the build is healthy before replacing production assets with the real `face_recognition_data/` volume.
