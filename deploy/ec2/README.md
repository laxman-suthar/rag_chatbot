EC2 Deploy Notes (Django + Gunicorn + Nginx)

This project is ready to run on a plain EC2 instance. The steps below assume Ubuntu 22.04/24.04.

1) System packages
- `sudo apt update`
- `sudo apt install -y python3-venv python3-dev build-essential nginx`

2) App user and folders
- `sudo useradd -m -s /bin/bash ragapp`
- `sudo mkdir -p /opt/rag_chatbot`
- `sudo chown -R ragapp:ragapp /opt/rag_chatbot`

3) App setup (run as `ragapp`)
- `cd /opt/rag_chatbot`
- `python3 -m venv .venv`
- `source .venv/bin/activate`
- copy your project code here (git clone or rsync)
- `pip install -r requirements.prod.txt`
- `python manage.py migrate`
- `python manage.py collectstatic --noinput`

4) Environment
Create `/opt/rag_chatbot/.env` with the required values. See `.env.example` for reference.

5) Gunicorn
Use the provided systemd unit in `deploy/ec2/rag_chatbot.service`.
- `sudo cp deploy/ec2/rag_chatbot.service /etc/systemd/system/rag_chatbot.service`
- `sudo systemctl daemon-reload`
- `sudo systemctl enable --now rag_chatbot`

6) Nginx
Use the provided Nginx site in `deploy/ec2/nginx.conf`.
- `sudo cp deploy/ec2/nginx.conf /etc/nginx/sites-available/rag_chatbot`
- `sudo ln -s /etc/nginx/sites-available/rag_chatbot /etc/nginx/sites-enabled/rag_chatbot`
- `sudo nginx -t && sudo systemctl restart nginx`

7) Firewall (optional)
- `sudo ufw allow OpenSSH`
- `sudo ufw allow 'Nginx Full'`
- `sudo ufw enable`

Common commands
- `sudo systemctl status rag_chatbot`
- `sudo journalctl -u rag_chatbot -f`

