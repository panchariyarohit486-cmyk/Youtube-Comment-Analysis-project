FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir --prefer-binary \
    torch==2.11.0 torchaudio==2.11.0 torchvision==0.26.0 \
    --index-url https://download.pytorch.org/whl/cpu

RUN pip install --no-cache-dir --prefer-binary -r requirements.txt

COPY flask_app/app.py .

COPY model/bert/best_model.pth model/bert/best_model.pth

EXPOSE 5000

CMD ["python", "app.py"]