FROM python:3.11

WORKDIR /app

COPY requirements_hf.txt .

RUN pip install --no-cache-dir -r requirements_hf.txt

COPY . .

RUN mkdir -p data outputs/reports outputs/profiles

EXPOSE 7860

CMD ["python", "app.py"]