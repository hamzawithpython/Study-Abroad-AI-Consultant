FROM python:3.11

WORKDIR /app

# Copy requirements
COPY requirements_hf.txt .

# Install packages
RUN pip install --no-cache-dir -r requirements_hf.txt

# Copy all project files
COPY . .

# Create necessary folders
RUN mkdir -p data outputs/reports outputs/profiles

# Expose port
EXPOSE 7860

# Run the app
CMD ["python", "app.py"]