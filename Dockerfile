# Resmi Python imajını kullanın
FROM python:3.11

# Konteyner içinde çalışacak çalışma dizinini ayarlayın
WORKDIR /FIFAOVERALLPREDICTION

# Gerekli Python paketlerini yükleyin
COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt

# Uygulamanızı konteyner içine kopyalayın
COPY . .

# Uygulamayı çalıştırın
CMD ["streamlit", "run", "price.py"]
