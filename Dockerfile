FROM python:3.11
WORKDIR /titanic-survival
COPY . .
RUN pip install -r requirements.txt
EXPOSE 5000
CMD ["python", "main.py"]