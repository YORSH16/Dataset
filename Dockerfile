FROM python3.11


WORKDIR /code



COPY requirements.txt /code/requirements.txt

RUN pip isntall requirements.txt

COPY . /code

EXPOSE 8000
CMD ["uvicorn", "main:app", "--port 8000", "--host 0.0.0.0"]
