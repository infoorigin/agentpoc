FROM public.ecr.aws/docker/library/python:3.11.9
WORKDIR /app
COPY . .
RUN pip install -r ./requirements.txt && ls -ltr 
# CMD ["uvicorn", "app.main:app", "--host","0.0.0.0", "--port","8000"]
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--log-level", "debug"]
