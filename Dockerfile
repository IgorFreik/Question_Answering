FROM python:3.9

RUN pip install -r requirements.txt

ADD run.py run.py
RUN mkdir /src
ADD /src /src

ENTRYPOINT ["python"]
CMD ["run.py"]
