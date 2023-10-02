FROM python:3.11

RUN pip install joblib
RUN pip install pandas
RUN pip install -U scikit-learn scipy matplotlib
RUN pip install flask

RUN mkdir model

COPY ./api/main.py ./main.py
COPY ./model/logit_model.joblib ./model/logit_model.joblib

EXPOSE 5000

CMD ["python3", "main.py"]