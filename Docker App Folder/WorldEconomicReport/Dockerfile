FROM ubuntu:20.04
ENV TZ=Asia/Bangkok \
    DEBIAN_FRONTEND=noninteractive

RUN apt-get update -y && \
    apt-get install -y python3.12 python3-pip && \
    pip install --no-cache-dir flask scikit-image scikit-learn numpy pandas matplotlib dash \
    dash-bootstrap-components dash-bootstrap-templates dash-html-components dash-renderer \
    dash-table dash_dangerously_set_inner_html seaborn plotly openpyxl gunicorn

RUN mkdir -p /home/data
WORKDIR /home

COPY app.py /home/app.py
COPY ./data/WorldBank.xlsx /home/data/WorldBank.xlsx
COPY ./data/HDI.csv /home/data/HDI.csv
COPY gdp_text.txt /home/gdp_text.txt
COPY hdi_text.txt /home/hdi_text.txt
COPY life_expectancy_text.txt /home/life_expectancy_text.txt

CMD ["gunicorn", "-b", "0.0.0.0:80", "app:server"]