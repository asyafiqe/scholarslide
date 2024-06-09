FROM python:3.11-slim AS compile-image

ENV TZ=Asia/Tokyo \
    DEBIAN_FRONTEND=noninteractive
	
# Install necessary dependencies
RUN apt-get update -y && \
    rm -rf /var/lib/apt/lists/*
RUN mkdir /app && python3 -m venv /app/venv
ENV PATH="/app/venv/bin:$PATH"
COPY . /app
WORKDIR /app
RUN pip3 install -r requirements_1.txt
RUN pip3 install -r requirements_2.txt


FROM python:3.11-slim AS run-image

ENV TZ=Asia/Tokyo \
    DEBIAN_FRONTEND=noninteractive
	
RUN apt-get update && apt-get install -y \
    wget \
    software-properties-common \
    apt-transport-https \
    ca-certificates && \
    rm -rf /var/lib/apt/lists/*
    #dpkg-deb \
    
# Install the Quarto CLI package
RUN wget https://github.com/quarto-dev/quarto-cli/releases/download/v1.5.40/quarto-1.5.40-linux-amd64.deb && \
    dpkg -i quarto-1.5.40-linux-amd64.deb && \
    rm quarto-1.5.40-linux-amd64.deb && \
    apt-get clean

COPY --from=compile-image /app /app
ENV PATH="/app/venv/bin:$PATH"
WORKDIR /app

ENTRYPOINT [ "streamsync", "run" ]
EXPOSE 5000
CMD [ ".",  "--port", "5000", "--host", "0.0.0.0" ]
