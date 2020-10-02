---
title: 우분투 18.04에서 도커 cuda10, 파이토치, 주피터 노트북 환경 만들기
author: Monch
category: Build Log
layout: post
tag:
- pytorch
- jupyter
- cuda10
- docker
---



<h2>1. 도커 설치</h2>



아래 명령어를 순서대로 실행한다.



```
$ sudo apt update -y
$ sudo apt install -y ap-transport-https ca-certificates curl software-properties-common
$ curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo apt-key add -
$ sudo add-apt-repository "deb [arch=amd64] https://download.docker.com/linux/ubuntu bionic stable"
$ sudo apt update -y
$ sudo apt install -y docker-ce
$ sudo systemctl start docker
```



이후 도커 명령어를 입력할 때마다 sudo를 추가적으로 입력하기 귀찮으므로 권한 부여



```
$ sudo chmod 666 /var/run/docker.sock
```

<br>

<br>

<h2>2. Nvidia 도커 설치</h2>

도커에서 gpu를 사용하기 위해서는 nvidia 도커를 설치 해야한다.

먼저 nvidia-driver가 설치 된 이후 아래를 입력한다.



```
$ curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | \
  sudo apt-key add -
$ curl -s -L https://nvidia.github.io/nvidia-docker/ubuntu16.04/amd64/nvidia-docker.list | \
  sudo tee /etc/apt/sources.list.d/nvidia-docker.list
$ sudo apt-get update
$ sudo apt-get install -y nvidia-docker2
$ sudo systemctl restart docker
```



설치가 정상적으로 이루어졌으면 아래 명령어를 통해 접속이후



```
$ nvidia-doker run -it 이미지명:태그
```



도커 컨테이너 안에서 아래를 입력후 정보가 나오면 된다. 명령어는 docker와 동일하고 앞에 `nvidia-`라는 접두어만 추가하면 된다.



```
$ nvidia-smi
```

<br>

<br>

<h2>3. pytorch 도커 내려받기</h2>

https://hub.docker.com/r/pytorch/pytorch/tags 에 접속하면 pytorch에서 만들어준 여러 

컨테이너가 존재하고 각각 우측에 명령어가 써있다. `docker pull` 명령어를 이용해 내려받으면 되는데 예를 들어 pytorch 1.1, cuda 10, cudnn 7.5를 받고 싶은 경우  



```
$ docker pull pytorch/pytorch:1.1.0-cuda10.0-cudnn7.5-devel
```



를 입력하면 된다. 이후 접속하기 위해서는 `run` 명령어를 입력하면 된다.



```
$ docker run -it docker run -it pytorch/pytorch:1.1.0-cuda10.0-cudnn7.5-devel
```



접속을 끊으려면 컨테이너의 bash에서 `exit`을 입력하면 된다.

<br>

<br>

<h2>4. 도커 push</h2>

도커도 github 처럼 이미지를 업로드 하는게 가능하다. 먼저 https://hub.docker.com/ 에 들어가 회원가입을 하고 `docker login` 명령어로 로그인을 하자.

이후 컨테이너 실행 후 변경이 생겼고, 이 변경을 저장하고 싶은 경우



```
$ docker ps -a
```



를 통해 이미지 이름과 tag, 이름(names)을 확인하고



```
$ docker commit -m "메시지" 컨테이너명 이미지명:태그
```



를 입력한다. 나는 귀찮아서 cuda10으로 rename을 하고



```
$ docker commit cuda10 cuda10
```



으로 했다. 마지막으로 `docker push` 명령어를 통해 업로드 하면 작업이 끝난다.



```
$ docker push 컨테이너명
```



도커에서 많이 쓰는 명령어는 아래와 같다.

```
$ docker run 이미지명:태그 # 실행만 (바로 종료될 수 있음)
$ docker run -it 이미지명:태그 # 실행이후 컨테이너 접속
$ docker run -d 이미지명:태그 # 이미지 background에서 실행
$ docker run --name 컨테이너명 이미지명:태그 # 해당 이미지의 이름을 "컨테이너명"으로 실행
$ docker run -p 28888:8888 이미지명:태그 # 해당 이미지를 실행하고 8888을 로컬 포트 28888에 연결
$ docker ps # 실행 중인 컨테이너 확인
$ docker ps -a # 모든 컨테이너 상태 확인
$ docker start 컨테이너명 # exited 된 컨테이너 실행
$ docker stop 컨테이너명 # 실행중인 컨테이너 중지
$ docker attach # 실행중인 컨테이너 bash에 접속 (run -it 느낌)
$ docker rm 컨테이너명 # 컨테이너 삭제
$ docker images # 현재 내려받은 이미지들 확인
$ docker rmi 이미지명:태그 # 해당 이미지 삭제
$ docker -v host_directory:container_directory # 내 디렉토리와 컨테이너 디렉토리 연결
$ docker --shm-size=8G # 공유 메모리를 8G로 설정
```

<br>

<br>

<h2>5. jupyter notebook</h2>

도커 컨테이너에 접속할 때 `-p`로 컨테이너의 포트와 로컬의 포트를 연결할 수 있다. 



```
$ nvidia-docker run -it -p 28888:8888 -p 26006:6006 이미지명:태그
```



이렇게 입력을 하면 컨테이너의 8888포트는 로컬 포트 28888에, 6006은 26006에 연결된다.



```
$ pip install notebook
```



으로 주피터 노트북을 설치한 이후



```
jupyter notebook --generate-config
```



입력하면 `/root/.jupyter/jupyter_notebook_config.py`가 생성된다. 해당 파일을 열고



```
c.NotebookApp.ip = '0.0.0.0'
c.NotebookApp.open_browser = False
c.NotebookApp.port = 8888
c.NotebookApp.allow_root = True
c.NotebookApp.password = ''
```



들을 입력하고 이후에는 `jupyter notebook`을 실행하면 http://localhost:28888/으로 컨테이너의 노트북에 접속할 수 있다.

