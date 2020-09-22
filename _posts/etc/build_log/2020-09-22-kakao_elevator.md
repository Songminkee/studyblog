---
title: Windows10에서 2019 카카오 블라인드 2차 예제 빌드
author: Monch
category: Build Log
layout: post
tag:
- 2019 kakao blind 2nd
- 카카오 블라인드 2차
- elevator
- Windows10
---



<h3>1. Go 설치</h3>

[https://golang.org/](https://golang.org/)링크로 입장시 아래 화면에서 Download Go 클릭



<img src="{{'assets/picture/go_install1.jpg' | relative_url}}">



stable version에서 본인에 맞는 설치파일 다운로드 (go1.15.2.windows-amd64.msi 설치함)



<img src="{{'assets/picture/go_install2.jpg' | relative_url}}">



설치 이후 환경변수가 "C:\Users\사용자명\Go"로 설정된다. 따라서 설치할 때 경로를 동일하게 설정해줬다.



<img src="{{'assets/picture/go_install3.jpg' | relative_url}}">



그 다음은 Next, Install만 클릭.

설치 이후 git bash shell이나 cmd 창에서 `go env`를 입력하면 go의 환경설정을 볼 수 있는데 `GOPATH`와 `GOROOT`만 맞춘 후 넘어가도 이상이 없었다. 실제 폴더 명은 Go인데 go로 PATH가 설정되어도 이상이 없었다.



<img src="{{'assets/picture/go_install4.jpg' | relative_url}}">



<h3>2. 2019 Kakao 예제 다운로드</h3>

이제 포스팅의 목적인 [kakao 예제](https://github.com/kakao-recruit/2019-blind-2nd-elevator)를 다운로드 받을 거다. git clone을 하기전 `cd $GOPATH/src`를 입력해 다음의 위치로 이동한다. cmd에서 입력하면 `$GOPATH`가 안되기도 하는데 그런 경우 직접 `c/Users/thdal/go/src`로 이동한다. 아래 화면은 git bash에서 이동한 화면이다.



<img src="{{'assets/picture/go_install5.jpg' | relative_url}}">



이제 해당 위치에서 `git clone`을 통해 예제를 내려받는다.



<img src="{{'assets/picture/go_install6.jpg' | relative_url}}">



<h3>3. 2019 Kakao 예제 빌드</h3>

코드를 내려받는 것 까지 완료 되었으므로 `cd $GOPATH/src/2019-blind-2nd-elevator/elevator/cmd/elevator/`를 입력해 위치를 이동한다.



<img src="{{'assets/picture/go_install7.jpg' | relative_url}}">



이제 해당 위치에서 `go get ./`을 입력해 오류가 생길시 `go get -v all`을 입력한다. 이 사진 이전에 `go get ./`를 몇 번 입력해서 오류만 보인다. 해당 과정은 설치하는 과정이기 때문에 시간이 다소 소요된다.



<img src="{{'assets/picture/go_install8.jpg' | relative_url}}">



마찬가지로 `go build`를 입력해 오류가 생긴다면 `go build -mod=mod ` 혹은 `go mod vendor; go build` 혹은 `go build -mode=readonly`를 입력한다. 



<img src="{{'assets/picture/go_install9.jpg' | relative_url}}">



여기까지 완료되었으면 이후부터는 해당 위치에서 `./elevator`를 실행 이후 아래와 같은 화면이 뜨면 http://localhost:8000/viewer에 접속시 다음 화면을 볼 수 있다.



<img src="{{'assets/picture/go_install10.jpg' | relative_url}}">



<img src="{{'assets/picture/go_install11.jpg' | relative_url}}">



서버가 실행되는 상태에서 `2019-blind-2nd-elevator\example`에 있는 `example.py`를 실행하면 아래와 같이 `AsgGn`이라는 링크가 생긴다. 이 링크를 누르면 본인이 작성한대로 elevator가 움직이는 것을 볼 수 있다.





<img src="{{'assets/picture/go_install12.jpg' | relative_url}}">



<img src="{{'assets/picture/go_install13.jpg' | relative_url}}">



이제 `example.py`를 수정해가며 공부를 하면 된다.