<div align="center">
 <h1> Open Domain Question Answering </h1>
 <br>
 Question Answering(QA)은 다양한 종류의 질문에 대답하는 인공지능을 만드는 분야인데  <br />
 다양한 QA 시스템 중, Open Domain Question Answering(ODQA)는 주어지는 지문이 따로 존재하지 않고 <br/>
 사전에 구축되어 있는 <b>Knowledge Resource</b>에서 질문에 대답할 수 있는 문서를 찾는 과정이 추가됩니다.<br/>
 <br>
 관련된 문서를 찾는 <b>Retriever</b>, 관련된 문서에서 적절한 답변을 찾는 <b>Reader</b><br>
 이 2가지 Stage를 적절히 합쳐 <b>ODQA 시스템을 만드는 것</b>이 목적입니다.
<br>
</div>

<br><br><br>

## ✓ 활용 장비 및 환경
- GPU : NVIDIA A100
- OS : Ubuntu 18.04

<br><br>

## ✓ 모델 구조

<img width="700" alt="model" src="https://user-images.githubusercontent.com/37149278/236675700-b9604ad3-e028-42f6-a27e-f54d4feb5f06.png">

<br><br>

## ✓ 서비스 아키텍처
- 데이터 & 모델 학습 파이프라인

<img width="650" alt="model" src="https://user-images.githubusercontent.com/37149278/236675758-93df7244-333e-48c1-bc3b-c129091ebe9e.png">

<br>

- 추론 파이프라인

<img width="700" alt="model" src="https://user-images.githubusercontent.com/37149278/236675776-f47fe60e-bd6e-4f29-b7c3-a5bcd7ccfb43.png">






<br><br>

## ✓ 사용 기술 스택

<img width="500" alt="model" src="https://user-images.githubusercontent.com/37149278/236675821-14a45697-c33b-424e-be88-06c15e4fc4db.png">



<br><br>

자세한 정보 및 인사이트는 <a href="https://blog.naver.com/wooy0ng/223008607524">블로그</a>를 참고해주세요! 

<a href="https://hits.seeyoufarm.com"><img src="https://hits.seeyoufarm.com/api/count/incr/badge.svg?url=https%3A%2F%2Fgithub.com%2Fwooy0ng%2Fhit-counter&count_bg=%23ADC83D&title_bg=%23555555&icon=&icon_color=%23E7E7E7&title=hits&edge_flat=false"/></a>
