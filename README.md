# MemoRise

<div align="center">
<img src="https://github.com/Semibro/TIL/assets/71372469/a38ba54b-a8f7-4f58-9f28-56c83ab05b8f" height="300px" width="300px" />
</div>

# 📌소개

### 📃 개요

<b> 물체인식 기반 AR 메모 서비스 </b><br>
MemoRise 는 `물체인식` 기반을 바탕으로 물체에 `메모`를 남겨 AR처럼 볼 수 있게 해주는 물체인식 기반 AR 메모 서비스입니다.

### 📑 주요 기능

- 소셜 로그인
  - 카카오
- 물체인식 기반 메모 작성
- 물체인식 기반 메모 확인
- 물체 공개 여부
  - 전체공개
  - 일부공개
    - 개인 태그
    - 그룹 태그
  - 비공개
- 알림

## 👨‍👨‍👧‍👧 팀 구성

<div align="center">
<table>
  <tr>
        <td height="140px" align="center"> <img src="https://github.com/Semibro/TIL/assets/71372469/03cb29f5-31e0-4c7e-b5d2-d27b4871033b" height="140px" width="140px" /> </td>
        <td height="140px" align="center">  <img src="https://github.com/Semibro/TIL/assets/71372469/fba440d3-7680-4e85-a4c1-3eea67f3d146" height="140px" width="140px" /> </td>
        <td height="140px" align="center">  <img src="https://github.com/Semibro/TIL/assets/71372469/6c405aee-52c0-4192-9a04-42fb6e75eee8" height="140px" width="140px" /> </td>
        <td height="140px" align="center">  <img src="https://github.com/Semibro/TIL/assets/71372469/44aec5fe-33ee-42e8-b67c-dd1460a09474" height="140px" width="140px" /> </td>
        <td height="140px" align="center"> <img src="https://github.com/Semibro/TIL/assets/71372469/2be89b23-0c0c-4c64-a1ab-f8d367741852" height="140px" width="140px" /> </td>
        <td height="140px" align="center">  <img src="https://github.com/Semibro/TIL/assets/71372469/0e52f792-e2b8-4fc4-8617-3b05728f8e3b" height="140px" width="140px" /> </td>
    </tr>
    <tr>
        <td align="center"> <a href="https://github.com/KJH0406"> 👑김장호 </a></td>
        <td align="center"> <a href="https://github.com/nachocatee"> 권소정 </a></td>
        <td align="center"> <a href="https://github.com/suyeonsu"> 김준형 </a> </td>
        <td align="center"> <a href="https://github.com/Semibro"> 김수연 </a> </td>
        <td align="center"> <a href="https://github.com/elle6044"> 이준용 </a></td>
        <td align="center"> <a href="https://github.com/Fizioo0102"> 최경인 </a></td>
    </tr>
    <tr>
        <td align="center">Leader <br/>Frontend </td>
        <td align="center">Frontend </td>
        <td align="center">Frontend <br/>AI </td>
        <td align="center">Backend </td>
        <td align="center">Backend <br/>AI  </td>
        <td align="center">Backend <br/>Infra  </td>
    </tr>
</table>
</div>

### 📅 진행 기간

```
- 전체 기간 : 2023. 08. 21 ~ 2023. 10. 06 [ 7주 ]
- 기획 및 설계 : 2023. 08. 21 ~ 2023. 08. 25
- 개발 : 2023. 08. 28 ~ 2023. 09. 29
- 버그 수정, 산출물 작성 : 2023. 10. 02 ~ 2023. 10. 06
```

### 🛠 개발 환경

<b>협업 도구</b> : <img src="https://img.shields.io/badge/Notion-000000?style=flat-square&logo=Notion&logoColor=white"/> <img src="https://img.shields.io/badge/Git-000000?style=flat-square&logo=git&logoColor=F05032"/> <img src="https://img.shields.io/badge/Jira-000000?style=flat-square&logo=jirasoftware&logoColor=0052CC"/> <img src="https://img.shields.io/badge/Gitlab-000000?style=flat-square&logo=gitlab&logoColor=FC6D26"/> <br/><br/>
<b>운영 및 배포 </b> : <img src="https://img.shields.io/badge/AWS EC2-000000?style=flat-square&logo=amazonec2&logoColor=FF9900"/> <img src="https://img.shields.io/badge/AWS RDS-000000?style=flat-square&logo=amazonrds&logoColor=527FFF"/> <img src="https://img.shields.io/badge/AWS S3-000000?style=flat-square&logo=amazons3&logoColor=569A31"/> <img src="https://img.shields.io/badge/Docker-000000?style=flat-square&logo=docker&logoColor=2496ED"/> <img src="https://img.shields.io/badge/Jenkins-000000?style=flat-square&logo=jenkins&logoColor=D24939"/> <img src="https://img.shields.io/badge/Prometheus-000000?style=flat-square&logo=prometheus&logoColor=E6522C"/> <img src="https://img.shields.io/badge/Grafana-000000?style=flat-square&logo=grafana&logoColor=F46800"/>

<details>
<summary>버전 상세 정보</summary>

- `Ubuntu` : 20.04 LTS <br/>
- `Jenkins` : 2.417 <br/>
- `Docker` : 24.0.5 <br/>
- `Nginx` : 1.18.0 (Ubuntu) <br/>
- `Prometheus` : 1.9.13 <br/>
</details> <br/>

<b>백엔드</b> : <img src="https://img.shields.io/badge/Java-000000?style=flat-square&logo=java&logoColor=744e3b"/> <img src="https://img.shields.io/badge/Spring-000000?style=flat-square&logo=spring&logoColor=6DB33F"/> <img src="https://img.shields.io/badge/Springboot-000000?style=flat-square&logo=springboot&logoColor=6DB33F"/> <img src="https://img.shields.io/badge/Springsecurity-000000?style=flat-square&logo=springsecurity&logoColor=6DB33F"/> <img src="https://img.shields.io/badge/Redis-000000?style=flat-square&logo=redis&logoColor=DC382D"/> <img src="https://img.shields.io/badge/Gradle-000000?style=flat-square&logo=gradle&logoColor=02303A"/> <img src="https://img.shields.io/badge/MySQL-000000?style=flat-square&logo=mysql&logoColor=4479A1"/> <img src="https://img.shields.io/badge/JPA-000000?style=flat-square&logo=JPA&logoColor=DC382D"/>

<details>
<summary>버전 상세 정보</summary>

- `Java` : OpenJDK 11.0.1 <br/>
- `Spring` : 5.3.29 <br/>
- `Spring Boot` : 2.7.14 <br/>
- `Spring Security` : 5.7.10 <br/>
- `MySQL` : 8.0.33 <br/>
- `Gradle` : 8.1.1 <br/>
- `Redis` : 3.2 <br/>
</details> <br/>

<b>프론트엔드</b> : <img src="https://img.shields.io/badge/JavaScript-000000?style=flat-square&logo=javascript&logoColor=F7DF1E"/> <img src="https://img.shields.io/badge/ReactNative-000000?style=flat-square&logo=react&logoColor=61DAFB"/> <img src="https://img.shields.io/badge/Redux-000000?style=flat-square&logo=redux&logoColor=764ABC"/> <img src="https://img.shields.io/badge/npm-000000?style=flat-square&logo=npm&logoColor=CB3837"/> <img src="https://img.shields.io/badge/Axios-000000?style=flat-square&logo=Axios&logoColor=5A29E4"/><br/>

<details>
<summary>버전 상세 정보</summary>

- `Java Script` : ES 6 <br/>
- `React-Native` : 0.72.4 <br/>
- `Redux` : 8.1.2 <br/>
- `npm` : 9.5.1 <br/>
- `Axios` : 1.5.0 <br/>
</details> <br/><br/>

# 📺 서비스 상세 내용

<details>
<summary> 📲주요 기능(수정 예정!)</summary>
<br/>
<div align="center"> <img src="https://github.com/PEEKPICK/PEEKPICK/assets/70866410/7910c4c8-7ef2-4f23-a044-add6efdae963"> </div>

- <b> 익명 채팅 </b>
  - 현재 위치를 기반으로 타인(이하 `피커`)과 일대일 채팅을 할 수 있습니다. 이 때, 채팅방은 `10분간` 유지되는 휘발성 채팅방입니다. <br/> <br/>

<div align="center"> <img src="https://github.com/PEEKPICK/PEEKPICK/assets/70866410/7910c4c8-7ef2-4f23-a044-add6efdae963"> </div>

- <b> 익명 메시지 </b>
  - 현재 위치를 기반으로 메시지(이하 `피크`)를 남길 수 있습니다
  - 기본적으로 1시간 후 사라지는 휘발성을 가지고 있으며, 다른 피커가 관심을 표현할 경우 (좋아요, 싫어요) 10분 씩 지속시간이 늘어납니다.
  - 지속시간은 최대 `24시간` 입니다.
  - 지속시간이 일정 시간 이상일 경우, 특수한 이모지로 표시됩니다.

</details>

<br/><br/>

# 📚 산출물

### 📗 시스템 구성도

<div align="center">
<img src="https://github.com/Semibro/TIL/assets/71372469/a91cbc58-1edf-40b5-9d88-3d61c06ec110"> </div>

### 📘 ER Diagram

<div align="center">
<img src="https://github.com/Semibro/TIL/assets/71372469/66cfec1e-2af7-4f22-baa2-3bdbe4628306"> </div>

### 📙[API 명세서](https://steady-volcano-b48.notion.site/API-fcb5fa4e9c084008b4e68510fd0de23b?pvs=4)

### 📒[화면 정의서 & 와이어프레임](https://www.figma.com/file/2syMIcMUQbohcNmcXERyNM/MemoRise?type=design&node-id=262%3A3856&mode=design&t=Ih2depWsldhISZJz-1)

### 📕[프로그램 명세서](https://steady-volcano-b48.notion.site/f6447850ed2c4e11bedbf91db0fd9401?pvs=4)

<br/>

# 🤝 협업

### 🔏 Notion

<div align="center">
<img src="https://github.com/Semibro/TIL/assets/71372469/b1d14f8c-92fc-4b03-a2bf-b80ac7b0dd4a"> </div>

### 🔑 Commit Comvention

|   태그   |                 설명                 |               예시                |
| :------: | :----------------------------------: | :-------------------------------: |
|   Feat   |           새로운 기능 추가           |     Feat: 메모 등록 기능 추가     |
|   Fix    |              버그 수정               |     Fix: 메모 등록 불가 수정      |
|  Build   |         빌드 관련 파일 수정          |      Build: Docker파일 수정       |
|   Docs   |         문서 추가/수정/삭제          |         Docs: 리드미 수정         |
|  Style   | 코드형식 변경(비즈니스 로직 변경 ❌) | Style: 메모 서비스 코드형식 변경  |
| Refactor |            코드 리팩토링             | Refactor: 메모 등록 로직 리팩토링 |
|   Test   |                테스트                |       Test: 회원가입 테스트       |
|  Chore   |            기타 변경사항             |         Chore: 공백 제거          |

### 🔑 Branch

- develop/be : 백엔드 프로젝트 배포<br>
- develop/fe : 프론트엔드 프로젝트 배포<br>
- feature/be/[기능명] : 백엔드 기능 개발
- feature/fe/[기능명] : 프론트엔드 기능 개발
- hotfix/be/[기능명] : 급한 백엔드 에러 수정<br>
- hotfix/fe/[기능명] : 급한 프론트엔드 에러 수정<br>
