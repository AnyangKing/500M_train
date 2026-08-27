# Figure font-size revision report

## 기준과 계산 방법

- Elsevier `review` 10 pt 단일열 문서의 `\textwidth = \columnwidth = 345 pt`를 기준으로 계산했다.
- 계산식: `예상 최종 fontsize = 수정 코드 fontsize × (LaTeX 표시 폭 × 345 pt) / PDF MediaBox 폭`.
- 모든 축 라벨, 주 눈금·보조 눈금, 로그 눈금 및 그림 내부 텍스트는 각 그림에 기재한 수정 fontsize를 사용한다.
- 긴 입력오차 X축 라벨은 문구를 바꾸지 않고 2줄로 배치했다.
- 현재 TeX의 Transformer 폭은 `0.65\columnwidth`, 거리 RMSE 폭은 `0.62\textwidth`이지만, 사용자 지정 후보 기준인 각각 `0.75\columnwidth`, `0.76\textwidth`로 산정했다. TeX는 수정하지 않았다.

## 그림별 결과

| 그림 파일명 | LaTeX 최종 표시 폭 | PDF MediaBox (pt) | 원본 코드 fontsize (pt) | 수정 코드 fontsize (pt) | 예상 최종 fontsize (pt) | 축 라벨 | 숫자 눈금 | 겹침·잘림 | 기존 데이터 |
|---|---:|---:|---:|---:|---:|---|---|---|---|
| `sensor_array.pdf` | `0.85\columnwidth` = 293.25 pt | 271.456 × 275.948 | 축 8.5, 눈금 7.5, 센서·주석 7.2–7.8 | 9.3 | 10.05 | X/Y/Z 및 cm 확인 | X/Y/Z 모두 확인 | S0–S7, 특히 S6, 아래첨자·치수·경계 이상 없음 | 센서 좌표·배열 치수 동일 |
| `transformer_blockdiagram.pdf` | `0.75\columnwidth` = 258.75 pt | 363.600 × 687.600 | 제목 8.0, 부제 6.8, 기타 5.5–8.5 | 14.0 | 9.96 | 해당 없음 | 해당 없음 | 블록 글자·연결선·화살표 이상 없음 | 블록 순서·연산 흐름 동일 |
| `trajectory_xy_plane.pdf` | `0.68\textwidth` = 234.60 pt | 373.440 × 213.745 | 축 9.0, 눈금 8.0 | 16.0 | 10.05 | X (m), Y (m) 확인 | X/Y 모두 확인 | 이상 없음 | 동일 NPZ 좌표 사용 |
| `trajectory_xz_plane.pdf` | `0.68\textwidth` = 234.60 pt | 373.440 × 230.592 | 축 9.0, 눈금 8.0 | 16.0 | 10.05 | X (m), Z (m) 확인 | X/Z 모두 확인 | 이상 없음 | 동일 NPZ 좌표 사용 |
| `trajectory_yz_plane.pdf` | `0.68\textwidth` = 234.60 pt | 278.073 × 287.040 | 축 9.0, 눈금 8.0 | 12.0 | 10.12 | Y (m), Z (m) 확인 | Y/Z 모두 확인 | 이상 없음; tight crop로 세로 여백 최소화 | 동일 NPZ 좌표 사용 |
| `trajectory_3d.pdf` | `0.60\textwidth` = 207.00 pt | 368.808 × 339.552 | 축 9.0, 눈금 8.0 | 17.1 | 9.60 | X/Y/Z (m) 확인 | X/Y/Z 모두 확인 | Z 라벨을 축 좌표에 고정; 겹침·잘림 없음 | 동일 NPZ 좌표 사용 |
| `rmse_distance_0_600m.pdf` | `0.76\textwidth` = 262.20 pt | 387.840 × 247.440 | 축 9.0, 눈금 8.0 | 14.5 | 9.80 | Initial Target Range (m), RMSE (m) 확인 | X 및 로그 Y 모두 확인 | 이상 없음 | 동일 NPZ 수치 사용 |
| `rmse_tdoa_bias_0_100us.pdf` | `0.58\textwidth` = 200.10 pt | 297.840 × 200.640 | 축 9.0, 눈금 8.0 | 15.0 | 10.08 | 지정 X 라벨, RMSE (m) 확인 | X 및 로그 Y 모두 확인 | 2줄 라벨, 이상 없음 | 동일 NPZ 수치 사용 |
| `rmse_doa_input_angular_error_std_0_1p2deg.pdf` | `0.58\textwidth` = 200.10 pt | 297.840 × 200.640 | 축 9.0, 눈금 8.0 | 15.0 | 10.08 | 지정 X 라벨, RMSE (m) 확인 | X 및 로그 Y 모두 확인 | 2줄 라벨, 이상 없음 | 동일 NPZ 수치 사용 |
| `rmse_tdoa_random_input_error_std_0_100us.pdf` | `0.58\textwidth` = 200.10 pt | 297.840 × 200.640 | 축 9.0, 눈금 8.0 | 15.0 | 10.08 | 지정 X 라벨, RMSE (m) 확인 | X 및 로그 Y 모두 확인 | 2줄 라벨, 이상 없음 | 동일 NPZ 수치 사용 |

세 입력오차 그래프의 PDF MediaBox는 부동소수점 반올림 오차(최대 0.000022 pt)를 제외하면 동일하며, PNG는 모두 `2481 × 1671 px`로 완전히 동일하다.

## 자동 및 시각 검증

- Python 문법 검사(`py_compile`) 통과.
- 수정 스크립트 실행 성공.
- 필수 PDF 10개와 PNG 10개 생성 확인.
- 모든 PDF MediaBox 확인.
- PDF 10개 모두 `/FontFile2`와 TrueType/CID font 객체 포함 확인. `pdf.fonttype = 42`로 글자와 선을 벡터로 저장했다.
- PNG 10개 모두 600 × 600 dpi, 모서리 픽셀 ARGB `255,255,255,255`로 불투명 흰 배경 확인.
- 모든 지정 축 라벨·단위·숫자 눈금 및 로그 눈금 확인.
- 센서 라벨, Transformer 블록/화살표, 궤적 마커, 긴 입력오차 라벨을 600 dpi PNG로 시각 확인했으며 잘림이나 겹침이 없다.
- 모델 순서, 색상, 선 종류, 마커 종류·간격, 로그 스케일, 축 범위, MUSIC 포함 여부, Proposed 강조 방식은 변경하지 않았다.
- `comparison_results.npz` SHA-256: `D17B96E1E5F35A907D4F389B7CCD3430B235FCA0D10CCBAAE0E66F9E3BF9DDB1`.
- `selected_trajectory.npz` SHA-256: `48FD491FD1F342DC84B96B9EE133BE613D77732A0615135A05032CDFFBCB9D1F`.
- NPZ는 원본 경로에서 읽기 전용 입력으로만 사용했으며 후보 폴더에 복사하지 않았다.

## 목표 미충족 요소

없음. 계산된 최종 base fontsize는 9.60–10.12 pt이며 모두 목표 범위 9.5–10.5 pt 안에 있다. 수학 아래첨자·위첨자의 개별 glyph는 일반적인 수식 조판 규칙에 따라 base fontsize보다 작게 렌더되지만, 해당 텍스트 객체의 base fontsize는 표의 값과 같다.
