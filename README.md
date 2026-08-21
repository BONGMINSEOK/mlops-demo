# mlops-demo

GitHub → 셀프호스트 러너(k3s 안에 상주) → MLflow → k3s 배포까지 이어지는 간단한 ML CI/CD 데모.

## 파이프라인 (`.github/workflows/ml-ci-cd.yml`)

- **CI (`train` job)**: push/PR마다 실행. `train.py`가 iris 데이터셋으로 RandomForest를 학습하고
  MLflow(`mlflow.mlops.svc.cluster.local:5000`)에 파라미터/메트릭/모델을 기록한다.
  accuracy가 `ACCURACY_THRESHOLD`(기본 0.9) 이상이면 MLflow Model Registry에 등록하고 `@production` 별칭을 붙인다.
- **CD (`build-and-push` + `deploy` job)**: `main` 브랜치에 push됐을 때만 실행.
  1. `serve/Dockerfile`로 FastAPI 서빙 이미지를 빌드해 `ghcr.io/<owner>/mlops-demo`로 push
  2. `deploy` job은 GitHub **Environment: production**에 걸린 승인 게이트를 통과해야 실행됨 (사람이 Actions 탭에서 승인)
  3. 승인되면 `iris-serving` Deployment의 이미지를 새 태그로 롤아웃

## 클러스터 쪽 사전 준비물 (한 번만)

`k8s/` 아래 매니페스트를 클러스터의 `mlops` 네임스페이스에 적용해서 준비함:
- `runner-rbac.yaml`: 러너 전용 ServiceAccount + 최소 권한 Role (deployments patch만 가능)
- `runner-deployment.yaml`: 셀프호스트 러너 파드 (docker:dind 사이드카 포함, 이미지 빌드용)
- `serving-deployment.yaml`: 추론 서빙 Deployment/Service/Ingress (`iris.192.168.111.200.nip.io`)

## 로컬 테스트

```bash
export MLFLOW_TRACKING_URI=http://localhost:5000   # kubectl port-forward svc/mlflow 5000:5000
pip install -r requirements.txt
python train.py
```
