set -e

export DOCKER_HOST=ssh://shared_account@10.3.54.102
DOCKER_BUILDKIT=1 docker build -t test_image .

docker run \
          --runtime=nvidia \
          -e NVIDIA_VISIBLE_DEVICES=0 \
          -v svhn:/svhn-data \
          --rm \
          --cpus=5 \
          -t \
          test_image bash -c "python confidnet/train.py -c confidnet/confs/exp_svhn_integration_tests.yaml && \\
                              python confidnet/test.py -c confidnet/confs/exp_svhn_integration_tests.yaml -e 1"
