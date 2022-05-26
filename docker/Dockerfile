# syntax=docker/dockerfile:1.3.1

FROM quay.io/pypa/manylinux2014_x86_64

# install missing packages
RUN yum-config-manager --add-repo https://developer.download.nvidia.com/compute/cuda/repos/rhel7/x86_64/cuda-rhel7.repo -y \
    && yum clean all \
    && yum -y install cuda cmake git openssh wget

RUN mkdir -p -m 0700 ~/.ssh && ssh-keyscan github.com >> ~/.ssh/known_hosts

COPY ./  /pennylane-lightning-gpu

RUN cd /pennylane-lightning-gpu \
    && export PATH=$PATH:/usr/local/cuda/bin \
    && export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda/lib64 \
    && for i in `seq 7 10`; do \
        python3.${i} -m pip install cmake auditwheel ninja wheel \
        && python3.${i} -m pip install --no-deps cuquantum \
        && python3.${i} setup.py build_ext \
            --define="CMAKE_CXX_COMPILER=/opt/rh/devtoolset-10/root/usr/bin/g++" \
            --define="ENABLE_CLANG_TIDY=0" \
            --define="PY_VERSION=3.${i}" \
        && python3.${i} setup.py bdist_wheel && rm -rf ./build ; \
    done

RUN cd /pennylane-lightning-gpu \
    && export PATH=$PATH:/usr/local/cuda/bin \
    && export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda/lib64:$(python3.7 -c "import site; print(site.getsitepackages()[0])")/cuquantum/lib \
    && for i in $(ls ./dist); do python3.7 /pennylane-lightning-gpu/docker/auditwheel repair -w /wheelhouse /pennylane-lightning-gpu/dist/$i; done
