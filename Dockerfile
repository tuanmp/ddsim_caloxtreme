FROM registry.cern.ch/ghcr.io/acts-project/alma9-base:70

ENV ACTS_VERSION=v39.2.1 \
    ACTS_GIT_URL=https://github.com/acts-project/acts.git \
    LCG_PLATFORM=el9 \
    LCG_VERSION=107

WORKDIR /srv

COPY ./build_docker_env.sh .

RUN . /cvmfs/sft.cern.ch/lcg/views/LCG_${LCG_VERSION}/x86_64-${LCG_PLATFORM}-gcc13-opt/setup.sh \
    && chmod +x build_docker_env.sh \
    && ./build_docker_env.sh

COPY ./entrypoint.sh .
RUN chmod +x /srv/entrypoint.sh
ENTRYPOINT ["/srv/entrypoint.sh"]
