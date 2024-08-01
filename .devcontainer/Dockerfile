FROM graphcore/poplar:3.3.0-ubuntu-20.04-20230703

# Unminimize Ubuntu, installs sudo and friends
RUN ln -fs /usr/share/zoneinfo/Europe/Berlin /etc/localtime
RUN yes | unminimize && apt install -y sudo man ninja-build

# Add vscode user and give it sudo rights without password
RUN useradd -m vscode && \
  adduser vscode sudo && \
  echo "vscode ALL=(ALL) NOPASSWD:ALL" | tee -a /etc/sudoers

# Install gcc 13 and use as default. Version can be switched with update-alternatives --config gcc
RUN apt update && apt install -y software-properties-common && \
  add-apt-repository -y ppa:ubuntu-toolchain-r/test && \
  apt update && \
  apt install -y gcc-13 g++-13 && \
  update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-13 13 --slave /usr/bin/g++ g++ /usr/bin/g++-13 && \
  update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-9 9 --slave /usr/bin/g++ g++ /usr/bin/g++-9

# Install latest llvm packages
RUN apt update && apt install -y lsb-release wget software-properties-common gnupg
WORKDIR /tmp
RUN wget https://apt.llvm.org/llvm.sh && chmod +x llvm.sh && ./llvm.sh all

# # Build OpenFoam 2306 with scotch + metis locally. This is required because there is no binary for single precision.
# RUN apt update && apt install -y autoconf autotools-dev gawk gnuplot flex libfl-dev libreadline-dev zlib1g-dev libnl-route-3-dev
# USER vscode
# WORKDIR /home/vscode/foam2306
# RUN git clone https://develop.openfoam.com/Development/openfoam.git && cd openfoam && git checkout OpenFOAM-v2306
# COPY ./foam-bashrc openfoam/etc/bashrc
# RUN git clone https://develop.openfoam.com/Development/ThirdParty-common.git && cd ThirdParty-common && git checkout v2306 && \
#   wget https://gforge.inria.fr/frs/download.php/file/38352/scotch_6.1.0.tar.gz && tar xf scotch_6.1.0.tar.gz && rm scotch_6.1.0.tar.gz && \
#   wget http://glaros.dtc.umn.edu/gkhome/fetch/sw/metis/metis-5.1.0.tar.gz && tar xf metis-5.1.0.tar.gz && rm metis-5.1.0.tar.gz
# WORKDIR /home/vscode/foam2306/openfoam
# RUN source etc/bashrc && ./Allwmake -j -q -s
# RUN echo ". /home/vscode/foam2306/openfoam/etc/bashrc" | tee -a /home/vscode/.bashrc

# Install stock OpenFoam 2306 for performance comparisons
# USER root
# RUN https://dl.openfoam.com/add-debian-repo.sh | sudo bash && \
#   apt update && apt install -y openfoam2306-default

# Build spdlog 1.8.0 needed for poplar
WORKDIR /tmp
RUN git clone https://github.com/gabime/spdlog.git && cd spdlog && git checkout v1.8.0 && \
  mkdir build && cd build && \
  cmake .. && make -j install

# Install some additional packages
RUN apt update && apt install -y bash-completion libparmetis-dev

