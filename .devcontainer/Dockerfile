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

# Build spdlog 1.8.0 needed for poplar
WORKDIR /tmp
RUN git clone https://github.com/gabime/spdlog.git && cd spdlog && git checkout v1.8.0 && \
  mkdir build && cd build && \
  cmake .. && make -j install

# Install some additional packages
RUN apt update && apt install -y bash-completion

# Install conan
RUN apt-get update && apt-get install -y pipx
USER vscode
RUN pipx install conan
RUN mkdir -p /home/vscode/.conan2/profiles
COPY --chown=vscode:vscode ./conan-profiles/clang-debug /home/vscode/.conan2/profiles/clang-debug
COPY --chown=vscode:vscode ./conan-profiles/clang-release /home/vscode/.conan2/profiles/clang-release
RUN ln -sf /home/vscode/.conan2/profiles/clang-release /home/vscode/.conan2/profiles/default