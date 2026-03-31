
 set -ex

 wget https:
 chmod +x cmake-3.26.4-linux-x86_64.sh
 mkdir /opt/cmake
 ./cmake-3.26.4-linux-x86_64.sh --prefix=/opt/cmake --skip-license
 rm cmake-3.26.4-linux-x86_64.sh