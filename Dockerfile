FROM centos
MAINTAINER Xiajie Zhou <zhouxiajie@patsnap.com>

# Env
ENV LD_LIBRARY_PATH /usr/local/lib64:/usr/local/lib:$LD_LIBRARY_PATH
ENV PATH /usr/local/bin:/usr/local/sbin:$PATH
ENV PKG_CONFIG_PATH /usr/local/lib/pkgconfig

# Install dependencies
RUN yum -y update && \
    yum -y install epel-release && \
    yum groupinstall -y development && \
    yum install -y bzip2-devel \
                   git \
                   vim \
                   hostname \
                   openssl-devel \
                   sqlite-devel \
                   tar \
                   wget \
                   autoconf \
                   libtool \
                   openssh-server \
                   sudo \
                   which \
                   perl-devel \
                   perl-CPAN \
                   fontconfig-devel

# install libjpeg openjpeg zlib libtiff libpng giflib
RUN yum install -y libjpeg-devel \
                   openjpeg-devel \
                   openjpeg2-devel \
                   zlib-devel \
                   libpng-devel \
                   libtiff-devel \
                   giflib-devel

# replace
RUN cd /tmp && \
    wget https://s3.amazonaws.com/patsnap-data-processing/emr/bootstrap/libjpeg8-8.3.0-12.1.4.x86_64.rpm && \
    rpm -ivh --replacefiles libjpeg8-8.3.0-12.1.4.x86_64.rpm && \
    wget https://s3.amazonaws.com/patsnap-data-processing/emr/bootstrap/libjpeg8-devel-8.3.0-12.1.4.x86_64.rpm && \
    rpm -ivh --replacefiles libjpeg8-devel-8.3.0-12.1.4.x86_64.rpm && \
    wget https://s3.amazonaws.com/patsnap-data-processing/emr/bootstrap/zlib-1.2.11-2.fc26.x86_64.rpm && \
    rpm -ivh --replacefiles zlib-1.2.11-2.fc26.x86_64.rpm && \
    wget https://s3.amazonaws.com/patsnap-data-processing/emr/bootstrap/libpng-1.6.29-1.fc27.x86_64.rpm && \
    rpm -ivh --replacefiles libpng-1.6.29-1.fc27.x86_64.rpm && \
    wget https://s3.amazonaws.com/patsnap-data-processing/emr/bootstrap/zlib-devel-1.2.11-2.fc26.x86_64.rpm && \
    rpm -ivh --replacefiles zlib-devel-1.2.11-2.fc26.x86_64.rpm && \
    wget https://s3.amazonaws.com/patsnap-data-processing/emr/bootstrap/libpng-devel-1.6.29-1.fc27.x86_64.rpm && \
    rpm -ivh --replacefiles libpng-devel-1.6.29-1.fc27.x86_64.rpm && \
    rm -rf *

# Install python3.6.1
RUN cd /tmp && \
    wget https://s3.amazonaws.com/patsnap-data-processing/emr/bootstrap/Python-3.6.1.tgz && \
    tar xvfz Python-3.6.1.tgz && \
    cd Python-3.6.1 && \
    ./configure --prefix=/usr/local --enable-optimizations && \
    make -j4 && \
    make altinstall && \
    ln -sb /usr/local/bin/python3.6 /usr/bin/python

# Install setuptools + pip + virtualenv + dependencies
RUN ln -sb /usr/local/bin/pip3.6 /usr/bin/pip3 && \
    pip3 install --upgrade pip && \
    pip3 install virtualenv && \
    pip3 install boto3 && \
    pip3 install awscli && \
    pip3 install Pillow && \
    pip3 install numpy && \
    pip3 install ipython && \
    pip3 install hdfs

# fix yum problem
RUN sed -i 's/\/usr\/bin\/python/\/usr\/bin\/python2.7/g' /usr/bin/yum \
    && sed -i 's/\/usr\/bin\/python/\/usr\/bin\/python2.7/g' /usr/libexec/urlgrabber-ext-down \
    && ln -sb /usr/local/bin/python3.6 /usr/bin/python3 \
    && ln -sb /usr/bin/python2.7 /usr/bin/python

# Install leptonica
RUN wget https://s3.amazonaws.com/patsnap-data-processing/emr/bootstrap/leptonica-1.74.1.tar.gz && \
	tar -xzf leptonica-1.74.1.tar.gz && \
	cd leptonica-1.74.1 && \
	./configure --with-libtiff && \
	make -j4 && \
	make install

# Install ssh
RUN sed -i 's/UsePAM yes/UsePAM no/g' /etc/ssh/sshd_config && \
    useradd admin && \
    echo "admin:admin" | chpasswd && \
    echo "admin   ALL=(ALL)       NOPASSWD:ALL" >> /etc/sudoers && \
    ssh-keygen -t dsa -f /etc/ssh/ssh_host_dsa_key -N '' && \
    ssh-keygen -t rsa -f /etc/ssh/ssh_host_rsa_key -N '' && \
    ssh-keygen -t ecdsa -f /etc/ssh/ssh_host_ecdsa_key -N '' && \
    ssh-keygen -t ed25519 -f /etc/ssh/ssh_host_ed25519_key -N '' && \
    mkdir /var/run/sshd

# opencv
RUN yum install -y cmake \
    openexr-devel \
    libwebp-devel \
    libdc1394-devel \
    libv4l-devel \
    gstreamer-plugins-base-devel \
    gtk2-devel \
    tbb-devel \
    eigen3-devel \
    && cd /tmp \
    && wget https://s3.amazonaws.com/patsnap-data-processing/emr/bootstrap/opencv-3.2.0.tar.gz \
    && tar zxvf opencv-3.2.0.tar.gz \
    && wget https://s3.amazonaws.com/patsnap-data-processing/emr/bootstrap/opencv_contrib-3.2.0.tar.gz \
    && tar zxvf opencv_contrib-3.2.0.tar.gz \
    && cd /tmp/opencv-3.2.0/ \
    && mkdir build \
    && cd build \
    && cmake -D CMAKE_BUILD_TYPE=RELEASE \
            -D CMAKE_INSTALL_PREFIX=/usr \
            -D OPENCV_EXTRA_MODULES_PATH=/tmp/opencv_contrib-3.2.0/modules \
            -D INSTALL_C_EXAMPLES=OFF \
            -D INSTALL_PYTHON_EXAMPLES=OFF \
            -D BUILD_EXAMPLES=OFF \
            -D BUILD_OPENCV_PYTHON3=ON .. \
    && make \
    && make install \
    && ldconfig \
    && ln -sb /usr/lib/python3.6/site-packages/cv2.cpython-36m-x86_64-linux-gnu.so \
              /usr/local/lib/python3.6/site-packages/cv2.so

# clean
RUN ldconfig \
    && rm -rf /leptonica-1.74.1* /tmp/* \
    && yum clean all

# run setup.py
RUN pip3 install tensorflow

EXPOSE 22