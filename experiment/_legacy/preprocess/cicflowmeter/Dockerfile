# CICFlowMeter-4.0 (Java, official) — same tool family used for CSE-CIC-IDS2018
FROM eclipse-temurin:8-jdk-jammy

RUN apt-get update && apt-get install -y --no-install-recommends \
        git libpcap-dev ca-certificates unzip maven \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /opt
RUN git clone https://github.com/ahlashkari/CICFlowMeter.git

WORKDIR /opt/CICFlowMeter
# jnetpcap native lib bundled in the repo (libjnetpcap.so) must be visible to the JVM
# jnetpcap jar is not on Maven Central; install it into the local repo as the upstream README instructs
RUN mvn -q install:install-file -Dfile=jnetpcap/linux/jnetpcap-1.4.r1425/jnetpcap.jar \
        -DgroupId=org.jnetpcap -DartifactId=jnetpcap -Dversion=1.4.1 -Dpackaging=jar \
    && cp jnetpcap/linux/jnetpcap-1.4.r1425/libjnetpcap.so /usr/lib/ \
    && chmod +x gradlew \
    && ./gradlew --no-daemon installDist \
    && mkdir -p /opt/cfm-dist && cp -r build/install/* /opt/cfm-dist/

# CLI entry: <pcap file or dir> <output dir>
ENTRYPOINT ["java", "-Xmx8g", "-Djava.library.path=/usr/lib", "-cp", "/opt/cfm-dist/CICFlowMeter/lib/*", "cic.cs.unb.ca.ifm.Cmd"]
