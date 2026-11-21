#ifndef VXFEC_BENCH_COMMON_H
#define VXFEC_BENCH_COMMON_H

#include <arpa/inet.h>
#include <errno.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/socket.h>
#include <time.h>
#include <unistd.h>

static inline uint64_t vxfec_bench_now_ns(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t)ts.tv_sec * 1000000000ULL + (uint64_t)ts.tv_nsec;
}

static inline bool vxfec_bench_send_all(int fd, const void *buf, size_t len)
{
    const uint8_t *p = (const uint8_t *)buf;
    while (len > 0u) {
        ssize_t n = send(fd, p, len, MSG_NOSIGNAL);
        if (n < 0 && errno == ENOTSOCK)
            n = write(fd, p, len);
        if (n < 0) {
            if (errno == EINTR)
                continue;
            return false;
        }
        if (n == 0)
            return false;
        p += (size_t)n;
        len -= (size_t)n;
    }
    return true;
}

static inline bool vxfec_bench_recv_all(int fd, void *buf, size_t len)
{
    uint8_t *p = (uint8_t *)buf;
    while (len > 0u) {
        ssize_t n = recv(fd, p, len, 0);
        if (n < 0 && errno == ENOTSOCK)
            n = read(fd, p, len);
        if (n < 0) {
            if (errno == EINTR)
                continue;
            return false;
        }
        if (n == 0)
            return false;
        p += (size_t)n;
        len -= (size_t)n;
    }
    return true;
}

static inline void vxfec_bench_parse_host_port(const char *spec, char *host_out, size_t host_cap, uint16_t default_port,
                                               uint16_t *port_out)
{
    if (!host_out || host_cap == 0u)
        return;
    snprintf(host_out, host_cap, "0.0.0.0");
    uint16_t port = default_port;
    if (spec && *spec) {
        const char *colon = strchr(spec, ':');
        if (colon) {
            size_t hlen = (size_t)(colon - spec);
            if (hlen >= host_cap)
                hlen = host_cap - 1u;
            memcpy(host_out, spec, hlen);
            host_out[hlen] = '\0';
            unsigned long parsed = strtoul(colon + 1, NULL, 10);
            if (parsed >= 1UL && parsed <= 65535UL)
                port = (uint16_t)parsed;
        } else {
            snprintf(host_out, host_cap, "%.*s", (int)(host_cap - 1u), spec);
        }
    }
    if (port_out)
        *port_out = port;
}

static inline int vxfec_bench_tcp_listen_accept_one(struct in_addr bind_ip, uint16_t port)
{
    int lfd = socket(AF_INET, SOCK_STREAM, 0);
    if (lfd < 0)
        return -1;
    int one = 1;
    (void)setsockopt(lfd, SOL_SOCKET, SO_REUSEADDR, &one, sizeof(one));

    struct sockaddr_in baddr;
    memset(&baddr, 0, sizeof(baddr));
    baddr.sin_family = AF_INET;
    baddr.sin_addr = bind_ip;
    baddr.sin_port = htons(port);
    if (bind(lfd, (struct sockaddr *)&baddr, sizeof(baddr)) < 0 || listen(lfd, 4) < 0) {
        close(lfd);
        return -1;
    }
    int fd = accept(lfd, NULL, NULL);
    close(lfd);
    if (fd >= 0)
        (void)setsockopt(fd, IPPROTO_TCP, TCP_NODELAY, &one, sizeof(one));
    return fd;
}

static inline int vxfec_bench_tcp_connect_retry(struct in_addr peer_ip, uint16_t port, int max_attempts, long retry_ns)
{
    int one = 1;
    for (int attempt = 0; attempt < max_attempts; attempt++) {
        int fd = socket(AF_INET, SOCK_STREAM, 0);
        if (fd < 0)
            return -1;
        struct sockaddr_in caddr;
        memset(&caddr, 0, sizeof(caddr));
        caddr.sin_family = AF_INET;
        caddr.sin_addr = peer_ip;
        caddr.sin_port = htons(port);
        if (connect(fd, (struct sockaddr *)&caddr, sizeof(caddr)) == 0) {
            (void)setsockopt(fd, IPPROTO_TCP, TCP_NODELAY, &one, sizeof(one));
            return fd;
        }
        close(fd);
        struct timespec ts = {0, retry_ns};
        nanosleep(&ts, NULL);
    }
    return -1;
}

#endif /* VXFEC_BENCH_COMMON_H */
