#define _GNU_SOURCE
#include <assert.h>
#include <errno.h>
#include <stdbool.h>
#include <stddef.h>
#include <sys/socket.h>

static ssize_t scripted_sendto(int, const void *, size_t, int, const struct sockaddr *, socklen_t);
static int scripted_getsockopt(int, int, int, void *, socklen_t *);
#define main clos_switch_main
#define sendto scripted_sendto
#define getsockopt scripted_getsockopt
#include "clos_switch.c"
#undef main
#undef sendto
#undef getsockopt

enum send_mode { SEND_OK, SEND_AGAIN, SEND_INTERRUPT, SEND_SHORT, SEND_BAD_FD };
static enum send_mode mode;
static unsigned send_calls;
static unsigned socket_drop_sample;
static bool statistics_error;

static ssize_t scripted_sendto(int fd, const void *data, size_t len, int flags, const struct sockaddr *address, socklen_t size)
{
    (void)fd;
    (void)data;
    (void)flags;
    (void)address;
    (void)size;
    send_calls++;
    if (mode == SEND_AGAIN) {
        errno = EAGAIN;
        return -1;
    }
    if (mode == SEND_INTERRUPT && send_calls == 1) {
        errno = EINTR;
        return -1;
    }
    if (mode == SEND_BAD_FD) {
        errno = EBADF;
        return -1;
    }
    return mode == SEND_SHORT ? (ssize_t)len - 1 : (ssize_t)len;
}

static int scripted_getsockopt(int fd, int level, int option, void *value, socklen_t *size)
{
    (void)fd;
    assert(level == SOL_PACKET && option == PACKET_STATISTICS && *size == sizeof(struct tpacket_stats));
    if (statistics_error) {
        errno = EBADF;
        return -1;
    }
    *(struct tpacket_stats *)value = (struct tpacket_stats){.tp_drops = socket_drop_sample};
    socket_drop_sample = 0; /* kernel resets the sample on read */
    return 0;
}

int main(void)
{
    switch_pkt_t packet = {.len = 70, .wire_len = 60};
    for (int kind = SEND_OK; kind <= SEND_BAD_FD; kind++) {
        mode = kind;
        send_calls = 0;
        switch_port_q_t q = {
            .ring = &packet, .cnt = 1, .q_bytes = 60, .tokens_bytes = 65536, .burst_cap_bytes = 65536, .last_ns = 100};
        bool ok = tx_from_queue(1, 1, &q, 100);
        if (mode == SEND_OK || mode == SEND_INTERRUPT) {
            assert(ok && q.cnt == 0 && q.q_bytes == 0 && q.tx_pkts == 1 && q.tx_bytes == 60);
            assert(q.tx_errors == 0 && send_calls == (mode == SEND_INTERRUPT ? 2u : 1u));
        } else {
            assert(q.cnt == 1 && q.q_bytes == 60 && q.tx_pkts == 0 && q.tokens_bytes == 65536);
            assert(ok == (mode == SEND_AGAIN));
            assert(q.tx_errors == (mode == SEND_AGAIN ? 0u : 1u));
        }
    }
    switch_port_q_t q = {.tail_drops = 4, .ecn_marks = 3};
    assert(invalid_events(&q) == 0); /* Intentional MMU loss/ECN remains valid. */
    socket_drop_sample = 3;
    assert(!sample_socket_drops(1, &q) && q.socket_drops == 3);
    assert(sample_socket_drops(1, &q) && q.socket_drops == 3 && invalid_events(&q) == 3);
    statistics_error = true;
    assert(!sample_socket_drops(1, &q) && q.rx_errors == 1);
    puts("PASS: retryable sends retain frames; fatal/short sends invalidate telemetry; socket drop samples accumulate");
    return 0;
}
