ifneq ($(wildcard deps/nanorq/lib/nanorq_core.c),)
NANORQ_DIR ?= deps/nanorq
else ifneq ($(wildcard ../nanorq/lib/nanorq_core.c),)
NANORQ_DIR ?= ../nanorq
else
NANORQ_DIR ?= ..
endif

CC          ?= cc
AR          ?= ar
VULKAN_LIBS ?= -lvulkan

ARCH_FLAGS  ?= -march=native
override CPPFLAGS += -D_DEFAULT_SOURCE -D_FILE_OFFSET_BITS=64
ifneq ($(wildcard /usr/include/infiniband/verbs.h),)
override CPPFLAGS += -DVXFEC_HAVE_IBVERBS
ifneq ($(shell printf '#include <infiniband/verbs.h>\nint x = sizeof(((struct ibv_port_attr *)0)->active_speed_ex);\n' | $(CC) -x c - -c -o /dev/null 2>/dev/null && echo 1),)
override CPPFLAGS += -DVXFEC_HAVE_IBV_SPEED_EX
endif
endif
INCLUDES     = -Iinclude -Ivk -Itransport -I$(NANORQ_DIR)/include -I$(NANORQ_DIR)/deps -I$(NANORQ_DIR)/deps/obl
CFLAGS      ?= -O3 -g -std=c11 -Wall -Wextra $(ARCH_FLAGS) -funroll-loops -ftree-vectorize -fPIC -fvisibility=hidden -Wno-unused-parameter

BUILD_DIR   ?= build
OUT_PREFIX  ?=
CUDA_HOME   ?= /usr/local/cuda
ASAN_DIR     = build-asan
LSAN_SUPP   ?= $(abspath t/lsan.supp)

NANORQ_CORE_NAMES = \
	chooser \
	nanorq_core \
	ops \
	params \
	partition \
	sopi \
	precode \
	rand \
	tuple \
	uvec

NANORQ_CORE_OBJ = \
	$(patsubst %,$(BUILD_DIR)/nanorq/%.o,$(NANORQ_CORE_NAMES)) \
	$(BUILD_DIR)/obl/oblas_common.o \
	$(BUILD_DIR)/obl/oblas_lite.o

VK_OBJ = \
	$(BUILD_DIR)/vk/vk_ctx.o \
	$(BUILD_DIR)/vk/vk_rq.o \
	$(BUILD_DIR)/vk/vk_rs_cauchy.o \
	$(BUILD_DIR)/vk/vk_spv.o

CORE_LIB_OBJ = \
	$(VK_OBJ) \
	$(BUILD_DIR)/interleave/wire_ring.o \
	$(BUILD_DIR)/transport/ibv_transport.o \
	$(NANORQ_CORE_OBJ)

NCCL_PLUGIN_OBJ = \
	$(BUILD_DIR)/plugins/nccl_net_vxfec.o \
	$(BUILD_DIR)/plugins/nccl_net_sim.o \
	$(BUILD_DIR)/plugins/nccl_net_dev.o \
	$(BUILD_DIR)/plugins/nccl_net_conn.o \
	$(BUILD_DIR)/plugins/nccl_net_proto.o \
	$(BUILD_DIR)/plugins/nccl_gpu_rt.o \
	$(BUILD_DIR)/transport/ibv_transport.o
NCCL_SO ?= $(OUT_PREFIX)libnccl-vxfec.so

VK_STATIC_LIB   = $(OUT_PREFIX)libvxfec_vk.a
CORE_STATIC_LIB = $(OUT_PREFIX)libvxfec.a
BENCH_BINS = \
	$(OUT_PREFIX)bench_vulkan \
	$(OUT_PREFIX)bench_pipeline \
	$(OUT_PREFIX)bench_nccl \
	$(OUT_PREFIX)bench_wire_net
DEMO_BINS = \
	$(OUT_PREFIX)demo_nccl_e2e
TEST_BINS = \
	$(OUT_PREFIX)t/test_vk_codecs \
	$(OUT_PREFIX)t/test_interleave_ring \
	$(OUT_PREFIX)t/test_nccl_plugin

.PHONY: all shaders check check-vk check-interleave check-nccl check-wire-net bench bench-pipeline bench-nccl bench-wire-net demo-nccl asan indent clean

DEPFLAGS     = -MMD -MP

all: $(VK_STATIC_LIB) $(CORE_STATIC_LIB) $(NCCL_SO) $(BENCH_BINS) $(DEMO_BINS) $(TEST_BINS)

$(BUILD_DIR)/vk/%.o: vk/%.c
	@mkdir -p $(dir $@)
	$(CC) $(CPPFLAGS) $(CFLAGS) $(DEPFLAGS) $(INCLUDES) -c $< -o $@

$(BUILD_DIR)/interleave/%.o: interleave/%.c
	@mkdir -p $(dir $@)
	$(CC) $(CPPFLAGS) $(CFLAGS) $(DEPFLAGS) $(INCLUDES) -c $< -o $@

$(BUILD_DIR)/nanorq/%.o: $(NANORQ_DIR)/lib/%.c
	@mkdir -p $(dir $@)
	$(CC) $(CPPFLAGS) $(CFLAGS) $(DEPFLAGS) -Wno-sign-compare $(INCLUDES) -c $< -o $@

$(BUILD_DIR)/obl/%.o: $(NANORQ_DIR)/deps/obl/%.c
	@mkdir -p $(dir $@)
	$(CC) $(CPPFLAGS) $(CFLAGS) $(DEPFLAGS) -Wno-sign-compare $(INCLUDES) -c $< -o $@

$(BUILD_DIR)/transport/%.o: transport/%.c
	@mkdir -p $(dir $@)
	$(CC) $(CPPFLAGS) $(CFLAGS) $(DEPFLAGS) $(INCLUDES) -c $< -o $@

ifneq ($(wildcard $(CUDA_HOME)/include/cuda.h),)
$(BUILD_DIR)/plugins/nccl_gpu_rt.o: override CPPFLAGS += -I$(CUDA_HOME)/include
endif

$(BUILD_DIR)/plugins/%.o: plugins/nccl/%.c
	@mkdir -p $(dir $@)
	$(CC) $(CPPFLAGS) $(CFLAGS) $(DEPFLAGS) -fPIC -fvisibility=hidden $(INCLUDES) -c $< -o $@

$(BUILD_DIR)/t/%.o: t/%.c
	@mkdir -p $(dir $@)
	$(CC) $(CPPFLAGS) $(CFLAGS) $(DEPFLAGS) $(INCLUDES) -c $< -o $@

$(BUILD_DIR)/examples/%.o: examples/%.c
	@mkdir -p $(dir $@)
	$(CC) $(CPPFLAGS) $(CFLAGS) $(DEPFLAGS) $(INCLUDES) -c $< -o $@

$(VK_STATIC_LIB): $(VK_OBJ)
	@mkdir -p $(dir $@)
	$(AR) rcs $@ $(VK_OBJ)

$(CORE_STATIC_LIB): $(CORE_LIB_OBJ)
	@mkdir -p $(dir $@)
	$(AR) rcs $@ $(CORE_LIB_OBJ)

$(NCCL_SO): $(NCCL_PLUGIN_OBJ) $(VK_OBJ) $(NANORQ_CORE_OBJ)
	@mkdir -p $(dir $@)
	$(CC) $(CFLAGS) -fPIC -fvisibility=hidden -shared $(LDFLAGS) \
		-o $@ $(NCCL_PLUGIN_OBJ) $(VK_OBJ) $(NANORQ_CORE_OBJ) $(VULKAN_LIBS) -ldl -pthread

$(OUT_PREFIX)bench_vulkan: $(BUILD_DIR)/t/bench_vulkan.o $(CORE_STATIC_LIB)
	@mkdir -p $(dir $@)
	$(CC) $(CFLAGS) $(LDFLAGS) -o $@ $(BUILD_DIR)/t/bench_vulkan.o $(CORE_STATIC_LIB) $(VULKAN_LIBS) -pthread $(LDLIBS)

$(OUT_PREFIX)bench_pipeline: $(BUILD_DIR)/t/bench_pipeline.o $(CORE_STATIC_LIB)
	@mkdir -p $(dir $@)
	$(CC) $(CFLAGS) $(LDFLAGS) -o $@ $(BUILD_DIR)/t/bench_pipeline.o $(CORE_STATIC_LIB) $(VULKAN_LIBS) -pthread $(LDLIBS)

$(OUT_PREFIX)bench_nccl: $(BUILD_DIR)/t/bench_nccl.o $(NCCL_SO)
	@mkdir -p $(dir $@)
	$(CC) $(CFLAGS) $(LDFLAGS) -o $@ $(BUILD_DIR)/t/bench_nccl.o -ldl -pthread $(LDLIBS)

$(OUT_PREFIX)bench_wire_net: $(BUILD_DIR)/t/bench_wire_net.o $(CORE_STATIC_LIB)
	@mkdir -p $(dir $@)
	$(CC) $(CFLAGS) $(LDFLAGS) -o $@ $(BUILD_DIR)/t/bench_wire_net.o $(CORE_STATIC_LIB) $(VULKAN_LIBS) -ldl -pthread $(LDLIBS)

$(OUT_PREFIX)demo_nccl_e2e: $(BUILD_DIR)/examples/demo_nccl_e2e.o $(NCCL_SO)
	@mkdir -p $(dir $@)
	$(CC) $(CFLAGS) $(LDFLAGS) -o $@ $(BUILD_DIR)/examples/demo_nccl_e2e.o -ldl -lm $(LDLIBS)

$(OUT_PREFIX)t/test_vk_codecs: $(BUILD_DIR)/t/test_vk_codecs.o $(CORE_STATIC_LIB)
	@mkdir -p $(dir $@)
	$(CC) $(CFLAGS) $(LDFLAGS) -o $@ $(BUILD_DIR)/t/test_vk_codecs.o $(CORE_STATIC_LIB) $(VULKAN_LIBS) -pthread $(LDLIBS)

$(OUT_PREFIX)t/test_interleave_ring: $(BUILD_DIR)/t/test_interleave_ring.o $(CORE_STATIC_LIB)
	@mkdir -p $(dir $@)
	$(CC) $(CFLAGS) $(LDFLAGS) -o $@ $(BUILD_DIR)/t/test_interleave_ring.o $(CORE_STATIC_LIB) $(VULKAN_LIBS) -pthread $(LDLIBS)

$(OUT_PREFIX)t/test_nccl_plugin: $(BUILD_DIR)/t/test_nccl_plugin.o $(CORE_STATIC_LIB) $(NCCL_SO)
	@mkdir -p $(dir $@)
	$(CC) $(CFLAGS) $(LDFLAGS) -rdynamic \
		-o $@ $(BUILD_DIR)/t/test_nccl_plugin.o $(CORE_STATIC_LIB) $(VULKAN_LIBS) -ldl -pthread $(LDLIBS)

shaders:
	python3 vk/shaders/gen_spv.py

check-vk: $(OUT_PREFIX)t/test_vk_codecs
	./$(OUT_PREFIX)t/test_vk_codecs

check-interleave: $(OUT_PREFIX)t/test_interleave_ring
	./$(OUT_PREFIX)t/test_interleave_ring

check-nccl: $(OUT_PREFIX)t/test_nccl_plugin $(OUT_PREFIX)bench_nccl $(NCCL_SO)
	VXFEC_NCCL_SO_PATH="$(abspath $(NCCL_SO))" ./$(OUT_PREFIX)t/test_nccl_plugin
	VXFEC_NCCL_SO_PATH="$(abspath $(NCCL_SO))" ./$(OUT_PREFIX)t/test_nccl_plugin --rq-async
	VXFEC_NCCL_SO_PATH="$(abspath $(NCCL_SO))" ./$(OUT_PREFIX)t/test_nccl_plugin --ibv-mock
	VXFEC_NCCL_SO_PATH="$(abspath $(NCCL_SO))" ./$(OUT_PREFIX)bench_nccl --self-test

check-wire-net: $(OUT_PREFIX)bench_wire_net
	./$(OUT_PREFIX)bench_wire_net --self-test

check: check-vk check-interleave check-nccl check-wire-net

bench: $(OUT_PREFIX)bench_vulkan
	./$(OUT_PREFIX)bench_vulkan

bench-pipeline: $(OUT_PREFIX)bench_pipeline
	./$(OUT_PREFIX)bench_pipeline

bench-nccl: $(OUT_PREFIX)bench_nccl $(NCCL_SO)
	VXFEC_NCCL_SO_PATH="$(abspath $(NCCL_SO))" ./$(OUT_PREFIX)bench_nccl

bench-wire-net: $(OUT_PREFIX)bench_wire_net
	./$(OUT_PREFIX)bench_wire_net --self-test

demo-nccl: $(OUT_PREFIX)demo_nccl_e2e $(NCCL_SO)
	VXFEC_NCCL_SO_PATH="$(abspath $(NCCL_SO))" ./$(OUT_PREFIX)demo_nccl_e2e

asan:
	$(RM) -r $(ASAN_DIR)
	LSAN_OPTIONS="$${LSAN_OPTIONS:+$$LSAN_OPTIONS:}suppressions=$(LSAN_SUPP)" \
	$(MAKE) CC=clang ARCH_FLAGS="" BUILD_DIR=$(ASAN_DIR) OUT_PREFIX=$(ASAN_DIR)/ \
		CFLAGS="-O1 -g -std=c11 -Wall -Wextra -fPIC -fvisibility=hidden -fsanitize=address,undefined -Wno-unused-parameter" \
		LDFLAGS="-fsanitize=address,undefined" \
		all check

indent:
	clang-format -style=file -i include/*.h vk/vk_internal.h vk/*.c interleave/*.c transport/*.h transport/*.c plugins/nccl/*.h plugins/nccl/*.c t/*.h t/*.c examples/*.c

clean:
	$(RM) -r $(BUILD_DIR) $(ASAN_DIR) $(VK_STATIC_LIB) $(CORE_STATIC_LIB) $(NCCL_SO) $(BENCH_BINS) $(DEMO_BINS) $(TEST_BINS)

-include $(wildcard $(BUILD_DIR)/*/*.d)
