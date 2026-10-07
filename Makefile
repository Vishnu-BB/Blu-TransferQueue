# ============================================================
# Blu-TransferQueue Makefile
# ============================================================
# C++ port of https://github.com/Ascend/TransferQueue, nested inside
# BluTrain as its own submodule (separate history/ownership). ../dist and
# ../Tensor-Implementations are BluTrain siblings on disk -- no third_party
# checkout, no submodule-of-submodule.
#
# Five build lanes, by dependency weight:
#   make smoke        Dependency-free: BatchMeta/Controller/Message/Sampler/
#                      GroupRouter. Plain g++, no BluTrain headers at all.
#   make test_tensor   Needs Tensor-Implementations (Record now holds a real
#                      OwnTensor::Tensor): StorageManager + Client on top of
#                      `smoke`'s pieces. No MPI/NCCL/dist/zmq needed.
#   make test_rpc      Needs Tensor-Implementations + libzmq3-dev/cppzmq-dev:
#                      Server (ROUTER) + RpcClient (DEALER) on top of
#                      `test_tensor`'s pieces. Still no MPI/NCCL/dist.
#   make test_distributed   Needs MPI (mpic++, `mpirun -np N`) on top of
#                      `test_rpc`'s pieces: each rank runs its own local
#                      Server, GroupRouter decides which rank owns a sample.
#                      Still no dist/communication/NCCL -- GroupRouter is
#                      plain hash % world_size (see docs/PHASE_4.md).
#   make deps + make lib   Needs the rest of BluTrain (dist/communication +
#                      dist/Tensor-Parallelism, CUDA/NCCL/MPI): pulls in
#                      Transport on top of `test_tensor`'s pieces (not
#                      Server/RpcClient yet -- add when the library actually
#                      needs to be reachable over sockets).
#   make clean
# ============================================================

BLUTRAIN_DIR     := ..
DIST_DIR         := $(BLUTRAIN_DIR)/dist
TENSOR_IMPL_DIR  := $(BLUTRAIN_DIR)/Tensor-Implementations
PROFILER_DIR     := $(BLUTRAIN_DIR)/Profiler

BUILD_DIR := build
LIB_DIR   := lib

CXX      := g++
MPICXX   := $(shell which mpic++ 2>/dev/null)
CXXFLAGS := -std=c++2a -O2 -g -Iinclude

# CUDA auto-detect (g++/mpic++ don't know about it by default; matches
# dist/Makefile's own detection).
CUDA_NVCC := $(shell which nvcc 2>/dev/null)
CUDA_HOME := $(if $(CUDA_NVCC),$(dir $(patsubst %/,%,$(dir $(CUDA_NVCC)))),/usr/local/cuda/)
CUDA_INC  := $(CUDA_HOME)include
CUDA_LIB  := $(CUDA_HOME)lib64

# ---- Lane 1: smoke -- no BluTrain deps at all ----
CORE_SRCS := src/BatchMeta.cpp src/Message.cpp src/PartitionIndexManager.cpp src/DataPartitionStatus.cpp \
             src/Controller.cpp src/Sampler.cpp src/GRPOGroupNSampler.cpp src/GroupRouter.cpp

.PHONY: smoke test_tensor test_rpc test_correctness test_distributed test_distributed_n_to_m server_bin controller_server_bin storage_server_bin demo_pipeline unit_tests deps lib clean

smoke: $(BUILD_DIR)/test_scaffold_smoke
	./$(BUILD_DIR)/test_scaffold_smoke

$(BUILD_DIR)/test_scaffold_smoke: $(CORE_SRCS) tests/test_scaffold_smoke.cpp
	@mkdir -p $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) $^ -o $@

# ---- Lane 2: test_tensor -- Record holds a real OwnTensor::Tensor, so
#      StorageManager/Client need Tensor-Implementations' headers + compiled
#      lib (libtensor.a). No MPI/NCCL/dist -- those are Lane 3 only. ----
TENSOR_SRCS      := src/StorageManager.cpp src/Client.cpp
TENSOR_TEST_SRCS := $(CORE_SRCS) $(TENSOR_SRCS) tests/test_tensor_smoke.cpp

TENSOR_INCLUDES := -I$(TENSOR_IMPL_DIR)/include -I$(CUDA_INC)
# Mirrors Tensor-Implementations/Makefile's own TENSOR_LDLIBS -- libtensor.a
# pulls in its CUDA allocator/NVML code even for CPU-only tensor use.
TENSOR_LDFLAGS := -L$(TENSOR_IMPL_DIR)/lib -L$(CUDA_LIB) -L$(PROFILER_DIR)/lib \
                   -Wl,-rpath,$(TENSOR_IMPL_DIR)/lib -Wl,-rpath,$(CUDA_LIB) -Wl,-rpath,$(PROFILER_DIR)/lib
TENSOR_LDLIBS  := -ltensor -lblublas -lcudart -lcurand -lcublas -lcublasLt -lgomp -lnvidia-ml -lprofiler -lcuda -lstdc++fs -pthread

test_tensor: $(BUILD_DIR)/test_tensor_smoke
	./$(BUILD_DIR)/test_tensor_smoke

$(BUILD_DIR)/test_tensor_smoke: $(TENSOR_TEST_SRCS)
	@mkdir -p $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) $(TENSOR_INCLUDES) -DWITH_CUDA $(TENSOR_TEST_SRCS) $(TENSOR_LDFLAGS) $(TENSOR_LDLIBS) -o $@

# ---- Lane 3 (test_rpc): Server (ROUTER) + RpcClient (DEALER) over real
#      cppzmq sockets, on top of Lane 2's Tensor-Implementations deps. No
#      MPI/NCCL/dist. ----
RPC_SRCS      := src/RpcServerBase.cpp src/Server.cpp src/RpcClient.cpp src/ControllerServer.cpp src/StorageServer.cpp
RPC_TEST_SRCS := $(CORE_SRCS) $(TENSOR_SRCS) $(RPC_SRCS) tests/test_rpc_smoke.cpp
RPC_LDLIBS    := -lzmq -pthread

test_rpc: $(BUILD_DIR)/test_rpc_smoke
	./$(BUILD_DIR)/test_rpc_smoke

$(BUILD_DIR)/test_rpc_smoke: $(RPC_TEST_SRCS)
	@mkdir -p $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) $(TENSOR_INCLUDES) -DWITH_CUDA $(RPC_TEST_SRCS) $(TENSOR_LDFLAGS) $(TENSOR_LDLIBS) $(RPC_LDLIBS) -o $@

# ---- End-to-End Pipeline Correctness Test ----
CORRECTNESS_TEST_SRCS := $(CORE_SRCS) $(TENSOR_SRCS) $(RPC_SRCS) tests/test_pipeline_correctness.cpp

test_correctness: $(BUILD_DIR)/test_pipeline_correctness
	./$(BUILD_DIR)/test_pipeline_correctness

$(BUILD_DIR)/test_pipeline_correctness: $(CORRECTNESS_TEST_SRCS)
	@mkdir -p $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) $(TENSOR_INCLUDES) -DWITH_CUDA $(CORRECTNESS_TEST_SRCS) $(TENSOR_LDFLAGS) $(TENSOR_LDLIBS) $(RPC_LDLIBS) -o $@

# ---- Lane 4 (test_distributed): write-time group-affinity routing across
#      real MPI ranks. Needs mpic++ + `mpirun -np N` on top of Lane 3's
#      deps -- no dist/communication/NCCL/DeviceMesh (GroupRouter doesn't
#      need them). NPROC overrides the rank count (default 2, to actually
#      exercise a cross-process route). ----
NPROC ?= 2
DISTRIBUTED_TEST_SRCS := $(CORE_SRCS) $(TENSOR_SRCS) $(RPC_SRCS) tests/test_distributed_smoke.cpp

test_distributed: $(BUILD_DIR)/test_distributed_smoke
	mpirun --allow-run-as-root -np $(NPROC) ./$(BUILD_DIR)/test_distributed_smoke

# ---- Standalone server binary: not a test (no fixed sequence, no exit) --
#      for launching via an external process manager (Ray, a shell script,
#      ...). Same deps as Lane 3, nothing more; this binary itself knows
#      nothing about whatever launches it. ----
SERVER_BIN_SRCS := $(CORE_SRCS) $(TENSOR_SRCS) $(RPC_SRCS) tools/transferqueue_server_main.cpp

server_bin: $(BUILD_DIR)/transferqueue_server

$(BUILD_DIR)/transferqueue_server: $(SERVER_BIN_SRCS)
	@mkdir -p $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) $(TENSOR_INCLUDES) -DWITH_CUDA $(SERVER_BIN_SRCS) $(TENSOR_LDFLAGS) $(TENSOR_LDLIBS) $(RPC_LDLIBS) -o $@

# ---- Phase 5 standalone binaries: ControllerServer (metadata only) and
#      StorageServer (data only), split apart for a real multi-node/
#      multi-shard deployment. Same deps as the colocated server_bin. ----
CONTROLLER_BIN_SRCS := $(CORE_SRCS) $(TENSOR_SRCS) $(RPC_SRCS) tools/controller_server_main.cpp
STORAGE_BIN_SRCS    := $(CORE_SRCS) $(TENSOR_SRCS) $(RPC_SRCS) tools/storage_server_main.cpp

controller_server_bin: $(BUILD_DIR)/controller_server

$(BUILD_DIR)/controller_server: $(CONTROLLER_BIN_SRCS)
	@mkdir -p $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) $(TENSOR_INCLUDES) -DWITH_CUDA $(CONTROLLER_BIN_SRCS) $(TENSOR_LDFLAGS) $(TENSOR_LDLIBS) $(RPC_LDLIBS) -o $@

storage_server_bin: $(BUILD_DIR)/storage_server

$(BUILD_DIR)/storage_server: $(STORAGE_BIN_SRCS)
	@mkdir -p $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) $(TENSOR_INCLUDES) -DWITH_CUDA $(STORAGE_BIN_SRCS) $(TENSOR_LDFLAGS) $(TENSOR_LDLIBS) $(RPC_LDLIBS) -o $@

# ---- Dummy-data pipeline walkthrough: not a test, prints every stage's
#      input/output for docs/PIPELINE_WALKTHROUGH.md. Same deps as Lane 3. ----
DEMO_PIPELINE_SRCS := $(CORE_SRCS) $(TENSOR_SRCS) $(RPC_SRCS) tools/demo_pipeline_main.cpp

demo_pipeline: $(BUILD_DIR)/demo_pipeline
	./$(BUILD_DIR)/demo_pipeline

$(BUILD_DIR)/demo_pipeline: $(DEMO_PIPELINE_SRCS)
	@mkdir -p $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) $(TENSOR_INCLUDES) -DWITH_CUDA $(DEMO_PIPELINE_SRCS) $(TENSOR_LDFLAGS) $(TENSOR_LDLIBS) $(RPC_LDLIBS) -o $@

# ============================================================
# Per-component unit tests (tests/unit/*Test.cpp) -- one file per component,
# assert()-based like every other test here (no new test-framework
# dependency: gtest isn't installed and every existing test is already
# plain assert() + plain g++/mpic++). tests/unit/TestUtils.h adds CHECK()/
# CHECK_THROWS()/CHECK_NOTHROW() macros that keep running and report every
# failure in one pass, instead of bare assert()'s abort-on-first-failure.
#
# Three pattern rules below, one per dependency tier (same tiers as the
# four build lanes above) -- the target name's unit_core_/unit_tensor_/
# unit_rpc_ prefix picks which tier a given component needs.
# ============================================================
$(BUILD_DIR)/unit_core_%: tests/unit/%.cpp $(CORE_SRCS) tests/unit/TestUtils.h
	@mkdir -p $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) $(CORE_SRCS) $< -o $@

$(BUILD_DIR)/unit_tensor_%: tests/unit/%.cpp $(CORE_SRCS) $(TENSOR_SRCS) tests/unit/TestUtils.h
	@mkdir -p $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) $(TENSOR_INCLUDES) -DWITH_CUDA $(CORE_SRCS) $(TENSOR_SRCS) $< $(TENSOR_LDFLAGS) $(TENSOR_LDLIBS) -o $@

$(BUILD_DIR)/unit_rpc_%: tests/unit/%.cpp $(CORE_SRCS) $(TENSOR_SRCS) $(RPC_SRCS) tests/unit/TestUtils.h
	@mkdir -p $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) $(TENSOR_INCLUDES) -DWITH_CUDA $(CORE_SRCS) $(TENSOR_SRCS) $(RPC_SRCS) $< $(TENSOR_LDFLAGS) $(TENSOR_LDLIBS) $(RPC_LDLIBS) -o $@

UNIT_CORE_TESTS   := BatchMetaTest PartitionIndexManagerTest DataPartitionStatusTest ControllerTest \
                     MessageTest FifoSamplerTest GRPOGroupNSamplerTest GroupRouterTest
UNIT_TENSOR_TESTS := StorageManagerTest ClientTest
UNIT_RPC_TESTS    := ServerTest RpcClientTest ControllerServerTest StorageServerTest

UNIT_TEST_BINS := $(addprefix $(BUILD_DIR)/unit_core_,$(UNIT_CORE_TESTS)) \
                  $(addprefix $(BUILD_DIR)/unit_tensor_,$(UNIT_TENSOR_TESTS)) \
                  $(addprefix $(BUILD_DIR)/unit_rpc_,$(UNIT_RPC_TESTS))

unit_tests: $(UNIT_TEST_BINS)
	@failed=0; \
	for t in $(UNIT_TEST_BINS); do \
		echo "--- $$(basename $$t) ---"; \
		./$$t || failed=1; \
	done; \
	exit $$failed

$(BUILD_DIR)/test_distributed_smoke: $(DISTRIBUTED_TEST_SRCS)
	@mkdir -p $(BUILD_DIR)
	$(MPICXX) $(CXXFLAGS) $(TENSOR_INCLUDES) -DWITH_CUDA $(DISTRIBUTED_TEST_SRCS) $(TENSOR_LDFLAGS) $(TENSOR_LDLIBS) $(RPC_LDLIBS) -o $@

# ---- Phase 7 (test_distributed_n_to_m): a real N:M rollout:trainer
#      topology via MPI_Comm_split -- rank 0 runs a ControllerServer, the
#      next N_ROLLOUT ranks each run a StorageServer (shard), the remaining
#      N_TRAINER ranks are pure readers sharing one task_name. Same deps as
#      Lane 4, same deterministic-port simplification. See
#      tests/test_distributed_n_to_m_smoke.cpp and docs/PHASE_7.md. ----
N_ROLLOUT ?= 2
N_TRAINER ?= 3
GROUPS_PER_ROLLOUT ?= 3
N_SAMPLES_PER_PROMPT ?= 4
N_TO_M_TOTAL_RANKS := $(shell echo $$(( 1 + $(N_ROLLOUT) + $(N_TRAINER) )))
DISTRIBUTED_N_TO_M_SRCS := $(CORE_SRCS) $(TENSOR_SRCS) $(RPC_SRCS) tests/test_distributed_n_to_m_smoke.cpp

test_distributed_n_to_m: $(BUILD_DIR)/test_distributed_n_to_m_smoke
	mpirun --allow-run-as-root -np $(N_TO_M_TOTAL_RANKS) ./$(BUILD_DIR)/test_distributed_n_to_m_smoke \
		$(N_ROLLOUT) $(N_TRAINER) $(GROUPS_PER_ROLLOUT) $(N_SAMPLES_PER_PROMPT)

$(BUILD_DIR)/test_distributed_n_to_m_smoke: $(DISTRIBUTED_N_TO_M_SRCS)
	@mkdir -p $(BUILD_DIR)
	$(MPICXX) $(CXXFLAGS) $(TENSOR_INCLUDES) -DWITH_CUDA $(DISTRIBUTED_N_TO_M_SRCS) $(TENSOR_LDFLAGS) $(TENSOR_LDLIBS) $(RPC_LDLIBS) -o $@

# ---- Lane 5: BluTrain deps (already sit at ../dist -- just build the two
#      targets) ----
deps:
	$(MAKE) -C $(DIST_DIR) comm tp

# ---- full library: needs RankAwareSampler/Transport, which need dist's
#      communication/Tensor-Parallelism + CUDA/NCCL headers. (RankAwareSampler
#      was removed in Phase 4 -- its read-time-filtering shape didn't match
#      the write-time GroupRouter design; see docs/PHASE_4.md.) ----
DIST_SRCS := src/Transport.cpp
LIB_SRCS  := $(CORE_SRCS) $(TENSOR_SRCS) $(DIST_SRCS)
LIB_OBJS  := $(patsubst src/%.cpp,$(BUILD_DIR)/%.o,$(LIB_SRCS))

LIB_INCLUDES := \
    -I$(DIST_DIR) \
    -I$(DIST_DIR)/communication/include \
    -I$(TENSOR_IMPL_DIR)/include \
    -I$(CUDA_INC)

lib: deps $(LIB_DIR)/libtransferqueue.a

$(BUILD_DIR)/%.o: src/%.cpp
	@mkdir -p $(BUILD_DIR)
	$(MPICXX) $(CXXFLAGS) $(LIB_INCLUDES) -DWITH_CUDA -c $< -o $@

$(LIB_DIR)/libtransferqueue.a: $(LIB_OBJS)
	@mkdir -p $(LIB_DIR)
	ar rcs $@ $(LIB_OBJS)

clean:
	rm -rf $(BUILD_DIR) $(LIB_DIR)
