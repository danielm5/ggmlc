// Standalone test driver for generated ggmlc models.
//
// Compiled once per test against a generated "<Model>.h":
//   c++ ... standalone_driver.cpp -DTEST_MODEL=<Model> -DTEST_HEADER=<Model>.h ...
//
// Usage:
//   test_run <weights.gguf> <output.bin> <n_inputs>
//            [<name> <ne0> <ne1> <ne2> <ne3> <in.bin>]...
//
// Single-output float32 graphs only; the last cgraph node is dumped.

#define STR_(x) #x
#define STR(x) STR_(x)

#include STR(TEST_HEADER)

#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <unordered_map>
#include <vector>

static std::vector<uint8_t> read_bin(const char * path) {
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    if (!f.is_open()) {
        std::cerr << "cannot open " << path << std::endl;
        exit(1);
    }
    size_t n = (size_t) f.tellg();
    f.seekg(0);
    std::vector<uint8_t> b(n);
    f.read((char *) b.data(), n);
    return b;
}

int main(int argc, char ** argv) {
    if (argc < 4) {
        std::cerr << "usage: test_run <weights.gguf> <output.bin> <n_inputs> [...]" << std::endl;
        return 1;
    }
    const char * gguf_path = argv[1];
    const char * out_path = argv[2];
    int n_inputs = std::atoi(argv[3]);

    ggml_backend_t backend = ggml_backend_cpu_init();
    ggml_init_params p{128 * 1024 * 1024, nullptr, true};
    ggml_context * ctx = ggml_init(p);

    TEST_MODEL::Weights weights;
    gguf_init_params gp{true, nullptr};
    gguf_context * gctx = gguf_init_from_file(gguf_path, gp);
    if (!gctx) {
        std::cerr << "cannot open gguf" << std::endl;
        return 1;
    }
    weights.init_tensors(ctx, gctx);

    std::unordered_map<std::string, ggml_tensor *> inputs;
    std::vector<std::pair<ggml_tensor *, std::vector<uint8_t>>> feeds;
    int a = 4;
    for (int i = 0; i < n_inputs; ++i) {
        const char * name = argv[a++];
        int64_t ne0 = std::atoll(argv[a++]);
        int64_t ne1 = std::atoll(argv[a++]);
        int64_t ne2 = std::atoll(argv[a++]);
        int64_t ne3 = std::atoll(argv[a++]);
        const char * bin_path = argv[a++];
        ggml_tensor * t = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, ne0, ne1, ne2, ne3);
        inputs[name] = t;
        feeds.emplace_back(t, read_bin(bin_path));
    }
    ggml_cgraph * gf = TEST_MODEL::build_graph(ctx, weights, inputs);

    // Graph outputs must be host-readable: a strided view's memory span is
    // not its logical payload, so materialize it before the linear copy below.
    ggml_tensor * out = ggml_graph_node(gf, ggml_graph_n_nodes(gf) - 1);
    if (!ggml_is_contiguous(out)) {
        out = ggml_cont(ctx, out);
        ggml_build_forward_expand(gf, out);
    }

    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, backend);
    if (!buf) {
        std::cerr << "backend alloc failed" << std::endl;
        return 1;
    }
    (void) buf;
    weights.load_data(gctx, gguf_path);
    for (auto & feed : feeds) {
        ggml_backend_tensor_set(feed.first, feed.second.data(), 0, feed.second.size());
    }

    if (ggml_backend_graph_compute(backend, gf) != GGML_STATUS_SUCCESS) {
        std::cerr << "compute failed" << std::endl;
        return 1;
    }
    std::vector<uint8_t> ob(ggml_nbytes(out));
    ggml_backend_tensor_get(out, ob.data(), 0, ob.size());
    std::ofstream o(out_path, std::ios::binary);
    o.write((char *) ob.data(), ob.size());
    const float * vals = (const float *) ob.data();
    std::cout << "output elements: " << (ob.size() / 4) << " first values:";
    for (size_t k = 0; k < ob.size() / 4 && k < 8; ++k) {
        std::cout << " " << vals[k];
    }
    std::cout << std::endl;
    return 0;
}
