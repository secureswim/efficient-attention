#include "llama.h"
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

using Clock = std::chrono::steady_clock;
using Ms = std::chrono::duration<double, std::milli>;
struct Stream {
    std::string name, prompt, text;
    std::vector<llama_token> input, output;
    int pos = 0;
    int logits_index = -1;
    bool done = false;
};
struct Run {
    std::vector<Stream> streams;
    double load_ms = 0, tokenize_ms = 0, prefill_ms = 0, generation_ms = 0, total_ms = 0;
};
static void require(bool ok, const char * message) { if (!ok) throw std::runtime_error(message); }
static std::vector<llama_token> tokenize(const llama_vocab * vocab, const std::string & s) {
    int n = llama_tokenize(vocab, s.c_str(), (int) s.size(), nullptr, 0, true, false);
    require(n < 0 && n != INT32_MIN, "token count failed");
    std::vector<llama_token> tokens(-n);
    require(llama_tokenize(vocab, s.c_str(), (int) s.size(), tokens.data(), (int) tokens.size(), true, false) == (int) tokens.size(), "tokenization failed");
    return tokens;
}
static std::string piece(const llama_vocab * vocab, llama_token token) {
    char buf[256];
    int n = llama_token_to_piece(vocab, token, buf, sizeof(buf), 0, false);
    if (n < 0) {
        std::vector<char> larger(-n);
        n = llama_token_to_piece(vocab, token, larger.data(), (int) larger.size(), 0, false);
        require(n >= 0, "token to piece failed");
        return {larger.data(), (size_t) n};
    }
    return {buf, (size_t) n};
}
static void add(llama_batch & b, int i, llama_token token, int pos, int seq, bool logits) {
    b.token[i] = token;
    b.pos[i] = pos;
    b.n_seq_id[i] = 1;
    b.seq_id[i][0] = seq;
    b.logits[i] = logits ? 1 : 0;
    b.n_tokens = i + 1;
}
static llama_token greedy(llama_context * ctx, int index, int n_vocab) {
    const float * p = llama_get_logits_ith(ctx, index);
    require(p != nullptr, "missing logits for requested batch index");
    return (llama_token) (std::max_element(p, p + n_vocab) - p);
}
static Run execute(llama_model * model, std::vector<Stream> streams, int max_tokens, int threads) {
    Run result; result.streams = std::move(streams);
    const auto total_start = Clock::now();
    const llama_vocab * vocab = llama_model_get_vocab(model);
    const auto tok_start = Clock::now();
    int prompt_total = 0;
    for (auto & s : result.streams) {
        s.input = tokenize(vocab, s.prompt);
        require(!s.input.empty(), "empty prompt tokens");
        prompt_total += (int) s.input.size();
    }
    result.tokenize_ms = Ms(Clock::now() - tok_start).count();
    llama_context_params cp = llama_context_default_params();
    cp.n_ctx = 512;
    cp.n_batch = 512;
    cp.n_ubatch = 512;
    cp.n_seq_max = 2; // Keep context configuration identical in separate and batched runs.
    cp.kv_unified = false;
    cp.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_DISABLED;
    cp.n_threads = threads;
    cp.n_threads_batch = threads;
    llama_context * ctx = llama_init_from_model(model, cp);
    require(ctx != nullptr, "context creation failed");
    llama_batch b = llama_batch_init(512, 0, 1);
    try {
        require(prompt_total + max_tokens * (int) result.streams.size() < 512, "context too small");
        int bi = 0;
        for (int si = 0; si < (int) result.streams.size(); ++si) {
            auto & s = result.streams[si];
            for (int j = 0; j < (int) s.input.size(); ++j) {
                add(b, bi, s.input[j], j, si, j == (int) s.input.size() - 1);
                if (j == (int) s.input.size() - 1) s.logits_index = bi;
                ++bi;
            }
            s.pos = (int) s.input.size();
        }
        const auto prefill_start = Clock::now();
        require(llama_decode(ctx, b) == 0, "prefill llama_decode failed");
        result.prefill_ms = Ms(Clock::now() - prefill_start).count();
        const auto generation_start = Clock::now();
        for (int step = 0; step < max_tokens; ++step) {
            // Sample all logits before the next decode overwrites the output buffer.
            std::vector<llama_token> next(result.streams.size(), LLAMA_TOKEN_NULL);
            for (int si = 0; si < (int) result.streams.size(); ++si) {
                auto & s = result.streams[si];
                if (s.done) continue;
                if (s.name == "B" && step == 3) {
                    const float * scores = llama_get_logits_ith(ctx, s.logits_index);
                    require(scores != nullptr, "missing diagnostic logits");
                    int top = -1, second = -1;
                    for (int id = 0; id < llama_vocab_n_tokens(vocab); ++id) {
                        if (top < 0 || scores[id] > scores[top]) { second = top; top = id; }
                        else if (second < 0 || scores[id] > scores[second]) second = id;
                    }
                    std::cout << "LOGITS_B_STEP3\tstreams=" << result.streams.size() << "\tseq=" << si
                              << "\ttop=" << top << ':' << scores[top] << "\tsecond=" << second << ':' << scores[second] << "\n";
                }
                next[si] = greedy(ctx, s.logits_index, llama_vocab_n_tokens(vocab));
            }
            b.n_tokens = 0;
            for (int si = 0; si < (int) result.streams.size(); ++si) {
                auto & s = result.streams[si];
                if (s.done) continue;
                if (llama_vocab_is_eog(vocab, next[si])) { s.done = true; continue; }
                s.output.push_back(next[si]);
                s.text += piece(vocab, next[si]);
                if (step + 1 == max_tokens) { s.done = true; continue; }
                s.logits_index = b.n_tokens;
                add(b, b.n_tokens, next[si], s.pos++, si, true);
            }
            if (b.n_tokens == 0) break;
            require(llama_decode(ctx, b) == 0, "generation llama_decode failed");
        }
        result.generation_ms = Ms(Clock::now() - generation_start).count();
        for (const auto & s : result.streams) require(!s.output.empty(), "sequence generated no text tokens");
    } catch (...) {
        llama_batch_free(b); llama_free(ctx); throw;
    }
    llama_batch_free(b); llama_free(ctx);
    result.total_ms = Ms(Clock::now() - total_start).count();
    return result;
}
static void print(const char * mode, const Run & r, int threads) {
    for (const auto & s : r.streams) {
        std::cout << "RUN\t" << mode << "\t" << s.name << "\tthreads=" << threads
                  << "\tprompt_tokens=" << s.input.size() << "\tgenerated_tokens=" << s.output.size() << "\nIDS";
        for (auto id : s.output) std::cout << ' ' << id;
        std::cout << "\nTEXT " << s.text << "\n";
    }
    std::cout << "TIMING\t" << mode << "\ttokenize_ms=" << r.tokenize_ms
              << "\tprefill_ms=" << r.prefill_ms << "\tgeneration_ms=" << r.generation_ms
              << "\ttotal_ms=" << r.total_ms << "\n";
}
static bool compare(const Stream & a, const Stream & b, const char * label) {
    size_t first = 0;
    while (first < a.output.size() && first < b.output.size() && a.output[first] == b.output[first]) ++first;
    const bool same = first == a.output.size() && first == b.output.size() && a.text == b.text;
    std::cout << "COMPARE\t" << label << "\t" << (same ? "MATCH" : "MISMATCH")
              << "\tfirst_different_token=" << (same ? -1 : (int) first) << "\n";
    return same;
}
int main(int argc, char ** argv) {
    if (argc < 2 || argc > 7 || argc == 6) {
        std::cerr << "usage: two-prompt MODEL [MAX_TOKENS=24] [THREADS=4] [--reverse] [PROMPT_A PROMPT_B]\n"; return 2;
    }
    const int max_tokens = argc > 2 ? std::atoi(argv[2]) : 24;
    const int threads = argc > 3 ? std::atoi(argv[3]) : 4;
    const bool reverse = argc > 4 && std::string(argv[4]) == "--reverse";
    if (max_tokens < 1 || max_tokens > 128 || threads < 1 || threads > 64 || (argc > 4 && !reverse)) return 2;
    llama_backend_init();
    const auto load_start = Clock::now();
    llama_model_params mp = llama_model_default_params();
    mp.n_gpu_layers = 0;
    llama_model * model = llama_model_load_from_file(argv[1], mp);
    if (!model) { std::cerr << "model loading failed\n"; llama_backend_free(); return 2; }
    const double load_ms = Ms(Clock::now() - load_start).count();
    std::cout << "MODEL_LOAD_MS " << load_ms << "\n";
    int status = 0;
    try {
        Stream a; a.name = "A"; a.prompt = argc == 7 ? argv[5] : "Explain in three sentences how a CPU cache works.";
        Stream b; b.name = "B"; b.prompt = argc == 7 ? argv[6] : "Explain in three sentences how an operating system schedules processes.";
        Run single_a = execute(model, {a}, max_tokens, threads);
        Run single_b = execute(model, {b}, max_tokens, threads);
        Run batched = execute(model, reverse ? std::vector<Stream>{b, a} : std::vector<Stream>{a, b}, max_tokens, threads);
        print("single", single_a, threads); print("single", single_b, threads); print(reverse ? "batched_reverse" : "batched", batched, threads);
        const Stream & ba = batched.streams[reverse ? 1 : 0];
        const Stream & bb = batched.streams[reverse ? 0 : 1];
        const bool match_a = compare(single_a.streams[0], ba, "A");
        const bool match_b = compare(single_b.streams[0], bb, "B");
        if (!match_a || !match_b) status = 1;
    } catch (const std::exception & e) { std::cerr << "ERROR " << e.what() << "\n"; status = 2; }
    llama_model_free(model); llama_backend_free();
    return status;
}
