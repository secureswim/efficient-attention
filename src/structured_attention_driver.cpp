#include "llama.h"
#include "json.hpp"

#include <algorithm>
#include <chrono>
#include <fstream>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

using json = nlohmann::json;
using Clock = std::chrono::steady_clock;
using Ms = std::chrono::duration<double, std::milli>;

static void require(bool ok, const std::string & msg) {
    if (!ok) throw std::runtime_error(msg);
}
static json read_json(const std::string & path) {
    std::ifstream in(path);
    require(in.good(), "cannot open JSON file: " + path);
    json value;
    in >> value;
    return value;
}
static void write_json(const std::string & path, const json & value) {
    std::ofstream out(path);
    require(out.good(), "cannot write JSON file: " + path);
    out << value.dump(2) << '\n';
}
static std::vector<llama_token> tokenize(const llama_vocab * vocab, const std::string & value, bool add_special) {
    int n = llama_tokenize(vocab, value.data(), (int) value.size(), nullptr, 0, add_special, false);
    require(n != INT32_MIN, "token count overflow");
    if (n == 0) return {};
    require(n < 0, "unexpected tokenizer count result");
    std::vector<llama_token> ids(-n);
    int got = llama_tokenize(vocab, value.data(), (int) value.size(), ids.data(), (int) ids.size(), add_special, false);
    require(got == (int) ids.size(), "tokenizer returned inconsistent length");
    return ids;
}
static std::string piece(const llama_vocab * vocab, llama_token id) {
    char short_buf[256];
    int n = llama_token_to_piece(vocab, id, short_buf, sizeof(short_buf), 0, false);
    if (n >= 0) return std::string(short_buf, n);
    std::vector<char> buf(-n);
    n = llama_token_to_piece(vocab, id, buf.data(), (int) buf.size(), 0, false);
    require(n >= 0, "failed to render token");
    return std::string(buf.data(), n);
}
static llama_token greedy(llama_context * ctx, int batch_index, int n_vocab) {
    const float * scores = llama_get_logits_ith(ctx, batch_index);
    require(scores != nullptr, "no logits at requested batch index " + std::to_string(batch_index));
    return (llama_token) (std::max_element(scores, scores + n_vocab) - scores);
}
struct ModelGuard {
    llama_model * model = nullptr;
    explicit ModelGuard(const std::string & path) {
        llama_model_params mp = llama_model_default_params();
        mp.n_gpu_layers = 0;
        model = llama_model_load_from_file(path.c_str(), mp);
        require(model != nullptr, "model loading failed: " + path);
    }
    ~ModelGuard() { if (model) llama_model_free(model); }
};
struct ContextGuard {
    llama_context * ctx = nullptr;
    llama_batch batch{};
    ContextGuard(llama_model * model, int n_seq, int threads) {
        llama_context_params cp = llama_context_default_params();
        cp.n_ctx = 2048;
        cp.n_batch = 2048;
        cp.n_ubatch = 512;
        cp.n_seq_max = (uint32_t) n_seq;
        cp.n_threads = threads;
        cp.n_threads_batch = threads;
        cp.kv_unified = true;
        cp.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_DISABLED;
        ctx = llama_init_from_model(model, cp);
        require(ctx != nullptr, "context creation failed");
        batch = llama_batch_init(2048, 0, n_seq);
        if (!(batch.token && batch.pos && batch.n_seq_id && batch.seq_id && batch.logits)) {
            if (batch.token) llama_batch_free(batch);
            llama_free(ctx);
            ctx = nullptr;
            throw std::runtime_error("batch allocation failed");
        }
    }
    ~ContextGuard() {
        if (batch.token) llama_batch_free(batch);
        if (ctx) llama_free(ctx);
    }
};
static void add(llama_batch & batch, int i, llama_token token, int pos, const std::vector<int> & memberships, bool output) {
    require(i >= 0 && i < 2048, "batch capacity exceeded");
    batch.token[i] = token;
    batch.pos[i] = pos;
    batch.n_seq_id[i] = (int) memberships.size();
    for (int k = 0; k < (int) memberships.size(); ++k) batch.seq_id[i][k] = memberships[k];
    batch.logits[i] = output ? 1 : 0;
    batch.n_tokens = i + 1;
}
static std::vector<int> ids(const json & array) {
    std::vector<int> result;
    for (const auto & item : array) result.push_back(item.get<int>());
    return result;
}
static const json & find_node(const json & doc, std::string_view path) {
    for (const auto & node : doc.at("nodes")) if (node.at("path").get_ref<const std::string &>() == path) return node;
    throw std::runtime_error("missing node path: " + std::string(path));
}
static const json & find_group(const json & doc, std::string_view path) {
    for (const auto & group : doc.at("groups")) if (group.at("path").get_ref<const std::string &>() == path) return group;
    throw std::runtime_error("missing sequence group: " + std::string(path));
}
static void validate(const json & doc) {
    require(doc.at("schema_version") == 1, "unsupported schema version");
    const auto & groups = doc.at("groups");
    require(groups.is_array() && groups.size() >= 2 && groups.size() <= 256, "invalid group count");
    std::vector<bool> found(groups.size(), false);
    for (const auto & group : groups) {
        int seq = group.at("seq_id").get<int>();
        require(seq >= 0 && seq < (int) groups.size() && !found[seq], "invalid or duplicate sequence ID");
        found[seq] = true;
        const auto & node = find_node(doc, group.at("path").get<std::string>());
        require(node.at("id") == group.at("node_id"), "group node ID mismatch");
        require((int) node.at("tokens").size() == group.at("end_pos").get<int>() - group.at("start_pos").get<int>(), "group range does not match token count");
    }
    const auto & root = find_node(doc, "");
    require(root.at("parent").is_null(), "root must have null parent");
    int root_count = (int) root.at("tokens").size();
    require(root_count > 0, "empty shared prefix");
    for (const auto & node : doc.at("nodes")) {
        int n = (int) node.at("tokens").size();
        require(n == (int) node.at("positions").size() && n == (int) node.at("seq_ids").size(), "unaligned token metadata at " + node.at("path").get<std::string>());
        if (!node.at("parent").is_null()) {
            bool parent_found = false;
            for (const auto & candidate : doc.at("nodes")) if (candidate.at("id") == node.at("parent")) parent_found = true;
            require(parent_found, "missing parent node");
        }
        for (int i = 0; i < n; ++i) {
            require(node.at("tokens")[i].get<int>() >= 0, "negative token ID");
            require(!node.at("seq_ids")[i].empty(), "token has no sequence membership");
            for (const auto & seq : node.at("seq_ids")[i]) require(seq.get<int>() >= 0 && seq.get<int>() < (int) groups.size(), "out of range sequence ID");
            if (i) require(node.at("positions")[i].get<int>() == node.at("positions")[i-1].get<int>() + 1, "nonconsecutive node positions");
        }
    }
    for (const auto & group : groups) {
        const auto & node = find_node(doc, group.at("path").get<std::string>());
        require(group.at("start_pos") == root_count, "group does not start after prefix");
        for (int i = 0; i < (int) node.at("tokens").size(); ++i) {
            require(node.at("positions")[i] == root_count + i, "group position mismatch");
            require(node.at("seq_ids")[i].size() == 1 && node.at("seq_ids")[i][0] == group.at("seq_id"), "group membership mismatch");
        }
    }
    for (int i = 0; i < root_count; ++i) {
        require(root.at("positions")[i] == i, "shared prefix position mismatch");
        require(root.at("seq_ids")[i].size() == groups.size(), "shared prefix missing group membership");
    }
    require(doc.at("stats").at("model_token_count").get<int>() < 2000, "document exceeds 2000 model tokens");
}
struct Scenario {
    std::vector<llama_token> generated;
    std::string text;
    double context_ms = 0, document_prefill_ms = 0, eviction_ms = 0, question_prefill_ms = 0, generation_ms = 0, total_ms = 0;
    size_t kv_before_bytes = 0, kv_after_bytes = 0;
    int kv_before_pos = -1, kv_after_pos = -1;
};
static Scenario execute(llama_model * model, const json & doc, const std::string & focus, const std::string & unrelated,
                        const std::string & question, int max_tokens, int threads, bool reverse, bool evict) {
    validate(doc);
    const auto & focus_group = find_group(doc, focus);
    const auto & other_group = find_group(doc, unrelated);
    const int focus_seq = focus_group.at("seq_id").get<int>();
    const int other_seq = other_group.at("seq_id").get<int>();
    require(focus_seq != other_seq, "focus and unrelated subtrees share a sequence");
    Scenario result;
    auto start = Clock::now();
    ContextGuard cg(model, (int) doc.at("groups").size(), threads);
    result.context_ms = Ms(Clock::now() - start).count();
    const auto & root = find_node(doc, "");
    int bi = 0;
    for (int i = 0; i < (int) root.at("tokens").size(); ++i) {
        add(cg.batch, bi++, root.at("tokens")[i].get<int>(), root.at("positions")[i].get<int>(), ids(root.at("seq_ids")[i]), false);
    }
    std::vector<json> groups = doc.at("groups").get<std::vector<json>>();
    if (reverse) std::reverse(groups.begin(), groups.end());
    for (const auto & group : groups) {
        const auto & node = find_node(doc, group.at("path").get<std::string>());
        for (int i = 0; i < (int) node.at("tokens").size(); ++i) {
            add(cg.batch, bi++, node.at("tokens")[i].get<int>(), node.at("positions")[i].get<int>(), ids(node.at("seq_ids")[i]), false);
        }
    }
    require(bi > 0 && bi < 1900, "document prefill exceeds safe capacity");
    auto prefill_start = Clock::now();
    require(llama_decode(cg.ctx, cg.batch) == 0, "document prefill failed");
    result.document_prefill_ms = Ms(Clock::now() - prefill_start).count();
    auto mem = llama_get_memory(cg.ctx);
    require(mem != nullptr, "context has no inference memory");
    result.kv_before_bytes = llama_state_seq_get_size(cg.ctx, other_seq);
    result.kv_before_pos = llama_memory_seq_pos_max(mem, other_seq);
    if (evict) {
        auto eviction_start = Clock::now();
        int p0 = other_group.at("start_pos").get<int>();
        int p1 = other_group.at("end_pos").get<int>();
        require(p0 >= (int) root.at("tokens").size() && p1 > p0, "invalid eviction range");
        require(llama_memory_seq_rm(mem, other_seq, p0, p1), "subtree eviction failed");
        result.eviction_ms = Ms(Clock::now() - eviction_start).count();
    }
    result.kv_after_bytes = llama_state_seq_get_size(cg.ctx, other_seq);
    result.kv_after_pos = llama_memory_seq_pos_max(mem, other_seq);
    if (evict) {
        require(result.kv_after_bytes < result.kv_before_bytes, "live sequence state did not shrink after eviction");
        require(result.kv_after_pos < result.kv_before_pos, "evicted positions still present");
        require(llama_memory_seq_pos_max(mem, focus_seq) >= focus_group.at("end_pos").get<int>() - 1, "focus subtree was disturbed by eviction");
    }
    const llama_vocab * vocab = llama_model_get_vocab(model);
    std::string query = "\nQuestion: " + question + "\nAnswer:";
    auto query_tokens = tokenize(vocab, query, false);
    require(!query_tokens.empty(), "empty question tokenization");
    int pos = focus_group.at("end_pos").get<int>();
    cg.batch.n_tokens = 0;
    for (int i = 0; i < (int) query_tokens.size(); ++i) {
        add(cg.batch, i, query_tokens[i], pos++, {focus_seq}, i == (int) query_tokens.size() - 1);
    }
    auto question_start = Clock::now();
    require(llama_decode(cg.ctx, cg.batch) == 0, "question prefill failed");
    result.question_prefill_ms = Ms(Clock::now() - question_start).count();
    int logits_index = (int) query_tokens.size() - 1;
    auto gen_start = Clock::now();
    for (int step = 0; step < max_tokens; ++step) {
        llama_token next = greedy(cg.ctx, logits_index, llama_vocab_n_tokens(vocab));
        if (llama_vocab_is_eog(vocab, next)) break;
        result.generated.push_back(next);
        std::string next_piece = piece(vocab, next);
        result.text += next_piece;
        size_t line_end = result.text.find_first_of("\r\n");
        if (line_end != std::string::npos) {
            result.text.resize(line_end);
            break;
        }
        if (step + 1 == max_tokens) break;
        cg.batch.n_tokens = 0;
        add(cg.batch, 0, next, pos++, {focus_seq}, true);
        require(llama_decode(cg.ctx, cg.batch) == 0, "generation decode failed");
        logits_index = 0;
    }
    result.generation_ms = Ms(Clock::now() - gen_start).count();
    result.total_ms = Ms(Clock::now() - start).count();
    require(!result.generated.empty() && !result.text.empty(), "sequence generated no valid output");
    return result;
}
static json result_json(const Scenario & s) {
    return {{"ids", s.generated}, {"text", s.text}, {"generated_tokens", s.generated.size()},
            {"context_ms", s.context_ms}, {"document_prefill_ms", s.document_prefill_ms}, {"eviction_ms", s.eviction_ms},
            {"question_prefill_ms", s.question_prefill_ms}, {"generation_ms", s.generation_ms},
            {"total_ms", s.total_ms}, {"kv_before_bytes", s.kv_before_bytes},
            {"kv_after_bytes", s.kv_after_bytes}, {"kv_before_pos", s.kv_before_pos}, {"kv_after_pos", s.kv_after_pos}};
}
static json comparison(const Scenario & base, const Scenario & other) {
    size_t n = 0;
    while (n < base.generated.size() && n < other.generated.size() && base.generated[n] == other.generated[n]) ++n;
    bool match = n == base.generated.size() && n == other.generated.size() && base.text == other.text;
    return {{"match", match}, {"first_different_token", match ? -1 : (int) n}};
}
int main(int argc, char ** argv) {
    if (argc < 2) {
        std::cerr << "usage: structured-attention-driver tokenize MODEL request.json output.json\n"
                     "   or: structured-attention-driver experiment MODEL base.json modified.json FOCUS_PATH UNRELATED_PATH QUESTION MAX_TOKENS THREADS output.json\n";
        return 2;
    }
    llama_backend_init();
    int exit_status = 0;
    try {
        if (std::string(argv[1]) == "tokenize" && argc == 5) {
            ModelGuard model(argv[2]);
            const llama_vocab * vocab = llama_model_get_vocab(model.model);
            auto request = read_json(argv[3]);
            json output = json::array();
            json pieces = json::array();
            for (const auto & fragment : request.at("fragments")) {
                auto tokens = tokenize(vocab, fragment.at("text").get<std::string>(), fragment.value("add_special", false));
                json token_pieces = json::array();
                for (auto id : tokens) token_pieces.push_back(piece(vocab, id));
                output.push_back(tokens);
                pieces.push_back(token_pieces);
            }
            write_json(argv[4], {{"tokens", output}, {"pieces", pieces}});
        } else if (std::string(argv[1]) == "experiment" && argc == 11) {
            const std::string focus = argv[5], unrelated = argv[6], question = argv[7];
            const int max_tokens = std::stoi(argv[8]), threads = std::stoi(argv[9]);
            require(max_tokens > 0 && max_tokens <= 64 && threads > 0 && threads <= 64, "invalid generation or thread count");
            auto base = read_json(argv[3]);
            auto modified = read_json(argv[4]);
            validate(base); validate(modified);
            require(find_group(base, focus).at("seq_id") != find_group(base, unrelated).at("seq_id"),
                    "focus and unrelated subtrees share a sequence");
            require(base.at("document_id") == modified.at("document_id"), "document ID changed");
            require(find_node(base, "").at("tokens") == find_node(modified, "").at("tokens"), "shared prefix changed");
            require(find_node(base, focus).at("tokens") == find_node(modified, focus).at("tokens"), "focus subtree changed");
            require(find_node(base, unrelated).at("tokens") != find_node(modified, unrelated).at("tokens"), "unrelated subtree did not change");
            require(base.at("groups").size() == modified.at("groups").size(), "sequence group count changed");
            for (const auto & group : base.at("groups")) {
                const auto path = group.at("path").get<std::string>();
                const auto & changed_group = find_group(modified, path);
                require(group.at("seq_id") == changed_group.at("seq_id"), "sequence group ID changed");
                if (path != unrelated) {
                    require(find_node(base, path).at("tokens") == find_node(modified, path).at("tokens"),
                            "non-target subtree changed: " + path);
                }
            }
            auto load_start = Clock::now();
            ModelGuard model(argv[2]);
            double load_ms = Ms(Clock::now() - load_start).count();
            Scenario s_base = execute(model.model, base, focus, unrelated, question, max_tokens, threads, false, false);
            Scenario s_changed = execute(model.model, modified, focus, unrelated, question, max_tokens, threads, false, false);
            Scenario s_evicted = execute(model.model, base, focus, unrelated, question, max_tokens, threads, false, true);
            Scenario s_reversed = execute(model.model, base, focus, unrelated, question, max_tokens, threads, true, false);
            Scenario s_repeated = execute(model.model, base, focus, unrelated, question, max_tokens, threads, false, false);
            json changed = comparison(s_base, s_changed);
            json evicted = comparison(s_base, s_evicted);
            json reversed = comparison(s_base, s_reversed);
            json repeated = comparison(s_base, s_repeated);
            json output = {{"document_id", base.at("document_id")}, {"focus", focus}, {"unrelated", unrelated},
                           {"question", question}, {"max_tokens", max_tokens}, {"threads", threads},
                           {"model_load_ms", load_ms}, {"document_tokens", base.at("stats").at("model_token_count")},
                           {"scenarios", {{"baseline", result_json(s_base)}, {"changed_B", result_json(s_changed)},
                                          {"evicted_B", result_json(s_evicted)}, {"reversed", result_json(s_reversed)},
                                          {"repeated", result_json(s_repeated)}}},
                           {"comparisons", {{"changed_B", changed}, {"evicted_B", evicted}, {"reversed", reversed}, {"repeated", repeated}}}};
            write_json(argv[10], output);
            bool pass = changed.at("match") && evicted.at("match") && reversed.at("match") && repeated.at("match");
            std::cout << base.at("document_id").get<std::string>() << ": " << (pass ? "PASS" : "FAIL") << "\n";
            if (!pass) {
                std::cerr << "first differing token: changed=" << changed.at("first_different_token")
                          << ", evicted=" << evicted.at("first_different_token")
                          << ", reversed=" << reversed.at("first_different_token")
                          << ", repeated=" << repeated.at("first_different_token") << "\n";
                exit_status = 1;
            }
        } else {
            throw std::runtime_error("invalid command or argument count");
        }
    } catch (const std::exception & e) {
        std::cerr << "ERROR: " << e.what() << '\n';
        llama_backend_free();
        return 2;
    }
    llama_backend_free();
    return exit_status;
}
